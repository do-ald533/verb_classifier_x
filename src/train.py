from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable

import torch
import torch.nn as nn
import torch.optim as optim

from src.config import LinguisticConfig, N_CLASSES, NetworkConfig, REGULAR_CLASS
from src.data import Dataset, build_dataset
from src.logging_config import get_logger
from src.metrics import wug_metrics
from src.model import build_mlp
from src.pruning import (
    apply_masks,
    count_active_parameters,
    create_masks,
    parameter_norm,
    prune_one_parameter,
)

logger = get_logger(__name__)

MAX_EPOCHS = 100_000
EVAL_EVERY = 1000
EARLY_STOP_PATIENCE_CHECKPOINTS = 5
EARLY_STOP_MIN_DELTA = 1e-4


class TrainingDivergedError(RuntimeError):
    """Raised when training loss becomes non-finite — a genuine failure,
    never conflated with an Optuna pruning decision (plan §1.1 #5). Callers
    that want Optuna to record it as a FAIL should let it propagate into
    study.optimize(..., catch=(RuntimeError,)) rather than catching it here."""


@dataclass
class TrainResult:
    history: list[dict]
    final: dict
    baseline: dict
    wall_clock_seconds: float
    model: nn.Module | None = None
    dataset: Dataset | None = None


def train_one_run(
    linguistic: LinguisticConfig,
    network: NetworkConfig,
    seed: int,
    report_intermediate: Callable[[dict], None] | None = None,
    return_model: bool = False,
) -> TrainResult:
    """Trains one (linguistic, network, seed) configuration to convergence
    (or MAX_EPOCHS). Data-generation and model-init RNGs are explicit and
    decoupled from each other and from the global RNG (plan §8), so trials
    are safe to run in parallel processes and search/confirmation seeds are
    meaningfully independent."""
    start = time.monotonic()

    data_gen = torch.Generator().manual_seed(seed)
    model_gen = torch.Generator().manual_seed(seed + 1_000_000)

    dataset = build_dataset(linguistic, data_gen)
    model = build_mlp(network, input_dim=linguistic.D, n_classes=N_CLASSES, generator=model_gen)
    masks = create_masks(model) if linguistic.pruning_enabled else None
    total_params = sum(m.numel() for m in masks.values()) if masks is not None else None

    model.eval()
    with torch.no_grad():
        baseline_logits = model(dataset.X_wug)
    baseline = {
        f"baseline_{k}": v for k, v in wug_metrics(baseline_logits, REGULAR_CLASS).items()
    }

    criterion = nn.CrossEntropyLoss()
    optimizer: optim.Optimizer
    if network.optimizer_name == "SGD":
        optimizer = optim.SGD(
            model.parameters(), lr=network.learning_rate, momentum=network.momentum
        )
    else:
        optimizer = optim.Adam(model.parameters(), lr=network.learning_rate)

    history: list[dict] = []
    best_loss = float("inf")
    checkpoints_without_improvement = 0
    final_metrics: dict = {}

    for epoch in range(MAX_EPOCHS):
        model.train()
        logits = model(dataset.X_train)
        loss = criterion(logits, dataset.y_train)

        if not torch.isfinite(loss):
            raise TrainingDivergedError(
                f"Non-finite loss at epoch {epoch} "
                f"(lr={network.learning_rate}, optimizer={network.optimizer_name})"
            )

        optimizer.zero_grad()
        loss.backward()

        if masks is not None:
            for name, p in model.named_parameters():
                if name in masks and p.grad is not None:
                    p.grad.mul_(masks[name])

        optimizer.step()

        if masks is not None:
            apply_masks(model, masks)
            if network.prune_interval_epochs and (epoch + 1) % network.prune_interval_epochs == 0:
                _, active_count, _ = count_active_parameters(masks)
                pruned_so_far = total_params - active_count
                max_allowed_pruned = (
                    total_params * (network.max_prune_pct / 100.0)
                    if network.max_prune_pct is not None
                    else total_params
                )
                if pruned_so_far < max_allowed_pruned:
                    prune_one_parameter(model, masks)
                    apply_masks(model, masks)

        if (epoch + 1) % EVAL_EVERY == 0 or epoch == MAX_EPOCHS - 1:
            model.eval()
            with torch.no_grad():
                type_logits = model(dataset.X_types)
                type_preds = type_logits.argmax(dim=-1)
                acc_types = (type_preds == dataset.y_types).float().mean().item()
                regular_acc = (
                    type_preds[: dataset.n_regular] == dataset.y_types[: dataset.n_regular]
                ).float().mean().item()
                irregular_acc = (
                    type_preds[dataset.n_regular :] == dataset.y_types[dataset.n_regular :]
                ).float().mean().item()

                wug_logits = model(dataset.X_wug)
                wm = wug_metrics(wug_logits, REGULAR_CLASS)

                norm = parameter_norm(model)
                active_pct = 100.0
                if masks is not None:
                    _, _, active_pct = count_active_parameters(masks)

            step_metrics = {
                "epoch": epoch + 1,
                "loss": loss.item(),
                "acc_types": acc_types,
                "regular_acc": regular_acc,
                "irregular_acc": irregular_acc,
                "param_norm": norm,
                "active_pct": active_pct,
                **wm,
            }
            history.append(step_metrics)
            final_metrics = step_metrics

            if report_intermediate is not None:
                report_intermediate(step_metrics)

            if best_loss - loss.item() > EARLY_STOP_MIN_DELTA:
                best_loss = loss.item()
                checkpoints_without_improvement = 0
            else:
                checkpoints_without_improvement += 1
                if checkpoints_without_improvement >= EARLY_STOP_PATIENCE_CHECKPOINTS:
                    logger.info(
                        "Early stop at epoch %d (no loss improvement for %d checkpoints)",
                        epoch + 1,
                        EARLY_STOP_PATIENCE_CHECKPOINTS,
                    )
                    break

    wall_clock = time.monotonic() - start
    return TrainResult(
        history=history,
        final=final_metrics,
        baseline=baseline,
        wall_clock_seconds=wall_clock,
        model=model if return_model else None,
        dataset=dataset if return_model else None,
    )
