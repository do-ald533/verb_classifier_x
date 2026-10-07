from __future__ import annotations

import math
import time
from collections.abc import Callable
from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.optim as optim

from src.config import REGULAR_CLASS, LinguisticConfig, NetworkConfig
from src.data import Dataset, TokenSource, build_dataset
from src.evaluation import compute_metrics
from src.logging_config import get_logger
from src.metrics import wug_metrics
from src.model import build_model, count_parameters
from src.pruning import (
    apply_masks,
    count_active_parameters,
    create_masks,
    prune_one_parameter,
)
from src.specs import NetworkSpec

logger = get_logger(__name__)


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
    stop_reason: str = "max_tokens"  # or "early_stopped"
    tokens_seen: int = 0
    n_params: int = 0
    extra: dict = field(default_factory=dict)


def _build_optimizer(model: nn.Module, net: NetworkSpec) -> optim.Optimizer:
    if net.optimizer == "sgd":
        return optim.SGD(
            model.parameters(), lr=net.lr, momentum=net.momentum, weight_decay=net.weight_decay
        )
    if net.optimizer == "adamw":
        return optim.AdamW(model.parameters(), lr=net.lr, weight_decay=net.weight_decay)
    return optim.Adam(model.parameters(), lr=net.lr, weight_decay=net.weight_decay)


def _lr_factor(net: NetworkSpec, max_steps: int) -> Callable[[int], float] | None:
    if net.lr_schedule == "cosine":
        return lambda step: 0.5 * (1.0 + math.cos(math.pi * min(1.0, step / max_steps)))
    if net.lr_schedule == "linear_decay":
        return lambda step: max(0.0, 1.0 - step / max_steps)
    return None


def train_model(
    dataset: Dataset,
    net: NetworkSpec,
    report_intermediate: Callable[[dict], None] | None = None,
) -> TrainResult:
    """Trains one (dataset, network) pair, sampling tokens on the fly (or
    cycling a stored finite training set) and reporting against cumulative
    tokens seen. `net` must have its seeds resolved (NetworkSpec.with_seed):
    init and data order use separate explicit generators, so runs are
    reproducible, parallel-safe, and each variance source can be isolated."""
    if net.init_seed is None or net.data_order_seed is None:
        raise ValueError("NetworkSpec seeds must be resolved; call spec.with_seed(seed)")
    start = time.monotonic()

    init_gen = torch.Generator().manual_seed(net.init_seed)
    order_gen = torch.Generator().manual_seed(net.data_order_seed)
    if net.dropout > 0:
        # dropout masks draw from the global RNG; pin it for reproducibility
        torch.manual_seed(net.data_order_seed)

    model = build_model(
        net,
        input_dim=dataset.input_dim,
        n_positions=dataset.spec.D,
        n_classes=dataset.n_classes,
        generator=init_gen,
    )
    n_params = count_parameters(model)
    masks = create_masks(model) if net.pruning_enabled else None
    total_params = sum(m.numel() for m in masks.values()) if masks is not None else None

    model.eval()
    with torch.no_grad():
        baseline = {
            f"baseline_{k}": v for k, v in wug_metrics(model(dataset.X_wug), REGULAR_CLASS).items()
        }

    criterion = nn.CrossEntropyLoss()
    optimizer = _build_optimizer(model, net)
    batch = net.batch_size
    eval_every_steps = max(1, net.eval_every_tokens // batch)
    max_steps = max(1, net.max_tokens // batch)
    factor = _lr_factor(net, max_steps)
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, factor) if factor else None

    source = TokenSource(dataset, order_gen)
    history: list[dict] = []
    best_loss = float("inf")
    stale_checkpoints = 0
    final_metrics: dict = {}
    stop_reason = "max_tokens"
    loss_sum = torch.zeros(())
    chunk = torch.empty(0, dtype=torch.long)

    model.train()
    for step in range(max_steps):
        if step % eval_every_steps == 0:
            chunk = source.take(eval_every_steps * batch)
            loss_sum = torch.zeros(())
        offset = (step % eval_every_steps) * batch
        idx = chunk[offset : offset + batch]

        loss = criterion(model(dataset.X_types[idx]), dataset.y_types[idx])
        if not torch.isfinite(loss):
            raise TrainingDivergedError(
                f"Non-finite loss at step {step} (lr={net.lr}, optimizer={net.optimizer})"
            )
        loss_sum += loss.detach()

        optimizer.zero_grad()
        loss.backward()
        if masks is not None:
            for name, p in model.named_parameters():
                if name in masks and p.grad is not None:
                    p.grad.mul_(masks[name])
        if net.grad_clip is not None:
            nn.utils.clip_grad_norm_(model.parameters(), net.grad_clip)
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        if masks is not None:
            apply_masks(model, masks)
            if net.prune_interval_steps and (step + 1) % net.prune_interval_steps == 0:
                _, active_count, _ = count_active_parameters(masks)
                pruned_so_far = total_params - active_count
                max_allowed = total_params * (net.max_prune_pct / 100.0)
                if pruned_so_far < max_allowed:
                    prune_one_parameter(model, masks)
                    apply_masks(model, masks)

        if (step + 1) % eval_every_steps == 0 or step == max_steps - 1:
            active_pct = count_active_parameters(masks)[2] if masks is not None else 100.0
            n_in_chunk = (step % eval_every_steps) + 1
            step_metrics = {
                "tokens_seen": (step + 1) * batch,
                "train_loss": float(loss_sum) / n_in_chunk,
                "lr": optimizer.param_groups[0]["lr"],
                **compute_metrics(model, dataset),
                "active_pct": active_pct,
            }
            history.append(step_metrics)
            final_metrics = step_metrics

            if report_intermediate is not None:
                report_intermediate(step_metrics)

            if net.early_stop:
                if best_loss - step_metrics["loss"] > net.early_stop_min_delta:
                    best_loss = step_metrics["loss"]
                    stale_checkpoints = 0
                else:
                    stale_checkpoints += 1
                    if stale_checkpoints >= net.early_stop_patience:
                        logger.info(
                            "Early stop at %d tokens (no loss improvement for %d checkpoints)",
                            step_metrics["tokens_seen"],
                            net.early_stop_patience,
                        )
                        stop_reason = "early_stopped"
                        break

    return TrainResult(
        history=history,
        final=final_metrics,
        baseline=baseline,
        wall_clock_seconds=time.monotonic() - start,
        model=model,
        dataset=dataset,
        stop_reason=stop_reason,
        tokens_seen=final_metrics.get("tokens_seen", 0),
        n_params=n_params,
    )


def train_one_run(
    linguistic: LinguisticConfig,
    network: NetworkConfig,
    seed: int,
    report_intermediate: Callable[[dict], None] | None = None,
    return_model: bool = False,
) -> TrainResult:
    """Legacy entry point of the Optuna pipeline (sweep/confirm/report):
    adapts (LinguisticConfig, NetworkConfig, seed) to the declarative
    specs. Data seed = `seed`, init seed = `seed + 1_000_000`."""
    dataset = build_dataset(linguistic.to_dataset_spec(seed))
    result = train_model(dataset, network.to_network_spec(seed), report_intermediate)
    if not return_model:
        result.model = None
        result.dataset = None
    return result
