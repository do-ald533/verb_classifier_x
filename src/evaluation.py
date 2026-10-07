from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.config import REGULAR_CLASS
from src.data import Dataset
from src.metrics import wug_metrics
from src.pruning import parameter_norm

ACCURACY_THRESHOLDS = (0.5, 0.9, 0.99)


@torch.inference_mode()
def compute_metrics(model: nn.Module, dataset: Dataset) -> dict[str, float]:
    """All point-in-time metrics for a model on a dataset. Used for the
    training checkpoints and for the final (re-runnable) evaluation node, so
    both report exactly the same quantities."""
    was_training = model.training
    model.eval()

    type_logits = model(dataset.X_types)
    preds = type_logits.argmax(dim=-1)
    n_reg = dataset.n_regular
    correct = preds == dataset.y_types

    # expected per-token loss under the sampling distribution: a stable
    # convergence signal (single-token losses are noisy)
    per_type_loss = F.cross_entropy(type_logits, dataset.y_types, reduction="none")
    expected_loss = float((per_type_loss * dataset.p_types).sum())

    metrics = {
        "loss": expected_loss,
        "acc_types": float(correct.float().mean()),
        "regular_acc": float(correct[:n_reg].float().mean()),
        "irregular_acc": float(correct[n_reg:].float().mean()),
        "overregularization_rate": float((preds[n_reg:] == REGULAR_CLASS).float().mean()),
        "param_norm": parameter_norm(model),
        **wug_metrics(model(dataset.X_wug), REGULAR_CLASS),
    }
    model.train(was_training)
    return metrics


@torch.inference_mode()
def predictions(model: nn.Module, dataset: Dataset) -> dict[str, np.ndarray]:
    model.eval()
    wug_prob = F.softmax(model(dataset.X_wug), dim=-1)
    return {
        "type_pred": model(dataset.X_types).argmax(dim=-1).numpy(),
        "type_true": dataset.y_types.numpy(),
        "wug_pred": wug_prob.argmax(dim=-1).numpy(),
        "wug_prob": wug_prob.numpy(),
    }


def tokens_to_accuracy(history: list[dict], thresholds=ACCURACY_THRESHOLDS) -> dict[str, float]:
    """First checkpoint (in tokens seen) where type accuracy reached each
    threshold — the learning-speed measure. NaN if never reached."""
    out: dict[str, float] = {}
    for thr in thresholds:
        reached = [h["tokens_seen"] for h in history if h["acc_types"] >= thr]
        out[f"tokens_to_acc_{round(thr * 100)}"] = float(reached[0]) if reached else math.nan
    return out
