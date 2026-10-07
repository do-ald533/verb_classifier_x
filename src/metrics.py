from __future__ import annotations

import torch
import torch.nn.functional as F


def wug_metrics(logits: torch.Tensor, regular_class: int = 0) -> dict[str, float]:
    """logits: (N_WUG, n_classes) raw model outputs on the held-out wug set.

    wug_regular_prob is the primary, continuous metric (mean softmax mass on
    the regular class) — replaces the discrete wug_regular_rate as the
    Optuna objective (plan §1.1 #3): 10-30 discrete wugs give a coarse,
    noisy signal, but the underlying probabilities are continuous and much
    less noisy to optimize against.
    """
    probs = F.softmax(logits, dim=-1)

    regular_prob = probs[:, regular_class].mean().item()

    preds = probs.argmax(dim=-1)
    regular_rate = (preds == regular_class).float().mean().item()

    safe_probs = probs.clamp_min(1e-12)
    entropy = (-(safe_probs * safe_probs.log()).sum(dim=-1)).mean().item()

    top2 = probs.topk(2, dim=-1).values
    top2_margin = (top2[:, 0] - top2[:, 1]).mean().item()

    return {
        "wug_regular_prob": regular_prob,
        "wug_regular_rate": regular_rate,
        "wug_entropy": entropy,
        "wug_top2_margin": top2_margin,
    }
