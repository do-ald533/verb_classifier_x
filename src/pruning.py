from __future__ import annotations

import torch
import torch.nn as nn


def create_masks(model: nn.Module) -> dict[str, torch.Tensor]:
    return {name: torch.ones_like(p) for name, p in model.named_parameters() if p.requires_grad}


def apply_masks(model: nn.Module, masks: dict[str, torch.Tensor]) -> None:
    with torch.no_grad():
        for name, p in model.named_parameters():
            if name in masks:
                p.mul_(masks[name])


def prune_one_parameter(
    model: nn.Module, masks: dict[str, torch.Tensor]
) -> tuple[str | None, float]:
    """Zeroes the smallest-magnitude still-active parameter, permanently."""
    min_value = float("inf")
    min_name: str | None = None
    min_index = None

    for name, p in model.named_parameters():
        if name not in masks:
            continue

        active = masks[name].bool()
        if active.sum() == 0:
            continue

        values = p.detach().abs()
        values_inactive_as_inf = values.clone()
        values_inactive_as_inf[~active] = float("inf")

        current_min = values_inactive_as_inf.min()
        if current_min.item() < min_value:
            min_value = current_min.item()
            min_name = name
            min_index = values_inactive_as_inf.argmin()

    if min_name is not None:
        flat_mask = masks[min_name].view(-1)
        flat_mask[min_index] = 0.0

    return min_name, min_value


def parameter_norm(model: nn.Module) -> float:
    total = torch.zeros(())
    for p in model.parameters():
        total = total + p.detach().pow(2).sum()
    return total.sqrt().item()


def count_active_parameters(masks: dict[str, torch.Tensor]) -> tuple[int, int, float]:
    active = 0
    total = 0
    for mask in masks.values():
        active += int(mask.sum().item())
        total += mask.numel()
    pct = 100 * active / total
    return active, total, pct
