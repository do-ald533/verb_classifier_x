import torch
import torch.nn as nn

from src.pruning import apply_masks, count_active_parameters, create_masks, prune_one_parameter


def _tiny_model() -> nn.Module:
    torch.manual_seed(0)
    return nn.Sequential(nn.Linear(3, 4), nn.ReLU(), nn.Linear(4, 2))


def test_prune_zeroes_smallest_and_mask_monotonic():
    model = _tiny_model()
    masks = create_masks(model)
    active_before, total, _ = count_active_parameters(masks)

    prune_one_parameter(model, masks)
    apply_masks(model, masks)
    active_after, _, _ = count_active_parameters(masks)

    assert active_after == active_before - 1
    assert active_after < total


def test_pruned_params_stay_zero_after_grad_update():
    model = _tiny_model()
    masks = create_masks(model)
    prune_one_parameter(model, masks)
    apply_masks(model, masks)

    x = torch.randn(5, 3)
    loss = model(x).sum()
    loss.backward()
    for name, p in model.named_parameters():
        if p.grad is not None:
            p.grad.mul_(masks[name])
    with torch.no_grad():
        for p in model.parameters():
            if p.grad is not None:
                p -= 0.1 * p.grad
    apply_masks(model, masks)

    active, total, _ = count_active_parameters(masks)
    assert active == total - 1


def test_repeated_pruning_never_reactivates_a_parameter():
    model = _tiny_model()
    masks = create_masks(model)
    _, total, _ = count_active_parameters(masks)

    for i in range(5):
        prune_one_parameter(model, masks)
        apply_masks(model, masks)
        active, _, _ = count_active_parameters(masks)
        assert active == total - (i + 1)
