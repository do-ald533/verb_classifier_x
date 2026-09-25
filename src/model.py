from __future__ import annotations

import math

import torch
import torch.nn as nn

from src.config import NetworkConfig


def _reinit_linear(layer: nn.Linear, generator: torch.Generator) -> None:
    """Replicates nn.Linear's own default init (kaiming_uniform_ on the
    weight, matching uniform bound on the bias) but driven by an explicit
    generator instead of the global RNG, so model init is decoupled from
    data generation and safe to run in parallel processes."""
    nn.init.kaiming_uniform_(layer.weight, a=math.sqrt(5), generator=generator)
    if layer.bias is not None:
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(layer.weight)
        bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
        nn.init.uniform_(layer.bias, -bound, bound, generator=generator)


def build_mlp(
    network_config: NetworkConfig, input_dim: int, n_classes: int, generator: torch.Generator
) -> nn.Module:
    layers: list[nn.Module] = []
    in_dim = input_dim
    for _ in range(network_config.num_hidden_layers):
        linear = nn.Linear(in_dim, network_config.hidden_dim)
        _reinit_linear(linear, generator)
        layers.append(linear)
        layers.append(nn.ReLU())
        in_dim = network_config.hidden_dim

    output = nn.Linear(in_dim, n_classes)
    _reinit_linear(output, generator)
    layers.append(output)

    return nn.Sequential(*layers)
