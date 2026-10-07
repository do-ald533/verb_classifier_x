from __future__ import annotations

import math
from collections.abc import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.specs import NetworkSpec

ACTIVATIONS: dict[str, Callable[[], nn.Module]] = {
    "relu": nn.ReLU,
    "tanh": nn.Tanh,
    "gelu": nn.GELU,
    "sigmoid": nn.Sigmoid,
    "leaky_relu": nn.LeakyReLU,
    "silu": nn.SiLU,
}

_FUNCTIONAL_ACTIVATIONS: dict[str, str | Callable[[torch.Tensor], torch.Tensor]] = {
    "relu": "relu",
    "gelu": "gelu",
    "tanh": torch.tanh,
    "sigmoid": torch.sigmoid,
    "leaky_relu": F.leaky_relu,
    "silu": F.silu,
}


class VerbTransformer(nn.Module):
    """Each of the D radical positions is one token (value features ->
    d_model, plus a learned position embedding); a pre-norm transformer
    encoder, mean pooling over positions, linear classification head."""

    def __init__(self, net: NetworkSpec, n_positions: int, n_classes: int, features: int):
        super().__init__()
        self.n_positions = n_positions
        self.features = features
        self.embed = nn.Linear(features, net.d_model, bias=net.bias)
        self.pos = nn.Parameter(torch.zeros(n_positions, net.d_model))
        layer = nn.TransformerEncoderLayer(
            d_model=net.d_model,
            nhead=net.n_heads,
            dim_feedforward=net.ff_dim,
            dropout=net.dropout,
            activation=_FUNCTIONAL_ACTIVATIONS[net.activation],
            batch_first=True,
            norm_first=True,
            bias=net.bias,
        )
        self.encoder = nn.TransformerEncoder(layer, net.n_layers, enable_nested_tensor=False)
        self.final_norm: nn.Module = (
            nn.LayerNorm(net.d_model) if net.norm == "layernorm" else nn.Identity()
        )
        self.head = nn.Linear(net.d_model, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        tokens = x.reshape(x.shape[0], self.n_positions, self.features)
        h = self.encoder(self.embed(tokens) + self.pos)
        return self.head(self.final_norm(h.mean(dim=1)))


def _init_weight(weight: torch.Tensor, scheme: str, generator: torch.Generator) -> None:
    if scheme == "kaiming":  # nn.Linear's own default
        nn.init.kaiming_uniform_(weight, a=math.sqrt(5), generator=generator)
    elif scheme == "xavier":
        nn.init.xavier_uniform_(weight, generator=generator)
    else:
        fan_in = nn.init._calculate_fan_in_and_fan_out(weight)[0]
        nn.init.normal_(weight, 0.0, 1.0 / math.sqrt(max(fan_in, 1)), generator=generator)


def init_parameters(model: nn.Module, scheme: str, generator: torch.Generator) -> None:
    """Re-initializes every parameter from an explicit generator instead of
    the global RNG, so model init is decoupled from data generation and safe
    to run in parallel processes."""
    for module in model.modules():
        if isinstance(module, nn.Linear):
            _init_weight(module.weight, scheme, generator)
            if module.bias is not None:
                if scheme == "kaiming":
                    bound = 1 / math.sqrt(module.in_features) if module.in_features > 0 else 0
                    nn.init.uniform_(module.bias, -bound, bound, generator=generator)
                else:
                    nn.init.zeros_(module.bias)
        elif isinstance(module, nn.MultiheadAttention):
            _init_weight(module.in_proj_weight, scheme, generator)
            if module.in_proj_bias is not None:
                nn.init.zeros_(module.in_proj_bias)
        elif isinstance(module, VerbTransformer):
            nn.init.normal_(module.pos, 0.0, 0.02, generator=generator)


def build_model(
    net: NetworkSpec,
    *,
    input_dim: int,
    n_positions: int,
    n_classes: int,
    generator: torch.Generator,
) -> nn.Module:
    if input_dim % n_positions != 0:
        raise ValueError("input_dim must be a multiple of n_positions")

    model: nn.Module
    if net.arch == "mlp":
        layers: list[nn.Module] = []
        in_dim = input_dim
        for width in net.hidden:
            layers.append(nn.Linear(in_dim, width, bias=net.bias))
            if net.norm == "layernorm":
                layers.append(nn.LayerNorm(width))
            elif net.norm == "batchnorm":
                layers.append(nn.BatchNorm1d(width))
            layers.append(ACTIVATIONS[net.activation]())
            if net.dropout > 0:
                layers.append(nn.Dropout(net.dropout))
            in_dim = width
        layers.append(nn.Linear(in_dim, n_classes))
        model = nn.Sequential(*layers)
    else:
        model = VerbTransformer(net, n_positions, n_classes, input_dim // n_positions)

    init_parameters(model, net.init, generator)
    return model


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())
