import pytest
import torch
from src.model import build_model, count_parameters
from src.specs import NetworkSpec


def _build(net: NetworkSpec, input_dim=6, n_positions=6, n_classes=20, seed=0):
    return build_model(
        net,
        input_dim=input_dim,
        n_positions=n_positions,
        n_classes=n_classes,
        generator=torch.Generator().manual_seed(seed),
    )


def test_output_shape():
    out = _build(NetworkSpec(hidden=(8, 8)))(torch.zeros(5, 6))
    assert out.shape == (5, 20)


def test_deterministic_init_given_generator():
    a, b = _build(NetworkSpec(), seed=42), _build(NetworkSpec(), seed=42)
    for p_a, p_b in zip(a.parameters(), b.parameters(), strict=True):
        assert torch.equal(p_a, p_b)


def test_different_generator_gives_different_init():
    a, b = _build(NetworkSpec(), seed=1), _build(NetworkSpec(), seed=2)
    assert any(
        not torch.equal(p_a, p_b) for p_a, p_b in zip(a.parameters(), b.parameters(), strict=True)
    )


def test_hidden_layers_respected():
    model = _build(NetworkSpec(hidden=(8, 8, 8)))
    assert len([m for m in model if isinstance(m, torch.nn.Linear)]) == 3 + 1


@pytest.mark.parametrize("activation", ["relu", "tanh", "gelu", "sigmoid", "leaky_relu", "silu"])
@pytest.mark.parametrize("init", ["kaiming", "xavier", "normal"])
def test_mlp_activation_and_init_variants_run(activation, init):
    out = _build(NetworkSpec(activation=activation, init=init))(torch.randn(3, 6))
    assert torch.isfinite(out).all()


@pytest.mark.parametrize("norm", ["layernorm", "batchnorm"])
def test_mlp_norm_variants_run(norm):
    net = NetworkSpec(norm=norm, batch_size=4, dropout=0.1, bias=False)
    out = _build(net)(torch.randn(4, 6))
    assert out.shape == (4, 20)


def test_empty_hidden_is_a_linear_classifier():
    model = _build(NetworkSpec(hidden=()))
    assert count_parameters(model) == 6 * 20 + 20


@pytest.mark.parametrize("activation", ["relu", "gelu", "tanh", "silu"])
def test_transformer_shapes_for_raw_and_one_hot(activation):
    net = NetworkSpec(arch="transformer", d_model=16, n_heads=4, n_layers=2, activation=activation)
    raw = _build(net, input_dim=6, n_positions=6)(torch.randn(5, 6))
    one_hot = _build(net, input_dim=18, n_positions=6)(torch.randn(5, 18))
    assert raw.shape == one_hot.shape == (5, 20)


def test_transformer_init_is_deterministic():
    net = NetworkSpec(arch="transformer", d_model=16, n_heads=2)
    a, b = _build(net, seed=3), _build(net, seed=3)
    for p_a, p_b in zip(a.parameters(), b.parameters(), strict=True):
        assert torch.equal(p_a, p_b)


@pytest.mark.parametrize(
    "overrides",
    [
        dict(arch="transformer", d_model=10, n_heads=4),
        dict(norm="batchnorm", batch_size=1),
        dict(prune_interval_steps=10),
        dict(hidden=(0,)),
    ],
)
def test_invalid_network_specs_rejected(overrides):
    with pytest.raises(ValueError):
        NetworkSpec(**overrides)
