import torch

from src.config import NetworkConfig

N_CLASSES = 20
from src.model import build_mlp


def _network_config() -> NetworkConfig:
    return NetworkConfig(hidden_dim=8, num_hidden_layers=2, optimizer_name="SGD", learning_rate=0.05)


def test_output_shape():
    model = build_mlp(_network_config(), input_dim=6, n_classes=N_CLASSES, generator=torch.Generator().manual_seed(0))
    out = model(torch.zeros(5, 6))
    assert out.shape == (5, N_CLASSES)


def test_deterministic_init_given_generator():
    config = _network_config()
    model_a = build_mlp(config, input_dim=6, n_classes=N_CLASSES, generator=torch.Generator().manual_seed(42))
    model_b = build_mlp(config, input_dim=6, n_classes=N_CLASSES, generator=torch.Generator().manual_seed(42))
    for p_a, p_b in zip(model_a.parameters(), model_b.parameters()):
        assert torch.equal(p_a, p_b)


def test_different_generator_gives_different_init():
    config = _network_config()
    model_a = build_mlp(config, input_dim=6, n_classes=N_CLASSES, generator=torch.Generator().manual_seed(1))
    model_b = build_mlp(config, input_dim=6, n_classes=N_CLASSES, generator=torch.Generator().manual_seed(2))
    params_a = list(model_a.parameters())
    params_b = list(model_b.parameters())
    assert not torch.equal(params_a[0], params_b[0])


def test_num_hidden_layers_respected():
    config = NetworkConfig(hidden_dim=8, num_hidden_layers=3, optimizer_name="Adam", learning_rate=0.01)
    model = build_mlp(config, input_dim=6, n_classes=N_CLASSES, generator=torch.Generator().manual_seed(0))
    linear_layers = [m for m in model if isinstance(m, torch.nn.Linear)]
    assert len(linear_layers) == 3 + 1  # hidden layers + output layer
