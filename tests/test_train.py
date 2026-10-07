import pytest
from src.config import LinguisticConfig, NetworkConfig
from src.data import build_dataset
from src.specs import DatasetSpec, NetworkSpec
from src.train import TrainingDivergedError, train_model, train_one_run

DATASET = DatasetSpec(D=6, lexicon_size=100, p_irregular_types=0.1, p_token_irregular=0.7, seed=0)


def _net(**overrides) -> NetworkSpec:
    base = dict(hidden=(8,), max_tokens=600, eval_every_tokens=200, early_stop=False, lr=0.01)
    return NetworkSpec(**{**base, **overrides}).with_seed(0)


def test_history_is_in_tokens_seen_and_output_dim_matches_classes():
    result = train_model(build_dataset(DATASET), _net())
    assert [h["tokens_seen"] for h in result.history] == [200, 400, 600]
    assert result.model[-1].out_features == DATASET.n_classes == 11


def test_batch_size_counts_tokens_not_steps():
    result = train_model(build_dataset(DATASET), _net(batch_size=10))
    assert [h["tokens_seen"] for h in result.history] == [200, 400, 600]


def test_training_is_deterministic_given_seeds():
    ds = build_dataset(DATASET)
    assert train_model(ds, _net()).final == train_model(ds, _net()).final


def test_init_and_data_order_seeds_are_independent():
    ds = build_dataset(DATASET)
    base = train_model(ds, _net()).final["loss"]
    other_init = train_model(ds, _net(init_seed=99)).final["loss"]
    other_order = train_model(ds, _net(data_order_seed=99)).final["loss"]
    assert len({base, other_init, other_order}) == 3


def test_requires_resolved_seeds():
    with pytest.raises(ValueError):
        train_model(build_dataset(DATASET), NetworkSpec())


def test_pruning_reduces_active_params():
    result = train_model(build_dataset(DATASET), _net(prune_interval_steps=20, max_prune_pct=20.0))
    assert result.final["active_pct"] < 100.0


@pytest.mark.parametrize(
    "overrides",
    [
        dict(optimizer="sgd", momentum=0.9, lr=0.05),
        dict(optimizer="adamw", weight_decay=0.01),
        dict(lr_schedule="cosine"),
        dict(lr_schedule="linear_decay"),
        dict(grad_clip=1.0),
        dict(dropout=0.2, hidden=(16, 16)),
        dict(arch="transformer", d_model=16, n_heads=2, lr=0.003),
        dict(norm="batchnorm", batch_size=8),
    ],
)
def test_option_variants_train_without_error(overrides):
    result = train_model(build_dataset(DATASET), _net(**overrides))
    assert result.history and result.final["loss"] == result.final["loss"]


def test_lr_schedule_changes_learning_rate():
    result = train_model(build_dataset(DATASET), _net(lr_schedule="linear_decay"))
    lrs = [h["lr"] for h in result.history]
    assert lrs[0] > lrs[-1]


def test_early_stop_triggers_on_plateau():
    result = train_model(
        build_dataset(DATASET),
        _net(max_tokens=200_000, early_stop=True, early_stop_patience=2, early_stop_min_delta=10.0),
    )
    assert result.stop_reason == "early_stopped"
    assert result.tokens_seen < 200_000


def test_finite_dataset_trains():
    ds = build_dataset(
        DatasetSpec(**{**DATASET.model_dump(), "sampling": "finite", "train_size": 300})
    )
    assert train_model(ds, _net()).history


def test_divergence_is_raised():
    with pytest.raises(TrainingDivergedError):
        train_model(build_dataset(DATASET), _net(optimizer="sgd", lr=1e12, max_tokens=5000))


def test_legacy_wrapper_still_works():
    linguistic = LinguisticConfig(
        D=6, LEXICON_SIZE=100, P_IRREGULAR_TYPES=0.1, P_TOKEN_IRREGULAR=0.7, pruning_enabled=False
    )
    network = NetworkConfig(
        hidden_dim=8, num_hidden_layers=1, optimizer_name="Adam", learning_rate=0.01
    )
    import src.config as config

    original = config.MAX_TOKENS
    try:
        config.MAX_TOKENS = 400
        result = train_one_run(linguistic, network, seed=0, return_model=True)
    finally:
        config.MAX_TOKENS = original
    assert result.model is not None and result.dataset is not None
    assert result.dataset.n_classes == linguistic.n_classes
    assert "wug_regular_prob" in result.final
