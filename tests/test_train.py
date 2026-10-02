import pytest

import src.train as train
from src.config import LinguisticConfig, NetworkConfig


@pytest.fixture(autouse=True)
def _short_runs(monkeypatch):
    monkeypatch.setattr(train, "EVAL_EVERY_STEPS", 200)
    monkeypatch.setattr(train, "MAX_STEPS", 600)


def _condition(pruning: bool = False) -> LinguisticConfig:
    return LinguisticConfig(
        D=6, LEXICON_SIZE=100, P_IRREGULAR_TYPES=0.1, P_TOKEN_IRREGULAR=0.7, pruning_enabled=pruning
    )


def _network(pruning: bool = False) -> NetworkConfig:
    return NetworkConfig(
        hidden_dim=8,
        num_hidden_layers=1,
        optimizer_name="Adam",
        learning_rate=0.01,
        prune_interval_steps=50 if pruning else None,
        max_prune_pct=20.0 if pruning else None,
    )


def test_history_is_in_tokens_seen_and_output_dim_matches_classes():
    result = train.train_one_run(_condition(), _network(), seed=0, return_model=True)
    assert [h["tokens_seen"] for h in result.history] == [200, 400, 600]
    assert result.model[-1].out_features == _condition().n_classes == 11


def test_training_is_deterministic_given_seed():
    a = train.train_one_run(_condition(), _network(), seed=3)
    b = train.train_one_run(_condition(), _network(), seed=3)
    assert a.final == b.final


def test_pruning_reduces_active_params():
    result = train.train_one_run(_condition(True), _network(True), seed=0)
    assert result.final["active_pct"] < 100.0
