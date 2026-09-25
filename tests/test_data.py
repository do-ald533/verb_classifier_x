import torch

from src.config import LinguisticConfig, N_WUG
from src.data import build_dataset


def _condition() -> LinguisticConfig:
    return LinguisticConfig(D=6, N_IRREGULAR=36, P_TOKEN_IRREGULAR=0.7, pruning_enabled=False)


def test_determinism_given_seed():
    condition = _condition()
    gen_a = torch.Generator().manual_seed(123)
    gen_b = torch.Generator().manual_seed(123)
    a = build_dataset(condition, gen_a)
    b = build_dataset(condition, gen_b)
    assert torch.equal(a.X_train, b.X_train)
    assert torch.equal(a.y_train, b.y_train)


def test_different_seeds_give_different_data():
    condition = _condition()
    a = build_dataset(condition, torch.Generator().manual_seed(1))
    b = build_dataset(condition, torch.Generator().manual_seed(2))
    assert not torch.equal(a.X_train, b.X_train)


def test_wug_and_lexicon_disjoint():
    condition = _condition()
    dataset = build_dataset(condition, torch.Generator().manual_seed(0))
    wug_set = {tuple(row.tolist()) for row in dataset.X_wug}
    lexicon_set = {tuple(row.tolist()) for row in dataset.X_types}
    assert wug_set.isdisjoint(lexicon_set)
    assert dataset.X_wug.shape[0] == N_WUG


def test_class_balance():
    condition = _condition()
    dataset = build_dataset(condition, torch.Generator().manual_seed(0))
    assert (dataset.y_types[: dataset.n_regular] == 0).all()
    assert (dataset.y_types[dataset.n_regular :] != 0).all()
    assert dataset.n_irregular == 36


def test_train_tokens_shape():
    condition = _condition()
    dataset = build_dataset(condition, torch.Generator().manual_seed(0))
    assert dataset.X_train.shape[0] == 10_000
    assert dataset.y_train.shape[0] == 10_000
