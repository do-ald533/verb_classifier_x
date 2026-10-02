import torch

from src.config import LinguisticConfig, N_WUG
from src.data import build_dataset, sample_token_indices


def _condition() -> LinguisticConfig:
    return LinguisticConfig(D=6, LEXICON_SIZE=699, P_IRREGULAR_TYPES=0.05, P_TOKEN_IRREGULAR=0.7, pruning_enabled=False)


def test_sampling_determinism_given_seed():
    condition = _condition()
    ds = build_dataset(condition, torch.Generator().manual_seed(123))
    a = sample_token_indices(ds, 0.7, 500, torch.Generator().manual_seed(5))
    b = sample_token_indices(ds, 0.7, 500, torch.Generator().manual_seed(5))
    assert torch.equal(a, b)


def test_different_seeds_give_different_lexicon():
    condition = _condition()
    a = build_dataset(condition, torch.Generator().manual_seed(1))
    b = build_dataset(condition, torch.Generator().manual_seed(2))
    assert not torch.equal(a.X_types, b.X_types)


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
    assert dataset.n_irregular == 35


def test_each_irregular_verb_has_unique_class():
    condition = _condition()
    dataset = build_dataset(condition, torch.Generator().manual_seed(0))
    irregular_classes = dataset.y_types[dataset.n_regular :]
    assert irregular_classes.unique().numel() == dataset.n_irregular
    assert dataset.n_classes == dataset.n_irregular + 1
    assert int(dataset.y_types.max()) == dataset.n_classes - 1


def test_inputs_unique_so_labels_unambiguous():
    condition = LinguisticConfig(
        D=8, LEXICON_SIZE=5000, P_IRREGULAR_TYPES=0.1, P_TOKEN_IRREGULAR=0.7, pruning_enabled=False
    )
    dataset = build_dataset(condition, torch.Generator().manual_seed(0))
    assert torch.unique(dataset.X_types, dim=0).shape[0] == 5000


def test_sampler_matches_token_probabilities():
    condition = _condition()
    dataset = build_dataset(condition, torch.Generator().manual_seed(0))
    idx = sample_token_indices(dataset, 0.7, 50_000, torch.Generator().manual_seed(1))
    irregular_rate = (idx >= dataset.n_regular).float().mean().item()
    assert abs(irregular_rate - 0.7) < 0.02
    assert abs(dataset.p_types.sum().item() - 1.0) < 1e-5
