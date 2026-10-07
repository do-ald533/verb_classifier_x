import pytest
import torch
from src.data import (
    TokenSource,
    build_dataset,
    dataset_stats,
    load_dataset,
    sample_token_indices,
    save_dataset,
)
from src.specs import DatasetSpec


def _spec(**overrides) -> DatasetSpec:
    base = dict(
        D=6, lexicon_size=200, p_irregular_types=0.1, p_token_irregular=0.7, n_wug=30, seed=0
    )
    return DatasetSpec(**{**base, **overrides})


def test_build_is_deterministic_given_spec():
    a, b = build_dataset(_spec()), build_dataset(_spec())
    assert torch.equal(a.X_types, b.X_types)
    assert torch.equal(a.y_types, b.y_types)
    assert torch.equal(a.X_wug, b.X_wug)


def test_different_seeds_give_different_lexicon():
    assert not torch.equal(
        build_dataset(_spec(seed=1)).X_types, build_dataset(_spec(seed=2)).X_types
    )


def test_wug_and_lexicon_disjoint_for_every_strategy():
    for strategy in ("random", "near_irregular", "far_from_lexicon"):
        ds = build_dataset(_spec(wug_strategy=strategy))
        wug = {tuple(r.tolist()) for r in ds.radicals_wug}
        lex = {tuple(r.tolist()) for r in ds.radicals_types}
        assert wug.isdisjoint(lex), strategy
        assert len(wug) == 30, strategy


def test_inputs_are_unique_so_labels_are_unambiguous():
    ds = build_dataset(_spec(D=8, lexicon_size=5000))
    assert torch.unique(ds.radicals_types, dim=0).shape[0] == 5000


def test_group_sizes_and_ordering():
    ds = build_dataset(_spec())
    assert ds.n_irregular == 20 and ds.n_regular == 180
    assert (ds.y_types[: ds.n_regular] == 0).all()
    assert (ds.y_types[ds.n_regular :] != 0).all()


def test_unique_class_per_irregular_verb():
    ds = build_dataset(_spec())
    assert ds.y_types[ds.n_regular :].unique().numel() == ds.n_irregular
    assert ds.n_classes == ds.n_irregular + 1
    assert int(ds.y_types.max()) == ds.n_classes - 1


def test_shared_k_classes():
    ds = build_dataset(_spec(class_scheme="shared_k", k_classes=4))
    assert ds.n_classes == 5
    assert set(ds.y_types[ds.n_regular :].tolist()) == {1, 2, 3, 4}
    counts = torch.bincount(ds.y_types[ds.n_regular :])[1:]
    assert counts.max() - counts.min() <= 1  # balanced


def test_clustered_irregulars_are_closer_than_random():
    kwargs = dict(D=8, lexicon_size=300, p_irregular_types=0.1)
    random = dataset_stats(build_dataset(_spec(**kwargs)))
    clustered = dataset_stats(build_dataset(_spec(irregular_placement="clustered", **kwargs)))
    assert clustered["irregular_mean_pairwise_hamming"] < random["irregular_mean_pairwise_hamming"]


def test_by_cluster_classes():
    ds = build_dataset(
        _spec(irregular_placement="clustered", n_clusters=3, class_scheme="by_cluster")
    )
    assert ds.n_classes == 4
    assert set(ds.y_types[ds.n_regular :].tolist()) == {1, 2, 3}


def test_wug_strategies_move_wugs_where_promised():
    near = dataset_stats(build_dataset(_spec(D=8, lexicon_size=300, wug_strategy="near_irregular")))
    rand = dataset_stats(build_dataset(_spec(D=8, lexicon_size=300)))
    far = dataset_stats(
        build_dataset(_spec(D=8, lexicon_size=300, wug_strategy="far_from_lexicon"))
    )
    assert near["wug_mean_min_hamming_to_irregular"] < rand["wug_mean_min_hamming_to_irregular"]
    assert far["wug_mean_min_hamming_to_lexicon"] > rand["wug_mean_min_hamming_to_lexicon"]


def test_one_hot_encoding_shape_and_content():
    ds = build_dataset(_spec(encoding="one_hot"))
    assert ds.X_types.shape == (200, 6 * 3)
    assert (ds.X_types.reshape(200, 6, 3).sum(-1) == 1).all()
    assert ds.input_dim == 18


def test_sampler_matches_token_probabilities():
    ds = build_dataset(_spec())
    idx = sample_token_indices(ds, 100_000, torch.Generator().manual_seed(1))
    assert abs((idx >= ds.n_regular).float().mean().item() - 0.7) < 0.01
    assert abs(ds.p_types.sum().item() - 1.0) < 1e-5


def test_sampler_is_uniform_within_groups():
    ds = build_dataset(_spec())
    idx = sample_token_indices(ds, 200_000, torch.Generator().manual_seed(2))
    counts = torch.bincount(idx, minlength=200).float()
    for group in (counts[: ds.n_regular], counts[ds.n_regular :]):
        assert group.std() / group.mean() < 0.2


def test_zipf_sampler_follows_declared_probabilities():
    ds = build_dataset(_spec(freq_distribution="zipf", zipf_exponent=1.2))
    n = 400_000
    idx = sample_token_indices(ds, n, torch.Generator().manual_seed(3))
    empirical = torch.bincount(idx, minlength=200).float() / n
    assert (empirical - ds.p_types).abs().max() < 0.005
    assert ds.p_types.max() > 5 * ds.p_types.min()  # genuinely skewed


def test_sampling_is_deterministic_given_generator_seed():
    ds = build_dataset(_spec())
    a = sample_token_indices(ds, 500, torch.Generator().manual_seed(5))
    b = sample_token_indices(ds, 500, torch.Generator().manual_seed(5))
    assert torch.equal(a, b)


def test_finite_sampling_cycles_through_stored_tokens():
    ds = build_dataset(_spec(sampling="finite", train_size=50))
    assert ds.train_indices is not None and ds.train_indices.numel() == 50
    source = TokenSource(ds, torch.Generator().manual_seed(0))
    seen = source.take(120)  # > 2 passes
    assert seen.numel() == 120
    first_pass = torch.sort(seen[:50]).values
    assert torch.equal(first_pass, torch.sort(ds.train_indices).values)


def test_save_load_roundtrip_is_exact(tmp_path):
    ds = build_dataset(_spec(freq_distribution="zipf", encoding="one_hot"))
    directory = save_dataset(ds, tmp_path)
    assert directory.name == ds.spec.hash
    loaded = load_dataset(directory)
    for name in ("X_types", "X_wug", "y_types", "p_types", "w_regular", "w_irregular"):
        assert torch.equal(getattr(ds, name), getattr(loaded, name)), name
    assert save_dataset(ds, tmp_path) == directory  # idempotent / cached


@pytest.mark.parametrize(
    "overrides",
    [
        dict(D=4, lexicon_size=200),  # 81 radicals cannot hold 230 items
        dict(class_scheme="shared_k"),  # needs k_classes
        dict(class_scheme="by_cluster"),  # needs clustered placement
        dict(sampling="finite"),  # needs train_size
        dict(p_irregular_types=0.001),  # empty irregular group
    ],
)
def test_invalid_specs_rejected(overrides):
    with pytest.raises(ValueError):
        _spec(**overrides)
