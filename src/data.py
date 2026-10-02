from __future__ import annotations

import itertools
from dataclasses import dataclass

import torch

from src.config import LinguisticConfig, N_WUG, REGULAR_CLASS, VALUES


@dataclass
class Dataset:
    """The lexicon (types), not a training set: training tokens are sampled
    on the fly with `sample_token_indices`."""

    X_wug: torch.Tensor
    X_types: torch.Tensor
    y_types: torch.Tensor
    p_types: torch.Tensor  # token probability of each type under the sampler
    n_regular: int
    n_irregular: int
    n_classes: int


def generate_all_radicals(D: int, generator: torch.Generator) -> torch.Tensor:
    """All 3**D ternary radicals, shuffled with the given (parallel-safe,
    explicit) generator — never the global RNG."""
    all_radicals = torch.tensor(list(itertools.product(VALUES, repeat=D)), dtype=torch.float32)
    perm = torch.randperm(len(all_radicals), generator=generator)
    return all_radicals[perm]


def build_dataset(condition: LinguisticConfig, generator: torch.Generator) -> Dataset:
    n_regular = condition.n_regular
    n_irregular = condition.N_IRREGULAR

    all_radicals = generate_all_radicals(condition.D, generator)

    X_wug = all_radicals[:N_WUG]
    X_lexicon = all_radicals[N_WUG : N_WUG + n_regular + n_irregular]

    # Radicals are unique, so no two verbs share an input with different labels.
    X_types = X_lexicon
    y_regular = torch.full((n_regular,), REGULAR_CLASS, dtype=torch.long)
    # Each irregular verb gets its own class: 1..n_irregular.
    y_irregular = torch.arange(1, n_irregular + 1, dtype=torch.long)
    y_types = torch.cat([y_regular, y_irregular], dim=0)

    p_types = torch.cat(
        [
            torch.full((n_regular,), condition.p_token_regular / n_regular),
            torch.full((n_irregular,), condition.P_TOKEN_IRREGULAR / n_irregular),
        ]
    )

    return Dataset(
        X_wug=X_wug,
        X_types=X_types,
        y_types=y_types,
        p_types=p_types,
        n_regular=n_regular,
        n_irregular=n_irregular,
        n_classes=condition.n_classes,
    )


def sample_token_indices(
    dataset: Dataset, p_token_irregular: float, n: int, generator: torch.Generator
) -> torch.Tensor:
    """Indices into dataset.X_types for n i.i.d. tokens: first draw whether
    each token is irregular (prob. p_token_irregular), then a verb uniformly
    within the chosen group."""
    is_irregular = torch.rand(n, generator=generator) < p_token_irregular
    regular_idx = torch.randint(0, dataset.n_regular, (n,), generator=generator)
    irregular_idx = dataset.n_regular + torch.randint(
        0, dataset.n_irregular, (n,), generator=generator
    )
    return torch.where(is_irregular, irregular_idx, regular_idx)
