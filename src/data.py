from __future__ import annotations

import itertools
from dataclasses import dataclass

import torch

from src.config import LinguisticConfig, N_CLASSES, N_TRAIN_TOKENS, N_WUG, REGULAR_CLASS, VALUES


@dataclass
class Dataset:
    X_wug: torch.Tensor
    X_types: torch.Tensor
    y_types: torch.Tensor
    X_train: torch.Tensor
    y_train: torch.Tensor
    n_regular: int
    n_irregular: int


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

    X_regular = X_lexicon[:n_regular]
    X_irregular = X_lexicon[n_regular:]

    y_regular = torch.full((n_regular,), REGULAR_CLASS, dtype=torch.long)
    y_irregular = torch.randint(
        1, N_CLASSES, size=(n_irregular,), generator=generator, dtype=torch.long
    )

    X_types = torch.cat([X_regular, X_irregular], dim=0)
    y_types = torch.cat([y_regular, y_irregular], dim=0)

    regular_probs = torch.full((n_regular,), condition.p_token_regular / n_regular)
    irregular_probs = torch.full((n_irregular,), condition.P_TOKEN_IRREGULAR / n_irregular)
    sampling_probs = torch.cat([regular_probs, irregular_probs])
    sampling_probs = sampling_probs / sampling_probs.sum()

    train_indices = torch.multinomial(
        sampling_probs, num_samples=N_TRAIN_TOKENS, replacement=True, generator=generator
    )
    X_train = X_types[train_indices]
    y_train = y_types[train_indices]

    return Dataset(
        X_wug=X_wug,
        X_types=X_types,
        y_types=y_types,
        X_train=X_train,
        y_train=y_train,
        n_regular=n_regular,
        n_irregular=n_irregular,
    )
