from __future__ import annotations

import itertools
from typing import Literal

from pydantic import BaseModel, Field, model_validator

# --------------------------------------------------------------------------
# Fixed constants.
#
# Training is a stream of sampled tokens, not an epoch over a fixed dataset
# (see src/data.py::sample_token_indices): each step draws regular/irregular
# with probability P_TOKEN_IRREGULAR, then a verb uniformly inside that
# group. Learning is measured against the cumulative number of tokens seen.
#
# The lexicon size is a condition factor (LEXICON_SIZE_VALUES). Every
# radical must be unique, otherwise two verbs could share an input vector
# but have different output classes, so a condition is only feasible when
# 3**D >= LEXICON_SIZE + N_WUG (enforced in LinguisticConfig and respected
# by generate_condition_grid, which skips infeasible (D, size) pairs).
# --------------------------------------------------------------------------
VALUES = (-1.0, 0.0, 1.0)
REGULAR_CLASS = 0
N_WUG = 30
BATCH_SIZE = 1  # tokens per optimizer step; 1 = pure token-by-token sampling
MAX_TOKENS = 5_000_000
EVAL_EVERY_TOKENS = 10_000

SEARCH_SEEDS = list(range(5))
CONFIRMATION_SEEDS = list(range(1000, 1020))

TRIAL_BUDGET_SINGLE_OBJECTIVE = 35
TRIAL_BUDGET_MULTI_OBJECTIVE = 60

D_VALUES = (6, 8, 10)
LEXICON_SIZE_VALUES = (100, 500, 1000, 5000, 10000)
P_IRREGULAR_TYPES_VALUES = (0.02, 0.05, 0.10)  # fraction of lexicon types that are irregular
P_TOKEN_IRREGULAR_VALUES = (0.5, 0.7, 0.9)
PRUNING_VALUES = (False, True)


class LinguisticConfig(BaseModel):
    D: int
    LEXICON_SIZE: int
    P_IRREGULAR_TYPES: float
    P_TOKEN_IRREGULAR: float
    pruning_enabled: bool

    @model_validator(mode="after")
    def _check_feasible(self) -> "LinguisticConfig":
        available = 3**self.D
        needed = self.LEXICON_SIZE + N_WUG
        if needed > available:
            raise ValueError(
                f"D={self.D} gives only {available} unique radicals, but "
                f"LEXICON_SIZE+N_WUG={needed} are needed; duplicate inputs "
                f"could carry different output classes. Increase D or "
                f"reduce LEXICON_SIZE."
            )
        if not (0.0 < self.P_IRREGULAR_TYPES < 1.0):
            raise ValueError("P_IRREGULAR_TYPES must be in (0, 1)")
        if self.N_IRREGULAR < 1 or self.n_regular < 1:
            raise ValueError(
                f"LEXICON_SIZE={self.LEXICON_SIZE} with P_IRREGULAR_TYPES="
                f"{self.P_IRREGULAR_TYPES} leaves an empty regular/irregular group"
            )
        if not (0.0 <= self.P_TOKEN_IRREGULAR <= 1.0):
            raise ValueError("P_TOKEN_IRREGULAR must be in [0, 1]")
        return self

    @property
    def N_IRREGULAR(self) -> int:
        return round(self.LEXICON_SIZE * self.P_IRREGULAR_TYPES)

    @property
    def n_regular(self) -> int:
        return self.LEXICON_SIZE - self.N_IRREGULAR

    @property
    def n_classes(self) -> int:
        """One class per irregular verb, plus the shared regular class."""
        return self.N_IRREGULAR + 1

    @property
    def p_token_regular(self) -> float:
        return 1.0 - self.P_TOKEN_IRREGULAR

    @property
    def lexicon_density(self) -> float:
        """Fraction of the 3**D radical space occupied by the lexicon —
        varies with D and LEXICON_SIZE, logged so the confound is visible
        in analysis instead of hidden."""
        return self.LEXICON_SIZE / (3**self.D)

    @property
    def id(self) -> str:
        prune = "on" if self.pruning_enabled else "off"
        freq = str(self.P_TOKEN_IRREGULAR).replace(".", "")
        irr = f"{round(self.P_IRREGULAR_TYPES * 100)}"
        return f"D{self.D}_lex{self.LEXICON_SIZE}_irr{irr}_freq{freq}_prune{prune}"

    @property
    def is_multi_objective(self) -> bool:
        return self.pruning_enabled

    @property
    def trial_budget(self) -> int:
        return (
            TRIAL_BUDGET_MULTI_OBJECTIVE
            if self.is_multi_objective
            else TRIAL_BUDGET_SINGLE_OBJECTIVE
        )

    def as_params(self) -> dict:
        return {
            "D": self.D,
            "LEXICON_SIZE": self.LEXICON_SIZE,
            "P_IRREGULAR_TYPES": self.P_IRREGULAR_TYPES,
            "N_IRREGULAR": self.N_IRREGULAR,
            "N_REGULAR": self.n_regular,
            "N_CLASSES": self.n_classes,
            "P_TOKEN_IRREGULAR": self.P_TOKEN_IRREGULAR,
            "pruning_enabled": self.pruning_enabled,
            "lexicon_density": self.lexicon_density,
        }


class NetworkConfig(BaseModel):
    hidden_dim: int = Field(ge=1)
    num_hidden_layers: int = Field(ge=1, le=3)
    optimizer_name: Literal["SGD", "Adam"]
    learning_rate: float = Field(gt=0)
    momentum: float = Field(ge=0.0, le=1.0, default=0.0)
    prune_interval_steps: int | None = None
    max_prune_pct: float | None = None

    def as_params(self) -> dict:
        return self.model_dump()


def generate_condition_grid() -> list[LinguisticConfig]:
    """Every feasible combination of D x LEXICON_SIZE x P_IRREGULAR_TYPES x
    P_TOKEN_IRREGULAR x pruning. (D, LEXICON_SIZE) pairs with
    3**D < LEXICON_SIZE + N_WUG are skipped."""
    grid = []
    for D, size, p_types, p_tokens, pruning in itertools.product(
        D_VALUES,
        LEXICON_SIZE_VALUES,
        P_IRREGULAR_TYPES_VALUES,
        P_TOKEN_IRREGULAR_VALUES,
        PRUNING_VALUES,
    ):
        if size + N_WUG > 3**D:
            continue
        grid.append(
            LinguisticConfig(
                D=D,
                LEXICON_SIZE=size,
                P_IRREGULAR_TYPES=p_types,
                P_TOKEN_IRREGULAR=p_tokens,
                pruning_enabled=pruning,
            )
        )
    return grid
