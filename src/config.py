from __future__ import annotations

import itertools
from typing import Literal

from pydantic import BaseModel, Field, model_validator

# --------------------------------------------------------------------------
# Fixed constants (v4 revision — see EXPERIMENT_PIPELINE_PLAN.md §1.1 / §4).
# N_VERBS_TOTAL + N_WUG = 699 + 30 = 729 = 3**6, so the D=6 condition still
# uses exactly its whole radical space, matching the source notebook's own
# design. D=8 (6561 radicals) leaves most of the space unused on purpose —
# see LinguisticConfig.lexicon_density, which is logged, not hidden.
# --------------------------------------------------------------------------
VALUES = (-1.0, 0.0, 1.0)
N_CLASSES = 20
REGULAR_CLASS = 0
N_VERBS_TOTAL = 699
N_WUG = 30
N_TRAIN_TOKENS = 10_000

SEARCH_SEEDS = list(range(5))
CONFIRMATION_SEEDS = list(range(1000, 1020))

TRIAL_BUDGET_SINGLE_OBJECTIVE = 35
TRIAL_BUDGET_MULTI_OBJECTIVE = 60

D_VALUES = (6, 8)
N_IRREGULAR_VALUES = (12, 36, 72)
P_TOKEN_IRREGULAR_VALUES = (0.5, 0.7, 0.9)
PRUNING_VALUES = (False, True)


class LinguisticConfig(BaseModel):
    D: int
    N_IRREGULAR: int
    P_TOKEN_IRREGULAR: float
    pruning_enabled: bool

    @model_validator(mode="after")
    def _check_feasible(self) -> "LinguisticConfig":
        available = 3**self.D
        needed = N_VERBS_TOTAL + N_WUG
        if needed > available:
            raise ValueError(
                f"D={self.D} gives only {available} radicals, but "
                f"N_VERBS_TOTAL+N_WUG={needed} are needed (plan §1.1 #1: "
                f"D=4 was dropped from the sweep for exactly this reason). "
                f"Increase D or reduce N_VERBS_TOTAL/N_WUG."
            )
        if not (0 < self.N_IRREGULAR < N_VERBS_TOTAL):
            raise ValueError(f"N_IRREGULAR must be in (0, {N_VERBS_TOTAL})")
        if not (0.0 <= self.P_TOKEN_IRREGULAR <= 1.0):
            raise ValueError("P_TOKEN_IRREGULAR must be in [0, 1]")
        return self

    @property
    def n_regular(self) -> int:
        return N_VERBS_TOTAL - self.N_IRREGULAR

    @property
    def p_token_regular(self) -> float:
        return 1.0 - self.P_TOKEN_IRREGULAR

    @property
    def lexicon_density(self) -> float:
        """Fraction of the 3**D radical space occupied by the fixed-size
        lexicon — deliberately NOT held constant across D (plan §1.1 #2),
        logged so the confound is visible in analysis instead of hidden."""
        return N_VERBS_TOTAL / (3**self.D)

    @property
    def id(self) -> str:
        prune = "on" if self.pruning_enabled else "off"
        freq = str(self.P_TOKEN_IRREGULAR).replace(".", "")
        return f"D{self.D}_irr{self.N_IRREGULAR}_freq{freq}_prune{prune}"

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
            "N_IRREGULAR": self.N_IRREGULAR,
            "N_REGULAR": self.n_regular,
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
    prune_interval_epochs: int | None = None
    max_prune_pct: float | None = None

    def as_params(self) -> dict:
        return self.model_dump()


def generate_condition_grid() -> list[LinguisticConfig]:
    """36 conditions: 2 (D) x 3 (N_IRREGULAR) x 3 (P_TOKEN_IRREGULAR) x 2 (pruning)."""
    grid = []
    for D, n_irregular, p_irregular, pruning in itertools.product(
        D_VALUES, N_IRREGULAR_VALUES, P_TOKEN_IRREGULAR_VALUES, PRUNING_VALUES
    ):
        grid.append(
            LinguisticConfig(
                D=D,
                N_IRREGULAR=n_irregular,
                P_TOKEN_IRREGULAR=p_irregular,
                pruning_enabled=pruning,
            )
        )
    return grid
