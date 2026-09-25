import pytest
from pydantic import ValidationError

from src.config import N_VERBS_TOTAL, N_WUG, LinguisticConfig, generate_condition_grid


def test_d4_is_rejected_as_infeasible():
    """Regression test for plan §1.1 #1: 3**4=81 radicals can't hold the
    fixed lexicon + wugs."""
    with pytest.raises(ValidationError):
        LinguisticConfig(D=4, N_IRREGULAR=36, P_TOKEN_IRREGULAR=0.7, pruning_enabled=False)


def test_d6_is_feasible_and_uses_whole_space():
    cfg = LinguisticConfig(D=6, N_IRREGULAR=36, P_TOKEN_IRREGULAR=0.7, pruning_enabled=False)
    assert cfg.n_regular == N_VERBS_TOTAL - 36
    assert 3**cfg.D == N_VERBS_TOTAL + N_WUG


def test_condition_grid_size_and_unique_ids():
    grid = generate_condition_grid()
    assert len(grid) == 2 * 3 * 3 * 2
    assert len({c.id for c in grid}) == len(grid)


def test_lexicon_density_differs_across_d_and_is_not_hidden():
    """Regression test for plan §1.1 #2: density is a deliberate, visible
    confound, not something the sweep silently ignores."""
    d6 = LinguisticConfig(D=6, N_IRREGULAR=36, P_TOKEN_IRREGULAR=0.7, pruning_enabled=False)
    d8 = LinguisticConfig(D=8, N_IRREGULAR=36, P_TOKEN_IRREGULAR=0.7, pruning_enabled=False)
    assert d6.lexicon_density > d8.lexicon_density
    assert "lexicon_density" in d6.as_params()


def test_invalid_p_token_irregular_rejected():
    with pytest.raises(ValidationError):
        LinguisticConfig(D=6, N_IRREGULAR=36, P_TOKEN_IRREGULAR=1.5, pruning_enabled=False)
