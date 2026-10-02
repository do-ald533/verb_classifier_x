import pytest
from pydantic import ValidationError

from src.config import N_WUG, LinguisticConfig, generate_condition_grid


def test_d4_is_rejected_as_infeasible():
    """3**4=81 radicals can't hold the lexicon + wugs."""
    with pytest.raises(ValidationError):
        LinguisticConfig(D=4, LEXICON_SIZE=699, P_IRREGULAR_TYPES=0.05, P_TOKEN_IRREGULAR=0.7, pruning_enabled=False)


def test_d6_is_feasible_and_uses_whole_space():
    cfg = LinguisticConfig(D=6, LEXICON_SIZE=699, P_IRREGULAR_TYPES=0.05, P_TOKEN_IRREGULAR=0.7, pruning_enabled=False)
    assert cfg.N_IRREGULAR == 35
    assert cfg.n_regular == 699 - 35
    assert cfg.n_classes == 36
    assert 3**cfg.D == cfg.LEXICON_SIZE + N_WUG


def test_condition_grid_size_and_unique_ids():
    grid = generate_condition_grid()
    assert all(3**c.D >= c.LEXICON_SIZE + N_WUG for c in grid)
    assert any(c.LEXICON_SIZE == 10000 for c in grid)
    assert len({c.id for c in grid}) == len(grid)


def test_lexicon_density_differs_across_d_and_is_not_hidden():
    """Regression test for plan §1.1 #2: density is a deliberate, visible
    confound, not something the sweep silently ignores."""
    d6 = LinguisticConfig(D=6, LEXICON_SIZE=699, P_IRREGULAR_TYPES=0.05, P_TOKEN_IRREGULAR=0.7, pruning_enabled=False)
    d8 = LinguisticConfig(D=8, LEXICON_SIZE=699, P_IRREGULAR_TYPES=0.05, P_TOKEN_IRREGULAR=0.7, pruning_enabled=False)
    assert d6.lexicon_density > d8.lexicon_density
    assert "lexicon_density" in d6.as_params()


def test_invalid_p_token_irregular_rejected():
    with pytest.raises(ValidationError):
        LinguisticConfig(D=6, LEXICON_SIZE=699, P_IRREGULAR_TYPES=0.05, P_TOKEN_IRREGULAR=1.5, pruning_enabled=False)


def test_lexicon_larger_than_radical_space_rejected():
    with pytest.raises(ValidationError):
        LinguisticConfig(
            D=8, LEXICON_SIZE=10000, P_IRREGULAR_TYPES=0.05, P_TOKEN_IRREGULAR=0.7, pruning_enabled=False
        )
