import pandas as pd

from src.stats import benjamini_hochberg, run_primary_hypothesis_tests


def test_benjamini_hochberg_monotonic_and_bounded():
    result = benjamini_hochberg({"a": 0.001, "b": 0.04, "c": 0.5, "d": 0.9})
    assert result["a"]["significant_at_0.05"] is True
    assert result["d"]["significant_at_0.05"] is False
    for r in result.values():
        assert 0.0 <= r["adjusted_p_value"] <= 1.0


def test_benjamini_hochberg_adjusted_p_never_decreases_with_rank():
    result = benjamini_hochberg({"a": 0.01, "b": 0.02, "c": 0.03, "d": 0.04})
    ordered = [result[k]["adjusted_p_value"] for k in ["a", "b", "c", "d"]]
    assert ordered == sorted(ordered)


def test_run_primary_hypothesis_tests_empty_df_returns_empty():
    assert run_primary_hypothesis_tests(pd.DataFrame()) == {}


def test_run_primary_hypothesis_tests_with_synthetic_data():
    df = pd.DataFrame(
        {
            "metrics.wug_regular_prob": [0.9, 0.92, 0.4, 0.42, 0.3, 0.35],
            "params.P_TOKEN_IRREGULAR": [0.5, 0.5, 0.7, 0.7, 0.9, 0.9],
            "params.N_IRREGULAR": [12, 12, 36, 36, 72, 72],
            "params.D": [6, 6, 6, 6, 6, 6],
            "params.pruning_enabled": [False, False, False, False, False, False],
        }
    )
    result = run_primary_hypothesis_tests(df)
    assert "H1_frequency_skew" in result
    assert result["H1_frequency_skew"]["omnibus_p"] is not None
    assert "benjamini_hochberg" in result["H1_frequency_skew"]
