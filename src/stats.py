from __future__ import annotations

from typing import Any

import pandas as pd
from scipy import stats as scipy_stats

# Pre-registered primary hypotheses (plan §4a) — declared before running
# anything, so these four are confirmatory. Every other breakdown of the
# 36-cell grid is exploratory and does not get a correction here; it lives
# only in the report's descriptive plots.
PRIMARY_HYPOTHESES = {
    "H1_frequency_skew": "params.P_TOKEN_IRREGULAR",
    "H2_irregular_set_size": "params.N_IRREGULAR",
    "H3_dimensionality": "params.D",
    "H4_pruning": "params.pruning_enabled",
}

METRIC_COLUMN = "metrics.wug_regular_prob"


def benjamini_hochberg(
    p_values: dict[str, float], alpha: float = 0.05
) -> dict[str, dict[str, Any]]:
    """Benjamini-Hochberg FDR correction across the given named p-values."""
    items = sorted(p_values.items(), key=lambda kv: kv[1])
    m = len(items)

    adjusted: dict[str, float] = {}
    prev_adj = 1.0
    for rank, (name, p) in reversed(list(enumerate(items, start=1))):
        adj_p = min(prev_adj, p * m / rank)
        prev_adj = adj_p
        adjusted[name] = adj_p

    return {
        name: {
            "p_value": p,
            "adjusted_p_value": adjusted[name],
            "significant_at_0.05": adjusted[name] < alpha,
        }
        for name, p in items
    }


def _kruskal_and_pairwise(df: pd.DataFrame, factor_col: str, metric_col: str) -> dict[str, Any]:
    groups = [g[metric_col].dropna().values for _, g in df.groupby(factor_col) if len(g) > 1]
    if len(groups) < 2:
        return {"omnibus_p": None, "note": "not enough groups/data for a test"}

    omnibus = scipy_stats.kruskal(*groups)

    levels = sorted(df[factor_col].dropna().unique().tolist(), key=str)
    pairwise: dict[str, Any] = {}
    for i in range(len(levels)):
        for j in range(i + 1, len(levels)):
            a = df.loc[df[factor_col] == levels[i], metric_col].dropna()
            b = df.loc[df[factor_col] == levels[j], metric_col].dropna()
            if len(a) < 2 or len(b) < 2:
                continue
            result = scipy_stats.mannwhitneyu(a, b, alternative="two-sided")
            pairwise[f"{levels[i]}_vs_{levels[j]}"] = {
                "u_statistic": float(result.statistic),
                "p_value": float(result.pvalue),
            }

    return {"omnibus_p": float(omnibus.pvalue), "pairwise": pairwise}


def run_primary_hypothesis_tests(confirmation_df: pd.DataFrame) -> dict[str, Any]:
    """H1-H4: Kruskal-Wallis omnibus + pairwise Mann-Whitney U per
    pre-registered linguistic factor, Benjamini-Hochberg corrected across
    the 4 omnibus tests (plan §1.1 #9, §4a)."""
    if confirmation_df.empty or METRIC_COLUMN not in confirmation_df.columns:
        return {}

    per_hypothesis: dict[str, Any] = {}
    omnibus_p_values: dict[str, float] = {}

    for label, col in PRIMARY_HYPOTHESES.items():
        if col not in confirmation_df.columns:
            continue
        result = _kruskal_and_pairwise(confirmation_df, col, METRIC_COLUMN)
        per_hypothesis[label] = result
        if result.get("omnibus_p") is not None:
            omnibus_p_values[label] = result["omnibus_p"]

    if omnibus_p_values:
        corrected = benjamini_hochberg(omnibus_p_values)
        for label, correction in corrected.items():
            per_hypothesis[label]["benjamini_hochberg"] = correction

    return per_hypothesis
