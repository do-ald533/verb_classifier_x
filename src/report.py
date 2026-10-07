from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import optuna
import pandas as pd
import seaborn as sns
import torch
from optuna.storages import JournalStorage
from optuna.storages.journal import JournalFileBackend
from pandas.plotting import parallel_coordinates

from src.config import (
    CONFIRMATION_SEEDS,
    LinguisticConfig,
    N_WUG,
    NetworkConfig,
    generate_condition_grid,
)
from src.leaderboard import (
    check_trial_count_consistency,
    load_confirmation_runs,
    write_leaderboard_csv,
)
from src.logging_config import get_logger
from src.stats import run_primary_hypothesis_tests
from src.sweep import EXPERIMENT_NAME, JOURNAL_PATH, study_name
from src.train import train_one_run

logger = get_logger(__name__)

sns.set_theme(style="whitegrid", palette="colorblind")

LINGUISTIC_FACTOR_COLUMNS = [
    ("params.D", "Radical dimensionality (D)"),
    ("params.LEXICON_SIZE", "Lexicon size"),
    ("params.P_IRREGULAR_TYPES", "Irregular type proportion"),
    ("params.P_TOKEN_IRREGULAR", "Irregular token-frequency skew"),
    ("params.pruning_enabled", "Pruning enabled"),
]


def _savefig(fig: plt.Figure, out_dir: Path, name: str) -> Path:
    path = out_dir / "figures" / f"{name}.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def _network_config_from_mlflow_row(row: pd.Series) -> NetworkConfig:
    """MLflow params are always logged as strings, and the union of columns
    across confirmation runs means some are NaN for a given row (e.g. Adam
    runs have no 'momentum'). Coerce explicitly rather than relying on
    implicit type coercion of unfamiliar values."""

    def get(name):
        col = f"params.{name}"
        if col not in row.index or pd.isna(row[col]):
            return None
        return row[col]

    momentum = get("momentum")
    prune_interval = get("prune_interval_steps")
    max_prune_pct = get("max_prune_pct")

    return NetworkConfig(
        hidden_dim=int(float(get("hidden_dim"))),
        num_hidden_layers=int(float(get("num_hidden_layers"))),
        optimizer_name=get("optimizer_name"),
        learning_rate=float(get("learning_rate")),
        momentum=float(momentum) if momentum is not None else 0.0,
        prune_interval_steps=int(float(prune_interval)) if prune_interval is not None else None,
        max_prune_pct=float(max_prune_pct) if max_prune_pct is not None else None,
    )


def plot_leaderboard_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "_no confirmation runs yet_"
    cols = ["run_id"] + [c for c, _ in LINGUISTIC_FACTOR_COLUMNS if c in df.columns]
    cols += [
        c for c in ["metrics.wug_regular_prob", "metrics.wug_regular_prob_std"] if c in df.columns
    ]
    top = df.head(10)[cols]
    try:
        return top.to_markdown(index=False)
    except ImportError:
        return top.to_string(index=False)


def plot_best_so_far(out_dir: Path) -> Path | None:
    """Running max of the search-phase objective vs. trial index — a
    convergence diagnostic, not a leaderboard (plan §6.2)."""
    trials_df = mlflow.search_runs(
        experiment_names=[EXPERIMENT_NAME], filter_string="tags.run_type = 'trial'"
    )
    if trials_df.empty or "metrics.wug_regular_prob_mean" not in trials_df.columns:
        return None

    trials_df = trials_df.sort_values("start_time")
    running_best = trials_df["metrics.wug_regular_prob_mean"].cummax()

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(range(len(running_best)), running_best.values)
    ax.set_xlabel("Trial index (global, all conditions)")
    ax.set_ylabel("Best wug_regular_prob so far")
    ax.set_title("Search convergence")
    return _savefig(fig, out_dir, "best_so_far")


def plot_linguistic_factor_effects(confirmation_df: pd.DataFrame, out_dir: Path) -> Path | None:
    """Full confirmation-phase distributions per factor level, not a
    pre-filtered 'best' point — keeps pruning-on and pruning-off conditions
    on the same comparison basis (plan §1.1 #10)."""
    if confirmation_df.empty:
        return None
    factors = [(c, label) for c, label in LINGUISTIC_FACTOR_COLUMNS if c in confirmation_df.columns]
    if not factors or "metrics.wug_regular_prob" not in confirmation_df.columns:
        return None

    fig, axes = plt.subplots(1, len(factors), figsize=(4.5 * len(factors), 4.5), sharey=True)
    if len(factors) == 1:
        axes = [axes]
    for ax, (col, label) in zip(axes, factors):
        sns.boxplot(data=confirmation_df, x=col, y="metrics.wug_regular_prob", ax=ax)
        sns.stripplot(
            data=confirmation_df,
            x=col,
            y="metrics.wug_regular_prob",
            ax=ax,
            color="black",
            alpha=0.4,
            size=3,
        )
        ax.set_xlabel(label)
        ax.set_ylabel("wug_regular_prob (confirmation, 20 seeds)")
    fig.suptitle("Effect of each linguistic factor — full confirmation-phase distributions")
    fig.tight_layout()
    return _savefig(fig, out_dir, "linguistic_factor_effects")


def plot_confusion_matrix(
    condition: LinguisticConfig, network: NetworkConfig, seed: int, out_dir: Path
) -> Path:
    """Overregularization pattern: true irregular class vs. predicted class
    — direct visual evidence of the phenomenon this notebook studies, which
    the source notebook's own printout never showed (plan §6.4)."""
    result = train_one_run(condition, network, seed, return_model=True)
    model, dataset = result.model, result.dataset

    with torch.no_grad():
        preds = model(dataset.X_types).argmax(dim=-1)

    irregular_true = dataset.y_types[dataset.n_regular :].numpy()
    irregular_pred = preds[dataset.n_regular :].numpy()

    matrix = np.zeros((dataset.n_classes, dataset.n_classes), dtype=int)
    for t, p in zip(irregular_true, irregular_pred):
        matrix[t, p] += 1

    fig, ax = plt.subplots(figsize=(8, 7))
    sns.heatmap(matrix, cmap="rocket_r", ax=ax, cbar_kws={"label": "count"})
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("True irregular class")
    ax.set_title("Overregularization pattern (irregular lexicon items only)")
    return _savefig(fig, out_dir, "overregularization_confusion_matrix")


def plot_wug_response_distribution(
    condition: LinguisticConfig, network: NetworkConfig, seed: int, out_dir: Path
) -> Path:
    """How wugs distribute across all classes, plotted against the
    untrained-baseline regular rate as a null-comparison line (plan §1.1 #8,
    §6.5) — richer than the scalar wug_regular_rate alone."""
    trained = train_one_run(condition, network, seed, return_model=True)
    with torch.no_grad():
        trained_preds = trained.model(trained.dataset.X_wug).argmax(dim=-1).numpy()
    trained_counts = np.bincount(trained_preds, minlength=condition.n_classes)

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.bar(range(condition.n_classes), trained_counts)
    if "baseline_wug_regular_rate" in trained.baseline:
        baseline_regular_count = trained.baseline["baseline_wug_regular_rate"] * N_WUG
        ax.axhline(
            baseline_regular_count,
            color="red",
            linestyle="--",
            label=f"untrained baseline: {baseline_regular_count:.1f} of {N_WUG} on class 0",
        )
        ax.legend()
    ax.set_xlabel("Predicted class")
    ax.set_ylabel(f"# of {N_WUG} wugs")
    ax.set_title("Wug response distribution (trained champion model)")
    if condition.n_classes <= 40:
        ax.set_xticks(range(condition.n_classes))
    return _savefig(fig, out_dir, "wug_response_distribution")


def plot_learning_curves(
    condition: LinguisticConfig, network: NetworkConfig, seed: int, out_dir: Path
) -> Path:
    result = train_one_run(condition, network, seed)
    history = pd.DataFrame(result.history)

    fig, axes = plt.subplots(2, 2, figsize=(10, 7), sharex=True)
    for ax, col in zip(axes.flat, ["loss", "regular_acc", "irregular_acc", "wug_regular_prob"]):
        ax.plot(history["tokens_seen"], history[col])
        ax.set_title(col)
        ax.set_xlabel("tokens seen")
    fig.tight_layout()
    return _savefig(fig, out_dir, "learning_curves")


def _load_condition_study(condition: LinguisticConfig) -> optuna.Study | None:
    storage = JournalStorage(JournalFileBackend(JOURNAL_PATH))
    try:
        return optuna.load_study(study_name=study_name(condition), storage=storage)
    except KeyError:
        return None


def plot_param_importance(condition: LinguisticConfig, out_dir: Path) -> Path | None:
    study = _load_condition_study(condition)
    if study is None:
        return None
    n_complete = len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])
    if n_complete < 2:
        return None

    import optuna.visualization.matplotlib as opt_mpl

    ax = opt_mpl.plot_param_importances(study)
    return _savefig(ax.figure, out_dir, f"param_importance_{condition.id}")


def plot_pareto_front(condition: LinguisticConfig, out_dir: Path) -> Path | None:
    if not condition.is_multi_objective:
        return None
    study = _load_condition_study(condition)
    if study is None:
        return None

    import optuna.visualization.matplotlib as opt_mpl

    ax = opt_mpl.plot_pareto_front(study)
    return _savefig(ax.figure, out_dir, f"pareto_front_{condition.id}")


def plot_correlation_heatmap(condition: LinguisticConfig, out_dir: Path) -> Path | None:
    study = _load_condition_study(condition)
    if study is None:
        return None
    trials_df = study.trials_dataframe()
    if trials_df.empty:
        return None

    value_col = "value" if "value" in trials_df.columns else "values_0"
    numeric_cols = [c for c in trials_df.columns if c.startswith("params_")]
    if value_col not in trials_df.columns or not numeric_cols:
        return None

    corr_df = trials_df[[*numeric_cols, value_col]].select_dtypes(include="number")
    if corr_df.shape[1] < 2:
        return None
    corr = corr_df.corr()

    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="vlag", center=0, ax=ax)
    ax.set_title(f"Hyperparameter correlation — {condition.id}")
    return _savefig(fig, out_dir, f"correlation_{condition.id}")


def plot_parallel_coordinates(condition: LinguisticConfig, out_dir: Path) -> Path | None:
    study = _load_condition_study(condition)
    if study is None:
        return None
    trials_df = study.trials_dataframe()
    value_col = "value" if "value" in trials_df.columns else "values_0"
    if trials_df.empty or value_col not in trials_df.columns:
        return None

    param_cols = [c for c in trials_df.columns if c.startswith("params_")]
    plot_df = trials_df[param_cols].select_dtypes(include="number").copy()
    if plot_df.shape[1] < 2:
        return None

    threshold = trials_df[value_col].quantile(0.9)
    plot_df["top_decile"] = np.where(trials_df[value_col] >= threshold, "top decile", "other")

    fig, ax = plt.subplots(figsize=(9, 5))
    parallel_coordinates(plot_df, "top_decile", ax=ax, color=["#999999", "#d62728"])
    ax.set_title(f"Trial hyperparameters — {condition.id}")
    return _savefig(fig, out_dir, f"parallel_coordinates_{condition.id}")


def build_runtime_table() -> pd.DataFrame:
    trials_df = mlflow.search_runs(
        experiment_names=[EXPERIMENT_NAME], filter_string="tags.run_type = 'trial'"
    )
    if trials_df.empty:
        return pd.DataFrame()
    group_cols = [
        c
        for c in [
            "params.D",
            "params.LEXICON_SIZE",
            "params.P_IRREGULAR_TYPES",
            "params.P_TOKEN_IRREGULAR",
            "params.pruning_enabled",
        ]
        if c in trials_df.columns
    ]
    if not group_cols or "metrics.wall_clock_seconds_mean" not in trials_df.columns:
        return pd.DataFrame()
    return trials_df.groupby(group_cols, as_index=False)["metrics.wall_clock_seconds_mean"].mean()


def build_reproducibility_block(mismatches: list[str]) -> dict:
    return {
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "tpe_seed": 20260925,
        "trial_mlflow_consistency_warnings": mismatches or ["none"],
    }


def _champion_from_leaderboard(
    confirmation_df: pd.DataFrame,
) -> tuple[LinguisticConfig, NetworkConfig, dict]:
    best_row = confirmation_df.iloc[0]
    condition = LinguisticConfig(
        D=int(float(best_row["params.D"])),
        LEXICON_SIZE=int(float(best_row["params.LEXICON_SIZE"])),
        P_IRREGULAR_TYPES=float(best_row["params.P_IRREGULAR_TYPES"]),
        P_TOKEN_IRREGULAR=float(best_row["params.P_TOKEN_IRREGULAR"]),
        pruning_enabled=str(best_row["params.pruning_enabled"]) == "True",
    )
    network = _network_config_from_mlflow_row(best_row)
    return condition, network, best_row.to_dict()


def build_report(experiment: str = EXPERIMENT_NAME, out: str = "reports") -> None:
    """Builds the full report (leaderboard, all figures, stats, report.md).

    Extracted from main() so it can be called directly by src/pipeline.py's
    Prefect flow, not just from the command line."""
    mlflow.set_experiment(experiment)
    out_dir = Path(out)
    out_dir.mkdir(parents=True, exist_ok=True)

    write_leaderboard_csv(str(out_dir / "leaderboard.csv"))
    confirmation_df = load_confirmation_runs()
    mismatches = check_trial_count_consistency()

    sections = ["# Verb Regularization Sweep — Report\n"]
    sections.append("## Leaderboard (top 10 confirmation runs)\n")
    sections.append(plot_leaderboard_table(confirmation_df))

    plot_best_so_far(out_dir)
    plot_linguistic_factor_effects(confirmation_df, out_dir)

    if not confirmation_df.empty:
        champion_condition, champion_network, champion_row = _champion_from_leaderboard(
            confirmation_df
        )
        champion_seed = CONFIRMATION_SEEDS[0]

        plot_confusion_matrix(champion_condition, champion_network, champion_seed, out_dir)
        plot_wug_response_distribution(champion_condition, champion_network, champion_seed, out_dir)
        plot_learning_curves(champion_condition, champion_network, champion_seed, out_dir)

        (out_dir / "best_config.json").write_text(
            json.dumps(
                {
                    "linguistic": champion_condition.model_dump(),
                    "network": champion_network.model_dump(),
                    "wug_regular_prob": champion_row.get("metrics.wug_regular_prob"),
                    "run_id": champion_row.get("run_id"),
                },
                indent=2,
                default=str,
            )
        )

    for condition in generate_condition_grid():
        plot_param_importance(condition, out_dir)
        plot_pareto_front(condition, out_dir)
        plot_correlation_heatmap(condition, out_dir)
        plot_parallel_coordinates(condition, out_dir)

    runtime_table = build_runtime_table()
    repro = build_reproducibility_block(mismatches)
    stats_results = run_primary_hypothesis_tests(confirmation_df)

    sections.append("\n## Reproducibility\n")
    sections.append(f"```json\n{json.dumps(repro, indent=2)}\n```\n")
    sections.append("\n## Runtime (mean wall-clock seconds per trial, by condition)\n")
    sections.append(
        runtime_table.to_markdown(index=False)
        if not runtime_table.empty
        else "_no trial runtime data yet_"
    )
    sections.append(
        "\n## Primary hypothesis tests (H1-H4, pre-registered — plan §4a, "
        "Benjamini-Hochberg corrected)\n"
    )
    sections.append(f"```json\n{json.dumps(stats_results, indent=2, default=str)}\n```\n")
    sections.append(
        "\n_Everything above is confirmatory. Every other breakdown of the 36-cell "
        "grid shown in `figures/` is exploratory/descriptive and not corrected for "
        "multiple comparisons (plan §1.1 #9)._\n"
    )
    sections.append("\n## Figures\n")
    sections.append(
        "See `figures/` for: best-so-far convergence, per-factor effects, "
        "overregularization confusion matrix, wug response distribution, learning "
        "curves, hyperparameter importance, Pareto fronts, correlation heatmaps, "
        "and parallel-coordinates plots.\n"
    )

    (out_dir / "report.md").write_text("\n".join(sections))
    logger.info("Report written to %s", out_dir / "report.md")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", default=EXPERIMENT_NAME)
    parser.add_argument("--out", default="reports")
    args = parser.parse_args()
    build_report(experiment=args.experiment, out=args.out)


if __name__ == "__main__":
    main()
