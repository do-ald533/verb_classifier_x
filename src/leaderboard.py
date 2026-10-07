from __future__ import annotations

import pathlib

import mlflow
import optuna
import pandas as pd
from optuna.storages import JournalStorage
from optuna.storages.journal import JournalFileBackend

from src.config import generate_condition_grid
from src.logging_config import get_logger
from src.sweep import EXPERIMENT_NAME, JOURNAL_PATH, study_name

logger = get_logger(__name__)


def load_confirmation_runs() -> pd.DataFrame:
    """All 'best'/leaderboard comparisons read exclusively from
    confirmation-tagged runs (plan §1.1 #4, §10) — never search-phase
    trial/seed runs, which are noisier and at a different granularity."""
    mlflow.set_experiment(EXPERIMENT_NAME)
    return mlflow.search_runs(
        experiment_names=[EXPERIMENT_NAME],
        filter_string="tags.run_type = 'confirmation'",
        order_by=["metrics.wug_regular_prob DESC"],
        output_format="pandas",
    )


def print_leaderboard(top_n: int = 10) -> pd.DataFrame:
    df = load_confirmation_runs()
    if df.empty:
        print("No confirmation runs yet.")
        return df
    top = df.head(top_n)
    cols = ["run_id"] + [
        c for c in top.columns if c.startswith("metrics.") or c.startswith("params.")
    ]
    print(top[cols].to_string(index=False))
    return top


def write_leaderboard_csv(path: str = "reports/leaderboard.csv", top_n: int = 25) -> None:
    df = load_confirmation_runs()
    pathlib.Path(path).parent.mkdir(parents=True, exist_ok=True)
    df.head(top_n).to_csv(path, index=False)
    logger.info("Wrote leaderboard to %s", path)


def check_trial_count_consistency() -> list[str]:
    """Compares Optuna's own trial count per condition against the number of
    'trial'-tagged MLflow runs for that condition, and logs a warning on any
    mismatch instead of letting a silent desync pass unnoticed (plan §1.1 #12)."""
    mismatches: list[str] = []
    storage = JournalStorage(JournalFileBackend(JOURNAL_PATH))
    mlflow.set_experiment(EXPERIMENT_NAME)

    for condition in generate_condition_grid():
        try:
            study = optuna.load_study(study_name=study_name(condition), storage=storage)
        except KeyError:
            continue

        optuna_count = len(study.trials)
        mlflow_runs = mlflow.search_runs(
            experiment_names=[EXPERIMENT_NAME],
            filter_string=(
                f"tags.run_type = 'trial' and tags.mlflow.runName LIKE '{condition.id}_trial%'"
            ),
        )
        mlflow_count = len(mlflow_runs)

        if optuna_count != mlflow_count:
            msg = (
                f"{condition.id}: Optuna reports {optuna_count} trials but MLflow has "
                f"{mlflow_count} 'trial' runs — possible desync"
            )
            logger.warning(msg)
            mismatches.append(msg)

    return mismatches


if __name__ == "__main__":
    print_leaderboard()
    write_leaderboard_csv()
    for warning in check_trial_count_consistency():
        print("WARNING:", warning)
