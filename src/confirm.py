from __future__ import annotations

import statistics

import mlflow
import optuna
from optuna.storages import JournalStorage
from optuna.storages.journal import JournalFileBackend

from src.config import CONFIRMATION_SEEDS, LinguisticConfig, NetworkConfig
from src.logging_config import get_logger
from src.sweep import EXPERIMENT_NAME, JOURNAL_PATH, study_name
from src.train import TrainingDivergedError, train_one_run

logger = get_logger(__name__)

METRIC_NAMES = [
    "wug_regular_prob",
    "wug_regular_rate",
    "wug_entropy",
    "wug_top2_margin",
    "loss",
    "acc_types",
    "regular_acc",
    "irregular_acc",
    "param_norm",
    "active_pct",
]


def _network_config_from_params(params: dict) -> NetworkConfig:
    """params: an Optuna FrozenTrial.params dict (clean, no NaN — values
    only present when they were actually suggested for that trial)."""
    return NetworkConfig(
        hidden_dim=params["hidden_dim"],
        num_hidden_layers=params["num_hidden_layers"],
        optimizer_name=params["optimizer_name"],
        learning_rate=params["learning_rate"],
        momentum=params.get("momentum", 0.0),
        prune_interval_steps=params.get("prune_interval_steps"),
        max_prune_pct=params.get("max_prune_pct"),
    )


def _champion_trials(condition: LinguisticConfig) -> list[optuna.trial.FrozenTrial]:
    storage = JournalStorage(JournalFileBackend(JOURNAL_PATH))
    study = optuna.load_study(study_name=study_name(condition), storage=storage)
    if condition.is_multi_objective:
        return list(study.best_trials)  # Pareto-optimal set — confirm all of them
    return [study.best_trial]


def _confirm_one(condition: LinguisticConfig, network: NetworkConfig, label: str) -> None:
    seed_results = []

    with mlflow.start_run(run_name=f"{condition.id}_confirmation_{label}", nested=True) as run:
        mlflow.set_tag("run_type", "confirmation")
        mlflow.log_params(
            {**condition.as_params(), **network.as_params(), "champion_label": label}
        )

        for seed in CONFIRMATION_SEEDS:
            try:
                result = train_one_run(condition, network, seed)
            except TrainingDivergedError as exc:
                logger.warning("Confirmation %s seed %d diverged: %s", label, seed, exc)
                continue
            seed_results.append(result.final)

        if not seed_results:
            mlflow.set_tag("degenerate_confirmation", "true")
            raise RuntimeError(f"All confirmation seeds diverged for {condition.id} ({label})")

        aggregated = {
            name: statistics.fmean(r[name] for r in seed_results) for name in METRIC_NAMES
        }
        aggregated.update(
            {
                f"{name}_std": (
                    statistics.pstdev(r[name] for r in seed_results)
                    if len(seed_results) > 1
                    else 0.0
                )
                for name in METRIC_NAMES
            }
        )
        mlflow.log_metrics(aggregated)
        mlflow.log_metric("n_confirmation_seeds", len(seed_results))

        # "Best so far" is judged only against other confirmation runs, and
        # explicitly excludes this run itself so a condition's very first
        # confirmation is correctly flagged as best rather than compared
        # against its own just-logged row (plan §1.1 #7).
        prior_best = mlflow.search_runs(
            experiment_names=[EXPERIMENT_NAME],
            filter_string=(
                "tags.run_type = 'confirmation' "
                f"and attributes.run_id != '{run.info.run_id}'"
            ),
            order_by=["metrics.wug_regular_prob DESC"],
            max_results=1,
        )
        is_new_best = (
            prior_best.empty
            or aggregated["wug_regular_prob"] > prior_best.iloc[0]["metrics.wug_regular_prob"]
        )
        mlflow.set_tag("is_global_best", str(is_new_best))


def run_confirmation(condition: LinguisticConfig) -> None:
    """Re-runs the Optuna-selected champion config(s) for this condition on
    CONFIRMATION_SEEDS (20 fresh seeds), producing the unambiguous,
    well-powered number that all "best"/leaderboard comparisons use
    (plan §1.1 #4, §5, §10)."""
    mlflow.set_experiment(EXPERIMENT_NAME)
    champions = _champion_trials(condition)

    with mlflow.start_run(run_name=f"{condition.id}_confirm_parent"):
        mlflow.set_tag("run_type", "confirmation_parent")
        mlflow.log_param("n_champions", len(champions))

        for i, trial in enumerate(champions):
            network = _network_config_from_params(trial.params)
            label = f"pareto{i}" if condition.is_multi_objective else "best"
            _confirm_one(condition, network, label)
