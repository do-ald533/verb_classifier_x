from __future__ import annotations

import statistics
from typing import Any

import mlflow
import optuna

from src.config import LinguisticConfig, NetworkConfig, SEARCH_SEEDS
from src.logging_config import get_logger
from src.train import train_one_run

logger = get_logger(__name__)

TRIAL_METRIC_NAMES = [
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


def _suggest_network_config(trial: optuna.Trial, linguistic: LinguisticConfig) -> NetworkConfig:
    optimizer_name = trial.suggest_categorical("optimizer_name", ["SGD", "Adam"])
    hidden_dim = trial.suggest_int("hidden_dim", 4, 32, log=True)
    num_hidden_layers = trial.suggest_int("num_hidden_layers", 1, 3)
    learning_rate = trial.suggest_float("learning_rate", 1e-3, 3e-1, log=True)
    momentum = trial.suggest_float("momentum", 0.0, 0.9) if optimizer_name == "SGD" else 0.0

    prune_interval_steps = None
    max_prune_pct = None
    if linguistic.pruning_enabled:
        prune_interval_steps = trial.suggest_int("prune_interval_steps", 100, 20000, log=True)
        max_prune_pct = trial.suggest_float("max_prune_pct", 1.0, 50.0)

    return NetworkConfig(
        hidden_dim=hidden_dim,
        num_hidden_layers=num_hidden_layers,
        optimizer_name=optimizer_name,
        learning_rate=learning_rate,
        momentum=momentum,
        prune_interval_steps=prune_interval_steps,
        max_prune_pct=max_prune_pct,
    )


def make_objective(linguistic: LinguisticConfig):
    """Builds an Optuna objective for one linguistic condition. Runs
    SEARCH_SEEDS (cheap, 5) per trial purely to rank hyperparameters;
    the winning config is re-measured properly later in confirm.py on a
    larger seed batch (plan §1.1 #4) — this objective's job is only to
    search, never to produce the number that gets reported as "best"."""

    def objective(trial: optuna.Trial) -> Any:
        network = _suggest_network_config(trial, linguistic)
        seed_results = []

        with mlflow.start_run(run_name=f"{linguistic.id}_trial{trial.number}", nested=True):
            mlflow.set_tag("run_type", "trial")
            mlflow.log_params(
                {**linguistic.as_params(), **network.as_params(), "trial_number": trial.number}
            )

            for seed in SEARCH_SEEDS:
                with mlflow.start_run(
                    run_name=f"{linguistic.id}_trial{trial.number}_seed{seed}", nested=True
                ):
                    mlflow.set_tag("run_type", "seed")
                    mlflow.log_param("seed", seed)

                    def report_intermediate(step_metrics: dict) -> None:
                        mlflow.log_metrics(
                            {k: v for k, v in step_metrics.items() if k != "tokens_seen"},
                            step=int(step_metrics["tokens_seen"]),
                        )
                        # Pruning (early-stopping bad trials) is reserved for
                        # single-objective conditions only — multi-objective
                        # (accuracy vs. sparsity) pruning semantics in Optuna
                        # were not independently verified this session, so we
                        # don't rely on them (plan §1.1 #11).
                        if not linguistic.is_multi_objective:
                            trial.report(
                                step_metrics["wug_regular_prob"], step=step_metrics["tokens_seen"]
                            )
                            if trial.should_prune():
                                raise optuna.TrialPruned()

                    result = train_one_run(linguistic, network, seed, report_intermediate)

                    mlflow.log_metrics({f"baseline_{k}": v for k, v in result.baseline.items()})
                    mlflow.log_metric("wall_clock_seconds", result.wall_clock_seconds)
                    seed_results.append(result.final)

            aggregated = {
                f"{name}_mean": statistics.fmean(r[name] for r in seed_results)
                for name in TRIAL_METRIC_NAMES
            }
            aggregated.update(
                {
                    f"{name}_std": (
                        statistics.pstdev(r[name] for r in seed_results)
                        if len(seed_results) > 1
                        else 0.0
                    )
                    for name in TRIAL_METRIC_NAMES
                }
            )
            mlflow.log_metrics(aggregated)

        if linguistic.is_multi_objective:
            return aggregated["wug_regular_prob_mean"], aggregated["active_pct_mean"]
        return aggregated["wug_regular_prob_mean"]

    return objective
