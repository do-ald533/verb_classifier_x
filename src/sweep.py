from __future__ import annotations

import mlflow
import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
from optuna.storages import JournalStorage
from optuna.storages.journal import JournalFileBackend

from src.config import LinguisticConfig
from src.logging_config import get_logger
from src.objective import make_objective

logger = get_logger(__name__)

EXPERIMENT_NAME = "verb_regularization_sweep"
JOURNAL_PATH = "optuna_journal.log"
TPE_SEED = 20260925  # pinned & logged for reproducibility (plan §1.1 #12)
MIN_COMPLETE_FRACTION = 0.5


def study_name(condition: LinguisticConfig) -> str:
    return f"sweep_{condition.id}"


def run_condition_study(condition: LinguisticConfig) -> optuna.Study:
    """Runs one condition's Optuna study (search phase only — see confirm.py
    for the confirmation phase that produces the actually-reported number).
    Raises if the study turns out degenerate, so src/pipeline.py's
    sweep_condition_task never marks a broken condition as done
    (plan §1.1 #6)."""
    mlflow.set_experiment(EXPERIMENT_NAME)
    storage = JournalStorage(JournalFileBackend(JOURNAL_PATH))

    common_kwargs: dict = dict(
        study_name=study_name(condition),
        storage=storage,
        sampler=TPESampler(seed=TPE_SEED),
        load_if_exists=True,
    )

    if condition.is_multi_objective:
        study = optuna.create_study(directions=["maximize", "minimize"], **common_kwargs)
    else:
        study = optuna.create_study(
            direction="maximize",
            pruner=MedianPruner(n_startup_trials=5, n_warmup_steps=5000),
            **common_kwargs,
        )

    with mlflow.start_run(run_name=condition.id):
        mlflow.set_tag("run_type", "condition")
        mlflow.log_params(condition.as_params())
        mlflow.log_param("tpe_seed", TPE_SEED)
        mlflow.log_param("trial_budget", condition.trial_budget)

        objective = make_objective(condition)
        # TrainingDivergedError (a RuntimeError) is caught here so Optuna
        # records a genuine FAIL state and moves on to the next trial,
        # instead of the whole study crashing (plan §1.1 #5/#9).
        study.optimize(objective, n_trials=condition.trial_budget, catch=(RuntimeError,))

        complete_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
        n_complete = len(complete_trials)
        mlflow.log_metric("n_complete_trials", n_complete)
        mlflow.log_metric("n_total_trials", len(study.trials))

        if n_complete < MIN_COMPLETE_FRACTION * condition.trial_budget:
            mlflow.set_tag("degenerate_study", "true")
            raise RuntimeError(
                f"Condition {condition.id}: only {n_complete}/{condition.trial_budget} "
                f"trials completed (< {MIN_COMPLETE_FRACTION:.0%}) — study considered "
                f"degenerate; not marking this condition's sweep as done."
            )

    return study
