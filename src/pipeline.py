from __future__ import annotations

import argparse
from pathlib import Path

from prefect import flow, task
from prefect.futures import PrefectFuture, wait
from prefect.task_runners import ProcessPoolTaskRunner

from src.config import LinguisticConfig, generate_condition_grid
from src.confirm import run_confirmation
from src.logging_config import get_logger
from src.report import build_report
from src.sweep import EXPERIMENT_NAME, run_condition_study

logger = get_logger(__name__)


@task(name="sweep-condition", retries=1, retry_delay_seconds=5)
def sweep_condition_task(condition: LinguisticConfig) -> LinguisticConfig:
    """Runs one condition's Optuna search phase.

    Skips work already completed on a prior, interrupted run: the marker
    check is done here, in the task itself, rather than relying on the
    orchestrator for it, so resumability does not depend on which
    orchestrator happens to be running this (plan section 1.1, item 6).
    """
    marker = Path(f"reports/.done/sweep_{condition.id}.done")
    if marker.exists():
        logger.info("Sweep already done for %s, skipping", condition.id)
        return condition

    run_condition_study(condition)
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.touch()
    return condition


@task(name="confirm-condition", retries=1, retry_delay_seconds=5)
def confirm_condition_task(condition: LinguisticConfig) -> str:
    """Runs one condition's confirmation phase.

    Takes the upstream sweep task's future as its argument so Prefect waits
    for that specific condition's sweep to finish before starting its
    confirmation, while different conditions still run concurrently with
    each other.
    """
    marker = Path(f"reports/.done/confirm_{condition.id}.done")
    if not marker.exists():
        run_confirmation(condition)
        marker.parent.mkdir(parents=True, exist_ok=True)
        marker.touch()
    return condition.id


@task(name="build-report")
def build_report_task(experiment: str, out: str) -> None:
    build_report(experiment=experiment, out=out)


@flow(name="verb-regularization-pipeline")
def pipeline_flow(
    experiment: str = EXPERIMENT_NAME, out: str = "reports", run_report: bool = True
) -> list[str]:
    """The full experiment pipeline as a Prefect DAG.

    Each of the 36 linguistic conditions becomes two chained tasks (sweep,
    then confirm). Conditions run in parallel with each other; the sweep and
    confirm steps within one condition run in the order shown, because the
    confirm task depends on the sweep task's future for that same condition.
    The report step runs once, after every condition has confirmed.

    Parallelism and its degree are controlled entirely by the task_runner
    the caller attaches with with_options (see main() below) - this
    function itself makes no assumption about how many workers exist.
    """
    conditions = generate_condition_grid()

    sweep_futures: list[PrefectFuture] = [
        sweep_condition_task.submit(condition) for condition in conditions
    ]
    confirm_futures: list[PrefectFuture] = [
        confirm_condition_task.submit(future) for future in sweep_futures
    ]
    wait(confirm_futures)
    results = [future.result() for future in confirm_futures]  # re-raises any task failure

    if run_report:
        build_report_task(experiment, out)

    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the verb regularization experiment pipeline.")
    parser.add_argument(
        "--stage",
        choices=["sweep-and-confirm", "report", "all"],
        default="all",
        help="'sweep-and-confirm' runs the DAG without the report step; "
        "'report' only rebuilds the report from existing runs; "
        "'all' (default) runs everything.",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=4,
        help="Number of conditions to process in parallel, each in its own process "
        "(matches JournalStorage, which is safe for concurrent multi-process writes).",
    )
    parser.add_argument("--experiment", default=EXPERIMENT_NAME)
    parser.add_argument("--out", default="reports")
    args = parser.parse_args()

    if args.stage == "report":
        build_report(experiment=args.experiment, out=args.out)
        return

    run_report = args.stage == "all"
    configured_flow = pipeline_flow.with_options(
        task_runner=ProcessPoolTaskRunner(max_workers=args.max_workers)
    )
    configured_flow(experiment=args.experiment, out=args.out, run_report=run_report)


if __name__ == "__main__":
    main()
