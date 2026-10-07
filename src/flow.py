"""Wires the nodes of src/nodes.py into the experiment DAG.

    validate -> build_dataset (one per distinct dataset)
                   |
                   +-> train (one per run) -> evaluate -> aggregate -> report

Two engines run the same node functions: `prefect` (parallel, observable in
the Prefect UI) and `local` (sequential, no server; used by tests and for
debugging)."""

from __future__ import annotations

from prefect import flow, task
from prefect.cache_policies import NO_CACHE
from prefect.futures import wait
from prefect.task_runners import ProcessPoolTaskRunner

from src.config_loader import ExperimentContext, ResolvedDataset, ResolvedExperiment, RunSpec
from src.exp_report import build_experiment_report
from src.logging_config import get_logger
from src.nodes import (
    experiment_events,
    node_aggregate,
    node_build_dataset,
    node_evaluate,
    node_train,
    prepare_experiment,
)

logger = get_logger(__name__)


def _needed_datasets(exp: ResolvedExperiment) -> list[str]:
    return sorted({r.dataset_id for r in exp.runs})


def _finish(exp: ResolvedExperiment, run_report: bool) -> None:
    if run_report:
        build_experiment_report(exp)
    else:
        node_aggregate(exp)
    experiment_events(exp, stage="done").emit("experiment_done")


# ----------------------------- local engine -------------------------------
def run_local(exp: ResolvedExperiment, force: bool = False, run_report: bool = True) -> None:
    prepare_experiment(exp)
    ctx = exp.context
    infos = {
        ds_id: node_build_dataset(ctx, exp.datasets[ds_id], force)
        for ds_id in _needed_datasets(exp)
    }
    for run in exp.runs:
        info = infos[run.dataset_id]
        manifest = node_train(ctx, run, info, force)
        node_evaluate(ctx, run, manifest, info, force)
    _finish(exp, run_report)


# ----------------------------- prefect engine -----------------------------
@task(name="build-dataset", cache_policy=NO_CACHE)
def build_dataset_task(ctx: ExperimentContext, dataset: ResolvedDataset, force: bool) -> dict:
    return node_build_dataset(ctx, dataset, force)


@task(name="train", cache_policy=NO_CACHE)
def train_task(ctx: ExperimentContext, run: RunSpec, dataset_info: dict, force: bool) -> dict:
    return node_train(ctx, run, dataset_info, force)


@task(name="evaluate", cache_policy=NO_CACHE)
def evaluate_task(
    ctx: ExperimentContext, run: RunSpec, manifest: dict, dataset_info: dict, force: bool
) -> dict | None:
    return node_evaluate(ctx, run, manifest, dataset_info, force)


@flow(name="verb-experiment", validate_parameters=False)
def experiment_flow(exp: ResolvedExperiment, force: bool = False, run_report: bool = True) -> None:
    prepare_experiment(exp)
    ctx = exp.context
    dataset_futures = {
        ds_id: build_dataset_task.submit(ctx, exp.datasets[ds_id], force)
        for ds_id in _needed_datasets(exp)
    }
    train_futures = [
        train_task.submit(ctx, run, dataset_futures[run.dataset_id], force) for run in exp.runs
    ]
    evaluate_futures = [
        evaluate_task.submit(ctx, run, train_future, dataset_futures[run.dataset_id], force)
        for run, train_future in zip(exp.runs, train_futures, strict=True)
    ]
    wait(evaluate_futures)
    for future in evaluate_futures:
        future.result()  # re-raises infrastructure failures (run failures live in manifests)
    _finish(exp, run_report)


def run_prefect(
    exp: ResolvedExperiment, force: bool = False, run_report: bool = True, max_workers: int = 4
) -> None:
    configured = experiment_flow.with_options(
        task_runner=ProcessPoolTaskRunner(max_workers=max_workers)
    )
    configured(exp, force=force, run_report=run_report)
