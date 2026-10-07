"""DAG nodes. Each node is a plain function: inputs and outputs live on disk
under content hashes, so a node is skipped when its output already exists
(resumability) and any number can run in parallel. Orchestration (Prefect or
the sequential local engine) lives in src/flow.py and only wires these.

    build_dataset  ->  train  ->  evaluate  ->  aggregate  ->  report

Layout (root = experiment.output_dir):
    root/datasets/<dataset_hash>/        spec.json data.pt stats.json
    root/<experiment>/config.json        the user's file, verbatim
    root/<experiment>/resolved_config.json
    root/<experiment>/events.jsonl       experiment-level event stream
    root/<experiment>/trainings/<run_hash>/
        spec.json manifest.json events.jsonl history.jsonl model.pt
        metrics.json predictions.npz
    root/<experiment>/{runs.csv,aggregate.csv}
"""

from __future__ import annotations

import contextlib
import os
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from src.config_loader import ResolvedExperiment, RunSpec
from src.data import (
    build_dataset,
    dataset_dir,
    is_complete,
    load_dataset,
    save_dataset,
)
from src.evaluation import compute_metrics, predictions, tokens_to_accuracy
from src.model import build_model
from src.obs import (
    EventLog,
    Stopwatch,
    environment_info,
    mirror_to_mlflow,
    now_iso,
    read_json,
    read_jsonl,
    write_json,
    write_jsonl,
)
from src.specs import flatten
from src.train import TrainingDivergedError, train_model

REUSABLE_STATUSES = {"ok", "diverged"}  # deterministic outcomes; "error" gets re-run


@dataclass(frozen=True)
class Layout:
    root: Path
    experiment: str

    @property
    def exp_dir(self) -> Path:
        return self.root / self.experiment

    @property
    def events_path(self) -> Path:
        return self.exp_dir / "events.jsonl"

    def training_dir(self, run_hash: str) -> Path:
        return self.exp_dir / "trainings" / run_hash

    @staticmethod
    def of(exp: ResolvedExperiment) -> Layout:
        return Layout(Path(exp.output_dir), exp.name)


def experiment_events(exp: ResolvedExperiment, **context: Any) -> EventLog:
    layout = Layout.of(exp)
    return EventLog([layout.events_path], experiment=exp.name, **context)


def prepare_experiment(exp: ResolvedExperiment) -> None:
    """Persists the config and its resolved form next to the results."""
    layout = Layout.of(exp)
    layout.exp_dir.mkdir(parents=True, exist_ok=True)
    write_json(layout.exp_dir / "config.json", exp.raw)
    write_json(layout.exp_dir / "resolved_config.json", exp.summary())
    experiment_events(exp, stage="setup").emit(
        "experiment_start",
        config_hash=exp.config_hash,
        n_datasets=len(exp.datasets),
        n_networks=len(exp.networks),
        n_runs=len(exp.runs),
        environment=environment_info(),
    )


# --------------------------------------------------------------------------
# node 1: dataset
# --------------------------------------------------------------------------
def node_build_dataset(exp: ResolvedExperiment, dataset_id: str, force: bool = False) -> dict:
    ds = exp.datasets[dataset_id]
    layout = Layout.of(exp)
    events = experiment_events(exp, stage="dataset", dataset_id=dataset_id, dataset_hash=ds.hash)

    if ds.path is not None:  # user-supplied, pre-existing dataset
        directory = Path(ds.path)
        load_dataset(directory)  # fail fast if unreadable
        events.emit("dataset_reused", path=str(directory), reason="explicit path")
        return {"id": dataset_id, "hash": ds.hash, "dir": str(directory), "cached": True}

    directory = dataset_dir(layout.root, ds.spec)
    if is_complete(directory) and not force:
        events.emit("dataset_reused", path=str(directory), reason="content hash exists")
        return {"id": dataset_id, "hash": ds.hash, "dir": str(directory), "cached": True}

    watch = Stopwatch()
    events.emit("dataset_build_start", spec=ds.spec.model_dump(mode="json"))
    if force and directory.exists():
        import shutil

        shutil.rmtree(directory)
    saved = save_dataset(build_dataset(ds.spec), layout.root)
    stats = read_json(saved / "stats.json")
    events.emit("dataset_built", path=str(saved), seconds=watch.seconds, stats=stats)
    return {"id": dataset_id, "hash": ds.hash, "dir": str(saved), "cached": False}


# --------------------------------------------------------------------------
# node 2: train
# --------------------------------------------------------------------------
def _run_events(exp: ResolvedExperiment, run: RunSpec, stage: str) -> EventLog:
    layout = Layout.of(exp)
    return EventLog(
        [layout.training_dir(run.run_hash) / "events.jsonl", layout.events_path],
        experiment=exp.name,
        stage=stage,
        dataset_id=run.dataset_id,
        network_id=run.network_id,
        seed=run.seed,
        run_hash=run.run_hash,
        dataset_hash=run.dataset_hash,
    )


def node_train(
    exp: ResolvedExperiment, run: RunSpec, dataset_info: dict, force: bool = False
) -> dict:
    """Trains one run. Never raises for a run-level failure: the outcome
    (ok / diverged / error) is recorded in manifest.json so the DAG keeps
    going and the report can show what failed and why."""
    layout = Layout.of(exp)
    run_dir = layout.training_dir(run.run_hash)
    manifest_path = run_dir / "manifest.json"
    run_dir.mkdir(parents=True, exist_ok=True)
    events = _run_events(exp, run, "train")

    if manifest_path.exists() and not force:
        manifest = read_json(manifest_path)
        if manifest.get("status") in REUSABLE_STATUSES:
            events.emit("train_reused", status=manifest["status"], reason="content hash exists")
            return manifest

    torch.set_num_threads(int(os.environ.get("VERB_TORCH_THREADS", "1")))
    watch = Stopwatch()
    write_json(
        run_dir / "spec.json",
        {
            "run_hash": run.run_hash,
            "dataset_id": run.dataset_id,
            "dataset_hash": run.dataset_hash,
            "dataset_dir": dataset_info["dir"],
            "network_id": run.network_id,
            "network_hash": run.net.hash,
            "seed": run.seed,
            "network": run.net.model_dump(mode="json"),
        },
    )
    manifest: dict[str, Any] = {
        "run_hash": run.run_hash,
        "dataset_hash": run.dataset_hash,
        "network_hash": run.net.hash,
        "dataset_id": run.dataset_id,
        "network_id": run.network_id,
        "seed": run.seed,
        "started_at": watch.started_at,
        "environment": environment_info(),
    }
    events.emit("train_start", network=run.net.model_dump(mode="json"))

    try:
        dataset = load_dataset(dataset_info["dir"])
        if dataset.spec.hash != run.dataset_hash:
            raise RuntimeError(
                f"dataset hash mismatch: expected {run.dataset_hash}, found {dataset.spec.hash}"
            )
        result = train_model(
            dataset, run.net, report_intermediate=lambda m: events.emit("checkpoint", **m)
        )
        assert result.model is not None
        write_jsonl(run_dir / "history.jsonl", result.history)
        torch.save(
            {"state_dict": result.model.state_dict(), "network": run.net.model_dump(mode="json")},
            run_dir / "model.pt",
        )
        write_json(
            run_dir / "train_summary.json",
            {
                "baseline": result.baseline,
                "final": result.final,
                "stop_reason": result.stop_reason,
                "tokens_seen": result.tokens_seen,
                "n_params": result.n_params,
                "wall_clock_seconds": result.wall_clock_seconds,
            },
        )
        manifest.update(
            status="ok",
            stop_reason=result.stop_reason,
            tokens_seen=result.tokens_seen,
            n_params=result.n_params,
        )
        events.emit("train_done", status="ok", tokens_seen=result.tokens_seen, final=result.final)
    except TrainingDivergedError as exc:
        manifest.update(status="diverged", error=str(exc))
        events.emit("train_failed", level="WARNING", status="diverged", error=str(exc))
    except Exception as exc:  # noqa: BLE001 - recorded, not swallowed silently
        manifest.update(status="error", error=repr(exc), traceback=traceback.format_exc())
        events.emit("train_failed", level="ERROR", status="error", error=repr(exc))

    manifest.update(finished_at=now_iso(), wall_clock_seconds=watch.seconds)
    write_json(manifest_path, manifest)
    return manifest


# --------------------------------------------------------------------------
# node 3: evaluate
# --------------------------------------------------------------------------
def node_evaluate(
    exp: ResolvedExperiment,
    run: RunSpec,
    train_manifest: dict,
    dataset_info: dict,
    force: bool = False,
) -> dict | None:
    """Re-loads the saved model and computes the final metrics from scratch
    (so a model can be re-evaluated without retraining), plus learning-speed
    measures from the training history."""
    layout = Layout.of(exp)
    run_dir = layout.training_dir(run.run_hash)
    metrics_path = run_dir / "metrics.json"
    events = _run_events(exp, run, "evaluate")

    if train_manifest.get("status") != "ok":
        events.emit(
            "evaluate_skipped",
            level="WARNING",
            status=train_manifest.get("status"),
            reason="training did not finish ok",
        )
        return None
    if metrics_path.exists() and not force:
        events.emit("evaluate_reused", reason="content hash exists")
        return read_json(metrics_path)

    dataset = load_dataset(dataset_info["dir"])
    saved = torch.load(run_dir / "model.pt", weights_only=True)
    model = build_model(
        run.net,
        input_dim=dataset.input_dim,
        n_positions=dataset.spec.D,
        n_classes=dataset.n_classes,
        generator=torch.Generator().manual_seed(0),
    )
    model.load_state_dict(saved["state_dict"])

    history = read_jsonl(run_dir / "history.jsonl")
    summary = read_json(run_dir / "train_summary.json")
    final = compute_metrics(model, dataset)
    drift = abs(final["loss"] - history[-1]["loss"])
    if drift > 1e-4:
        events.emit(
            "evaluate_mismatch",
            level="WARNING",
            loss_eval=final["loss"],
            loss_last_checkpoint=history[-1]["loss"],
        )

    metrics = {
        **final,
        "active_pct": history[-1]["active_pct"],
        "tokens_seen": summary["tokens_seen"],
        "n_params": summary["n_params"],
        "wall_clock_seconds": summary["wall_clock_seconds"],
        "stop_reason": summary["stop_reason"],
        **summary["baseline"],
        **tokens_to_accuracy(history, exp.evaluation.accuracy_thresholds),
    }
    np.savez_compressed(run_dir / "predictions.npz", **predictions(model, dataset))
    write_json(metrics_path, {"run_hash": run.run_hash, "metrics": metrics})

    mlflow_run_id = mirror_to_mlflow(
        tracking=exp.tracking,
        output_dir=exp.output_dir,
        experiment=exp.name,
        run_name=f"{run.dataset_id}|{run.network_id}|seed{run.seed}",
        params={
            **flatten("dataset", dataset.spec),
            **flatten("network", run.net),
            "dataset_id": run.dataset_id,
            "network_id": run.network_id,
            "seed": run.seed,
        },
        tags={
            "experiment": exp.name,
            "run_hash": run.run_hash,
            "dataset_hash": run.dataset_hash,
            "network_hash": run.net.hash,
            "config_hash": exp.config_hash,
            "git_sha": str(read_json(run_dir / "manifest.json")["environment"]["git_sha"]),
        },
        history=history,
        metrics={k: v for k, v in metrics.items() if isinstance(v, (int, float))},
        artifacts=[run_dir / "manifest.json", metrics_path, run_dir / "spec.json"],
    )
    if mlflow_run_id:
        manifest = read_json(run_dir / "manifest.json")
        manifest["mlflow_run_id"] = mlflow_run_id
        write_json(run_dir / "manifest.json", manifest)
    events.emit("evaluate_done", metrics=metrics)
    return {"run_hash": run.run_hash, "metrics": metrics}


# --------------------------------------------------------------------------
# node 4: aggregate
# --------------------------------------------------------------------------
def collect_runs(exp: ResolvedExperiment) -> pd.DataFrame:
    """One row per planned run: ids, hashes, status, flattened specs, metrics."""
    layout = Layout.of(exp)
    rows = []
    for run in exp.runs:
        run_dir = layout.training_dir(run.run_hash)
        row: dict[str, Any] = {
            "dataset_id": run.dataset_id,
            "network_id": run.network_id,
            "seed": run.seed,
            "run_hash": run.run_hash,
            "dataset_hash": run.dataset_hash,
            "network_hash": run.net.hash,
            "status": "missing",
        }
        manifest_path = run_dir / "manifest.json"
        if manifest_path.exists():
            manifest = read_json(manifest_path)
            row["status"] = manifest.get("status", "unknown")
            row["error"] = manifest.get("error")
        row.update(flatten("dataset", exp.datasets[run.dataset_id].spec))
        row.update(flatten("network", run.net))
        metrics_path = run_dir / "metrics.json"
        if row["status"] == "ok" and metrics_path.exists():
            row.update(read_json(metrics_path)["metrics"])
        rows.append(row)
    df = pd.DataFrame(rows)
    protected = {"dataset_id", "network_id", "run_hash", "dataset_hash", "network_hash", "status"}
    for col in df.columns:
        if col in protected or col.startswith(("dataset.", "network.")):
            continue
        with contextlib.suppress(ValueError, TypeError):  # NaN is stored as null; restore dtype
            df[col] = pd.to_numeric(df[col])
    return df


def node_aggregate(exp: ResolvedExperiment) -> tuple[pd.DataFrame, pd.DataFrame]:
    layout = Layout.of(exp)
    runs = collect_runs(exp)
    runs.to_csv(layout.exp_dir / "runs.csv", index=False)

    ok = runs[runs["status"] == "ok"]
    metric_cols = [
        c
        for c in ok.columns
        if not c.startswith(("dataset.", "network."))
        and c not in {"seed", "run_hash", "dataset_hash", "network_hash"}
        and pd.api.types.is_numeric_dtype(ok[c])
    ]
    keys = ["dataset_id", "network_id"]
    if ok.empty:
        aggregate = pd.DataFrame(columns=keys)
    else:
        aggregate = ok.groupby(keys)[metric_cols].agg(["mean", "std"])
        aggregate.columns = [f"{m}_{stat}" for m, stat in aggregate.columns]
        aggregate = aggregate.reset_index()
        aggregate["n_ok"] = ok.groupby(keys).size().to_numpy()
    totals = runs.groupby(keys).size().rename("n_runs").reset_index()
    aggregate = totals.merge(aggregate, on=keys, how="left")
    aggregate["n_ok"] = aggregate.get("n_ok", pd.Series(dtype=float)).fillna(0).astype(int)
    aggregate.to_csv(layout.exp_dir / "aggregate.csv", index=False)

    counts = runs["status"].value_counts().to_dict()
    experiment_events(exp, stage="aggregate").emit(
        "aggregate_done", status_counts=counts, n_groups=len(aggregate)
    )
    return runs, aggregate
