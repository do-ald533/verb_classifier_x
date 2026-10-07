"""Observability helpers: structured JSONL events, run manifests, and an
optional MLflow mirror.

The source of truth is the run directory on disk (events.jsonl, manifest.json,
history.jsonl, metrics.json). MLflow is a convenience mirror: a failure to
reach it is logged and never fails a run.
"""

from __future__ import annotations

import logging
import os
import platform
import subprocess
import time
from functools import cache
from collections.abc import Iterable, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import orjson

from src.logging_config import get_logger

logger = get_logger(__name__)
REPO_ROOT = Path(__file__).resolve().parent.parent


def now_iso() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


_OPTIONS = orjson.OPT_SORT_KEYS | orjson.OPT_NON_STR_KEYS | orjson.OPT_SERIALIZE_NUMPY


def _default(value: Any) -> Any:
    if hasattr(value, "item") and callable(value.item):  # torch scalars
        return value.item()
    return str(value)  # paths and anything else odd: readable, never a crash


def dumps(obj: Any, *, indent: bool = False) -> bytes:
    """orjson: ~5-10x faster than json, and NaN/inf become null."""
    option = _OPTIONS | (orjson.OPT_INDENT_2 if indent else 0)
    return orjson.dumps(obj, default=_default, option=option)


def write_json(path: Path, obj: Any) -> None:
    """Atomic (temp file + rename) so readers never see a partial file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_bytes(dumps(obj, indent=True))
    os.replace(tmp, path)


def read_json(path: Path) -> Any:
    return orjson.loads(path.read_bytes())


def write_jsonl(path: Path, rows: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_bytes(b"".join(dumps(r) + b"\n" for r in rows))
    os.replace(tmp, path)


def read_jsonl(path: Path) -> list[dict]:
    return [orjson.loads(line) for line in path.read_bytes().splitlines() if line.strip()]


class EventLog:
    """Append-only JSONL event stream. Every event carries the bound context
    (experiment, dataset, network, seed, stage, hashes...) so any line is
    traceable on its own. Writes to several sinks at once (node-level file +
    experiment-level file)."""

    def __init__(self, paths: Sequence[Path], **context: Any):
        self.paths = [Path(p) for p in paths]
        self.context = context
        for p in self.paths:
            p.parent.mkdir(parents=True, exist_ok=True)

    def emit(self, event: str, level: str = "INFO", **fields: Any) -> None:
        record = {"ts": now_iso(), "level": level, "event": event, **self.context, **fields}
        line = dumps(record) + b"\n"
        for p in self.paths:
            with open(p, "ab") as handle:
                handle.write(line)  # one write call per line: safe for concurrent appenders
        if event != "checkpoint":  # checkpoints are far too chatty for the console
            logger.log(
                getattr(logging, level, logging.INFO),
                "%s %s",
                event,
                {k: v for k, v in {**self.context, **fields}.items() if k in _CONSOLE_KEYS},
            )


_CONSOLE_KEYS = {"stage", "dataset_id", "network_id", "seed", "status", "reason", "path", "error"}


def _git(*args: str) -> str | None:
    try:
        out = subprocess.run(
            ["git", *args], cwd=REPO_ROOT, capture_output=True, text=True, timeout=10, check=True
        )
        return out.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None


@cache
def environment_info() -> dict[str, Any]:
    import numpy
    import torch

    status = _git("status", "--porcelain")
    return {
        "git_sha": _git("rev-parse", "HEAD"),
        "git_branch": _git("rev-parse", "--abbrev-ref", "HEAD"),
        "git_dirty": bool(status) if status is not None else None,
        "python": platform.python_version(),
        "torch": torch.__version__,
        "numpy": numpy.__version__,
        "platform": platform.platform(),
        "cpu_count": os.cpu_count(),
        "torch_threads": torch.get_num_threads(),
        "cuda_available": torch.cuda.is_available(),
    }


class Stopwatch:
    def __init__(self) -> None:
        self.started_at = now_iso()
        self._t0 = time.monotonic()

    @property
    def seconds(self) -> float:
        return time.monotonic() - self._t0


def mirror_to_mlflow(
    *,
    tracking: Any,
    output_dir: str | Path,
    experiment: str,
    run_name: str,
    params: dict[str, Any],
    tags: dict[str, str],
    history: list[dict],
    metrics: dict[str, float],
    artifacts: Sequence[Path],
) -> str | None:
    """Mirrors one finished training run into MLflow. Returns the MLflow run
    id, or None when disabled/unavailable."""
    if not tracking.mlflow:
        return None
    try:
        import mlflow

        root = Path(output_dir).resolve()
        # the file-store backend is deprecated in current MLflow: default to a
        # SQLite db (and artifact dir) next to the results
        mlflow.set_tracking_uri(tracking.uri or f"sqlite:///{root / 'mlflow.db'}")
        name = tracking.experiment or experiment
        if mlflow.get_experiment_by_name(name) is None:
            mlflow.create_experiment(name, artifact_location=(root / "mlflow_artifacts").as_uri())
        mlflow.set_experiment(name)
        with mlflow.start_run(run_name=run_name) as run:
            mlflow.set_tags(tags)
            mlflow.log_params({k: ("" if v is None else v) for k, v in params.items()})
            for row in history:
                step = int(row["tokens_seen"])
                mlflow.log_metrics(
                    {k: v for k, v in row.items() if k != "tokens_seen" and v == v}, step=step
                )
            mlflow.log_metrics({f"final_{k}": v for k, v in metrics.items() if v == v})
            for path in artifacts:
                if path.exists():
                    mlflow.log_artifact(str(path))
            return run.info.run_id
    except Exception as exc:  # noqa: BLE001 - observability must never fail a run
        logger.warning("MLflow mirroring failed for %s: %s", run_name, exc)
        return None
