"""Experiment JSON -> validated, fully-resolved experiment.

An experiment file declares datasets, networks and which of them to cross
(`runs`). Resolution applies, in increasing priority: `defaults` < `extends`
chain < the entry's own `spec` < `matrix` combination. Everything is
validated up front (pydantic, extra="forbid") so a typo or an infeasible
combination fails before any compute is spent.
"""

from __future__ import annotations

import itertools
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from src.specs import DatasetSpec, NetworkSpec, hash_payload

DEFAULT_OUTPUT_DIR = "runs"
NAME_RE = re.compile(r"^[A-Za-z0-9_.-]+$")


class ConfigError(ValueError):
    pass


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class TrackingConfig(_Strict):
    mlflow: bool = True
    uri: str | None = None
    experiment: str | None = None


class EvaluationConfig(_Strict):
    accuracy_thresholds: list[float] = [0.5, 0.9, 0.99]


class Defaults(_Strict):
    dataset: dict[str, Any] = {}
    network: dict[str, Any] = {}


class Entry(_Strict):
    id: str
    extends: str | None = None
    spec: dict[str, Any] = {}
    matrix: dict[str, list[Any]] = {}


class DatasetEntry(Entry):
    path: str | None = None  # use an already-generated dataset directory


class NetworkEntry(Entry):
    pass


class RunEntry(_Strict):
    datasets: list[str] | Literal["*"] = "*"
    networks: list[str] | Literal["*"] = "*"
    seeds: list[int] = Field(min_length=1)


class ExperimentFile(_Strict):
    experiment: str
    description: str = ""
    output_dir: str = DEFAULT_OUTPUT_DIR
    tracking: TrackingConfig = TrackingConfig()
    evaluation: EvaluationConfig = EvaluationConfig()
    defaults: Defaults = Defaults()
    datasets: list[DatasetEntry] = Field(min_length=1)
    networks: list[NetworkEntry] = Field(min_length=1)
    runs: list[RunEntry] = Field(min_length=1)


# --------------------------------------------------------------------------
# resolved form
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class ResolvedDataset:
    id: str
    group: str
    spec: DatasetSpec
    path: str | None = None  # pre-existing dataset directory

    @property
    def hash(self) -> str:
        return self.spec.hash


@dataclass(frozen=True)
class ResolvedNetwork:
    id: str
    group: str
    spec: NetworkSpec


@dataclass(frozen=True)
class RunSpec:
    dataset_id: str
    network_id: str
    seed: int
    dataset_hash: str
    net: NetworkSpec  # seeds resolved

    @property
    def run_hash(self) -> str:
        return hash_payload({"dataset": self.dataset_hash, "network": self.net.hash})

    @property
    def label(self) -> str:
        return f"{self.dataset_id}|{self.network_id}|seed={self.seed}"


@dataclass
class ResolvedExperiment:
    name: str
    description: str
    output_dir: str
    tracking: TrackingConfig
    evaluation: EvaluationConfig
    datasets: dict[str, ResolvedDataset]
    networks: dict[str, ResolvedNetwork]
    runs: list[RunSpec]
    source_path: str | None = None
    raw: dict = field(default_factory=dict)

    @property
    def config_hash(self) -> str:
        return hash_payload(
            {
                "datasets": {k: v.hash for k, v in sorted(self.datasets.items())},
                "networks": {k: v.spec.hash for k, v in sorted(self.networks.items())},
                "runs": sorted(r.run_hash for r in self.runs),
            }
        )

    def summary(self) -> dict:
        return {
            "experiment": self.name,
            "description": self.description,
            "config_hash": self.config_hash,
            "datasets": {
                k: {"hash": v.hash, "path": v.path, "spec": v.spec.model_dump(mode="json")}
                for k, v in self.datasets.items()
            },
            "networks": {
                k: {"hash": v.spec.hash, "spec": v.spec.model_dump(mode="json")}
                for k, v in self.networks.items()
            },
            "runs": [
                {
                    "dataset": r.dataset_id,
                    "network": r.network_id,
                    "seed": r.seed,
                    "run_hash": r.run_hash,
                    "dataset_hash": r.dataset_hash,
                }
                for r in self.runs
            ],
        }


# --------------------------------------------------------------------------
# resolution
# --------------------------------------------------------------------------
def _matrix_suffix(combo: dict[str, Any]) -> str:
    return ",".join(f"{k}={json.dumps(v, separators=(',', ':'))}" for k, v in combo.items())


def _specs_for_entries(
    kind: str, entries: list[Any], defaults: dict[str, Any]
) -> list[tuple[str, str, dict[str, Any]]]:
    """-> [(expanded_id, group_id, merged_spec_dict)]"""
    by_id = {}
    for entry in entries:
        if not NAME_RE.match(entry.id):
            raise ConfigError(f"{kind} id {entry.id!r} must match {NAME_RE.pattern}")
        if entry.id in by_id:
            raise ConfigError(f"duplicate {kind} id {entry.id!r}")
        by_id[entry.id] = entry

    cache: dict[str, dict[str, Any]] = {}

    def base_spec(entry_id: str, stack: tuple[str, ...] = ()) -> dict[str, Any]:
        if entry_id in stack:
            raise ConfigError(f"{kind} 'extends' cycle: {' -> '.join((*stack, entry_id))}")
        if entry_id not in by_id:
            raise ConfigError(f"{kind} {stack[-1]!r} extends unknown id {entry_id!r}")
        if entry_id not in cache:
            entry = by_id[entry_id]
            parent = (
                base_spec(entry.extends, (*stack, entry_id)) if entry.extends else dict(defaults)
            )
            cache[entry_id] = {**parent, **entry.spec}
        return cache[entry_id]

    out = []
    for entry in entries:
        if entry.extends and by_id.get(entry.extends) is not None and by_id[entry.extends].matrix:
            raise ConfigError(f"{kind} {entry.id!r} cannot extend matrix entry {entry.extends!r}")
        spec = base_spec(entry.id)
        if not entry.matrix:
            out.append((entry.id, entry.id, spec))
            continue
        keys = list(entry.matrix)
        for values in itertools.product(*(entry.matrix[k] for k in keys)):
            combo = dict(zip(keys, values, strict=True))
            out.append((f"{entry.id}.{_matrix_suffix(combo)}", entry.id, {**spec, **combo}))
    return out


def _build(model_cls: type, kind: str, entry_id: str, payload: dict[str, Any]):
    try:
        return model_cls(**payload)
    except ValidationError as exc:
        details = "; ".join(
            f"{'.'.join(str(p) for p in e['loc']) or '<spec>'}: {e['msg']}" for e in exc.errors()
        )
        raise ConfigError(f"{kind} {entry_id!r}: {details}") from None


def _select(kind: str, selectors: list[str] | str, items: dict[str, Any]) -> list[str]:
    if selectors == "*":
        return list(items)
    chosen: list[str] = []
    for sel in selectors:
        matches = [k for k, v in items.items() if k == sel or v.group == sel]
        if not matches:
            raise ConfigError(f"runs reference unknown {kind} {sel!r}")
        chosen.extend(m for m in matches if m not in chosen)
    return chosen


def resolve_experiment(
    data: dict[str, Any], *, output_dir: str | None = None, source_path: str | None = None
) -> ResolvedExperiment:
    try:
        file = ExperimentFile.model_validate(data)
    except ValidationError as exc:
        details = "; ".join(
            f"{'.'.join(str(p) for p in e['loc'])}: {e['msg']}" for e in exc.errors()
        )
        raise ConfigError(f"invalid experiment file: {details}") from None
    if not NAME_RE.match(file.experiment):
        raise ConfigError(f"experiment name must match {NAME_RE.pattern}")

    datasets: dict[str, ResolvedDataset] = {}
    path_entries = {e.id: e for e in file.datasets if e.path}
    inline_entries = [e for e in file.datasets if not e.path]
    for entry in path_entries.values():
        if entry.spec or entry.matrix or entry.extends:
            raise ConfigError(f"dataset {entry.id!r}: 'path' cannot be combined with spec/matrix")
        spec_file = Path(entry.path or "") / "spec.json"
        if not spec_file.exists():
            raise ConfigError(f"dataset {entry.id!r}: {spec_file} does not exist")
        spec = _build(DatasetSpec, "dataset", entry.id, json.loads(spec_file.read_text()))
        datasets[entry.id] = ResolvedDataset(entry.id, entry.id, spec, path=entry.path)
    for ds_id, group, merged in _specs_for_entries(
        "dataset", inline_entries, file.defaults.dataset
    ):
        if ds_id in datasets:
            raise ConfigError(f"duplicate dataset id {ds_id!r}")
        datasets[ds_id] = ResolvedDataset(
            ds_id, group, _build(DatasetSpec, "dataset", ds_id, merged)
        )

    networks: dict[str, ResolvedNetwork] = {}
    for net_id, group, merged in _specs_for_entries(
        "network", file.networks, file.defaults.network
    ):
        networks[net_id] = ResolvedNetwork(
            net_id, group, _build(NetworkSpec, "network", net_id, merged)
        )

    runs: list[RunSpec] = []
    seen: set[tuple[str, str, int]] = set()
    for run_entry in file.runs:
        for ds_id in _select("dataset", run_entry.datasets, datasets):
            for net_id in _select("network", run_entry.networks, networks):
                for seed in run_entry.seeds:
                    if (ds_id, net_id, seed) in seen:
                        continue
                    seen.add((ds_id, net_id, seed))
                    runs.append(
                        RunSpec(
                            dataset_id=ds_id,
                            network_id=net_id,
                            seed=seed,
                            dataset_hash=datasets[ds_id].hash,
                            net=networks[net_id].spec.with_seed(seed),
                        )
                    )
    # identical (dataset, network, seeds) under different ids are the same computation
    hashes = [r.run_hash for r in runs]
    if len(set(hashes)) != len(hashes):
        raise ConfigError("two runs resolve to the same content hash (duplicate configurations)")

    return ResolvedExperiment(
        name=file.experiment,
        description=file.description,
        output_dir=output_dir or file.output_dir,
        tracking=file.tracking,
        evaluation=file.evaluation,
        datasets=datasets,
        networks=networks,
        runs=runs,
        source_path=source_path,
        raw=data,
    )


def load_experiment(path: str | Path, *, output_dir: str | None = None) -> ResolvedExperiment:
    path = Path(path)
    try:
        data = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ConfigError(f"cannot read {path}: {exc}") from None
    return resolve_experiment(data, output_dir=output_dir, source_path=str(path))
