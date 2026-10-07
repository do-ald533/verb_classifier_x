"""python -m src {validate|run|report} experiment.json"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from src.config_loader import ConfigError, ResolvedExperiment, load_experiment
from src.data import dataset_dir, is_complete
from src.exp_report import build_experiment_report
from src.nodes import Layout, collect_runs, prepare_experiment
from src.obs import read_json


def plan_text(exp: ResolvedExperiment) -> str:
    layout = Layout.of(exp)
    lines = [
        f"experiment {exp.name}  config_hash={exp.config_hash}",
        f"output: {layout.exp_dir}",
        f"{len(exp.datasets)} datasets, {len(exp.networks)} networks, {len(exp.runs)} runs",
        "",
        "datasets:",
    ]
    for ds in exp.datasets.values():
        directory = Path(ds.path) if ds.path else dataset_dir(layout.root, ds.spec)
        state = "cached" if is_complete(directory) else "to build"
        lines.append(f"  {ds.id:40s} {ds.hash}  {state}")
    lines += ["", "networks:"]
    lines += [f"  {n.id:40s} {n.spec.hash}" for n in exp.networks.values()]
    states: dict[str, int] = {}
    for run in exp.runs:
        manifest = layout.training_dir(run.run_hash) / "manifest.json"
        status = read_json(manifest).get("status", "unknown") if manifest.exists() else "pending"
        states[status] = states.get(status, 0) + 1
    lines += ["", "runs by state: " + ", ".join(f"{k}={v}" for k, v in sorted(states.items()))]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m src", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for name, help_ in [
        ("validate", "check the config and show the plan (nothing is written)"),
        ("run", "run the DAG: datasets -> train -> evaluate -> aggregate -> report"),
        ("report", "re-aggregate and rebuild the report from existing results"),
    ]:
        p = sub.add_parser(name, help=help_)
        p.add_argument("config")
        p.add_argument("--output-dir", help="overrides output_dir in the config (default: runs)")
        if name == "run":
            p.add_argument("--engine", choices=["prefect", "local"], default="prefect")
            p.add_argument("--max-workers", type=int, default=4)
            p.add_argument("--force", action="store_true", help="ignore cached artifacts")
            p.add_argument("--no-report", action="store_true")
    args = parser.parse_args(argv)

    try:
        exp = load_experiment(args.config, output_dir=args.output_dir)
    except ConfigError as exc:
        print(f"config error: {exc}", file=sys.stderr)
        return 2

    if args.command == "validate":
        print(plan_text(exp))
        return 0
    if args.command == "report":
        prepare_experiment(exp)
        print(build_experiment_report(exp))
        return 0

    from src.flow import run_local, run_prefect

    if args.engine == "local":
        run_local(exp, force=args.force, run_report=not args.no_report)
    else:
        run_prefect(
            exp, force=args.force, run_report=not args.no_report, max_workers=args.max_workers
        )

    runs = collect_runs(exp)
    bad = runs[runs["status"] != "ok"]
    print(f"{len(runs) - len(bad)}/{len(runs)} runs ok; results in {Layout.of(exp).exp_dir}")
    if not bad.empty:
        print(bad[["dataset_id", "network_id", "seed", "status"]].to_string(index=False))
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
