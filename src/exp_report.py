"""Experiment report: aggregate -> tables + figures + report.md.

Everything shown is traceable: each table row carries dataset/network ids
and hashes, and report.md records the config hash, git sha and the command
that reproduces the experiment."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from src.config_loader import ResolvedExperiment
from src.data import dataset_dir
from src.nodes import Layout, experiment_events, node_aggregate
from src.obs import environment_info, now_iso, read_json, read_jsonl

CURVE_METRICS = ["loss", "acc_types", "irregular_acc", "wug_regular_prob"]
RESULT_COLUMNS = [
    ("wug_regular_prob", "wug regular prob"),
    ("acc_types", "acc (types)"),
    ("irregular_acc", "acc (irregular)"),
    ("overregularization_rate", "overregularization"),
    ("tokens_seen", "tokens seen"),
    ("tokens_to_acc_90", "tokens to 90% acc"),
]


def _md(df: pd.DataFrame) -> str:
    if df.empty:
        return "_none_"
    try:
        return df.to_markdown(index=False)
    except ImportError:
        return df.to_string(index=False)


def _fmt(mean: float, std: float, digits: int = 3) -> str:
    if pd.isna(mean):
        return "n/a"
    return f"{mean:.{digits}f} ± {0.0 if pd.isna(std) else std:.{digits}f}"


def _savefig(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def _dataset_table(exp: ResolvedExperiment, layout: Layout) -> pd.DataFrame:
    rows = []
    for ds in exp.datasets.values():
        directory = Path(ds.path) if ds.path else dataset_dir(layout.root, ds.spec)
        stats_path = directory / "stats.json"
        stats = read_json(stats_path) if stats_path.exists() else {}
        s = ds.spec
        rows.append(
            {
                "id": ds.id,
                "hash": ds.hash,
                "D": s.D,
                "|alphabet|": len(s.alphabet),
                "lexicon": s.lexicon_size,
                "irr types": f"{s.p_irregular_types:.0%}",
                "irr tokens": f"{s.p_token_irregular:.0%}",
                "freq": s.freq_distribution,
                "classes": s.n_classes,
                "placement": s.irregular_placement,
                "wugs": f"{s.n_wug} ({s.wug_strategy})",
                "sampling": s.sampling,
                "token entropy": round(stats.get("token_entropy_nats", float("nan")), 3),
                "density": f"{s.lexicon_size / s.space_size:.2%}",
            }
        )
    return pd.DataFrame(rows)


def _network_table(exp: ResolvedExperiment) -> pd.DataFrame:
    rows = []
    for net in exp.networks.values():
        s = net.spec
        rows.append(
            {
                "id": net.id,
                "hash": s.hash,
                "arch": s.arch,
                "shape": (
                    f"{list(s.hidden)}"
                    if s.arch == "mlp"
                    else f"d={s.d_model} h={s.n_heads} L={s.n_layers}"
                ),
                "act": s.activation,
                "optimizer": f"{s.optimizer} lr={s.lr:g}",
                "batch": s.batch_size,
                "dropout": s.dropout,
                "norm": s.norm,
                "pruning": s.pruning_enabled,
            }
        )
    return pd.DataFrame(rows)


def _results_table(aggregate: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, r in aggregate.iterrows():
        row = {
            "dataset": r["dataset_id"],
            "network": r["network_id"],
            "ok/runs": f"{int(r['n_ok'])}/{int(r['n_runs'])}",
        }
        for col, label in RESULT_COLUMNS:
            row[label] = (
                _fmt(
                    r.get(f"{col}_mean", float("nan")),
                    r.get(f"{col}_std", float("nan")),
                    digits=0 if col.startswith("tokens") else 3,
                )
                if f"{col}_mean" in aggregate.columns
                else "n/a"
            )
        rows.append(row)
    return pd.DataFrame(rows)


def _plot_learning_curves(exp: ResolvedExperiment, runs: pd.DataFrame, layout: Layout, out: Path):
    paths = []
    ok = runs[runs["status"] == "ok"]
    for ds_id, ds_runs in ok.groupby("dataset_id"):
        frames = []
        for _, run in ds_runs.iterrows():
            history_path = layout.training_dir(run["run_hash"]) / "history.jsonl"
            if history_path.exists():
                h = pd.DataFrame(read_jsonl(history_path))
                h["network_id"], h["seed"] = run["network_id"], run["seed"]
                frames.append(h)
        if not frames:
            continue
        data = pd.concat(frames, ignore_index=True)
        fig, axes = plt.subplots(2, 2, figsize=(11, 7), sharex=True)
        for ax, metric in zip(axes.flat, CURVE_METRICS, strict=True):
            sns.lineplot(
                data=data, x="tokens_seen", y=metric, hue="network_id", errorbar="sd", ax=ax
            )
            ax.set_title(metric)
            ax.set_xlabel("tokens seen")
            if ax is not axes.flat[0]:
                legend = ax.get_legend()
                if legend:
                    legend.remove()
        fig.suptitle(f"Learning curves — dataset {ds_id} (mean ± sd over seeds)")
        fig.tight_layout()
        path = out / f"learning_curves_{ds_id}.png"
        _savefig(fig, path)
        paths.append(path)
    return paths


def _plot_final(runs: pd.DataFrame, out: Path) -> Path | None:
    ok = runs[runs["status"] == "ok"]
    if ok.empty or "wug_regular_prob" not in ok.columns:
        return None
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for ax, metric in zip(axes, ["wug_regular_prob", "irregular_acc"], strict=True):
        sns.barplot(data=ok, x="dataset_id", y=metric, hue="network_id", errorbar="sd", ax=ax)
        ax.set_title(f"final {metric}")
        ax.tick_params(axis="x", rotation=30)
        if ax is axes[1]:
            ax.get_legend().remove() if ax.get_legend() else None
    fig.tight_layout()
    path = out / "final_metrics.png"
    _savefig(fig, path)
    return path


def build_experiment_report(exp: ResolvedExperiment) -> Path:
    layout = Layout.of(exp)
    runs, aggregate = node_aggregate(exp)
    out = layout.exp_dir / "report"
    figures = out / "figures"
    curves = _plot_learning_curves(exp, runs, layout, figures)
    final_fig = _plot_final(runs, figures)

    env = environment_info()
    failures = runs[runs["status"] != "ok"][
        [
            "dataset_id",
            "network_id",
            "seed",
            "run_hash",
            "status",
            *(["error"] if "error" in runs else []),
        ]
    ]
    lines = [
        f"# Experiment `{exp.name}`",
        "",
        exp.description or "_no description_",
        "",
        f"- generated: {now_iso()}",
        f"- config hash: `{exp.config_hash}`",
        f"- git: `{env['git_sha']}` (branch `{env['git_branch']}`, dirty={env['git_dirty']})",
        f"- runs: {len(runs)} planned, {(runs['status'] == 'ok').sum()} ok, "
        f"{(runs['status'] != 'ok').sum()} not ok",
        f"- reproduce: `python -m src run {exp.source_path or '<config.json>'}`",
        "",
        "## Datasets",
        "",
        _md(_dataset_table(exp, layout)),
        "",
        "## Networks",
        "",
        _md(_network_table(exp)),
        "",
        "## Results (mean ± sd over seeds)",
        "",
        _md(_results_table(aggregate)),
        "",
        "## Not-ok runs",
        "",
        _md(failures),
        "",
        "## Figures",
        "",
    ]
    lines += [
        f"![{p.stem}](figures/{p.name})" for p in [*curves, *([final_fig] if final_fig else [])]
    ]
    lines += [
        "",
        "## Files",
        "",
        "- `../runs.csv` one row per run (specs, status, metrics)",
        "- `../aggregate.csv` mean/std per (dataset, network)",
        "- `../trainings/<run_hash>/` manifest, events, history, model, metrics, predictions",
        "- `../events.jsonl` experiment-level event stream",
    ]
    report_path = out / "report.md"
    out.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines) + "\n")
    experiment_events(exp, stage="report").emit("report_done", path=str(report_path))
    return report_path
