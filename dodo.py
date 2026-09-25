from pathlib import Path

from src.config import generate_condition_grid
from src.confirm import run_confirmation
from src.sweep import run_condition_study

DOIT_CONFIG = {"default_tasks": ["check"]}


def _mark_done(marker: Path) -> None:
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.touch()


def task_lint():
    return {"actions": ["ruff check src tests"]}


def task_format():
    return {"actions": ["ruff format src tests"]}


def task_typecheck():
    return {"actions": ["ty check src tests"]}


def task_test():
    return {"actions": ["pytest"]}


def task_check():
    """Fast gate: lint + types + unit tests. Everything else depends on this."""
    return {"actions": None, "task_dep": ["lint", "typecheck", "test"]}


def task_sweep():
    """One DAG node per linguistic condition, generated from the same grid
    the whole pipeline uses (plan §3/§4). run_condition_study raises if its
    Optuna study turns out degenerate (too many failed trials), so doit
    never marks a broken condition as done (plan §1.1 #6)."""
    for condition in generate_condition_grid():
        marker = Path(f"reports/.done/sweep_{condition.id}.done")
        yield {
            "name": condition.id,
            "actions": [(run_condition_study, [condition]), (_mark_done, [marker])],
            "targets": [str(marker)],
            "uptodate": [marker.exists],
            "task_dep": ["check"],
        }


def task_confirm():
    """One DAG node per condition: re-run the Optuna-selected champion
    config(s) on a fresh, larger seed batch (plan §1.1 #4)."""
    for condition in generate_condition_grid():
        sweep_marker = Path(f"reports/.done/sweep_{condition.id}.done")
        confirm_marker = Path(f"reports/.done/confirm_{condition.id}.done")
        yield {
            "name": condition.id,
            "actions": [(run_confirmation, [condition]), (_mark_done, [confirm_marker])],
            "file_dep": [str(sweep_marker)],
            "targets": [str(confirm_marker)],
            "uptodate": [confirm_marker.exists],
        }


def task_report():
    return {
        "actions": [
            "python -m src.report --experiment verb_regularization_sweep --out reports/"
        ],
        "task_dep": ["confirm"],
    }


def task_ui():
    return {"actions": ["mlflow ui --backend-store-uri ./mlruns"], "uptodate": [False]}
