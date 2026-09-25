#!/usr/bin/env bash
# Quality gate: lint, format check, type check, tests. Run before any sweep.
# These three checks are independent of each other and of the experiment
# pipeline, so they do not need a DAG runner. The pipeline's own DAG
# (parallel Optuna sweeps, then confirmation, then report) lives in
# src/pipeline.py and is run separately with Prefect.
set -euo pipefail

uv run ruff check src tests
uv run ruff format --check src tests
uv run ty check src tests
uv run pytest
