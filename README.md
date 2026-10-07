# Verb Classifier X

An experiment pipeline built around a small PyTorch model of verb past tense
regularization (a Rumelhart and McClelland style connectionist model, tested
with the Marcus et al. wug paradigm). The pipeline sweeps hyperparameters
across 36 linguistic conditions, searches network hyperparameters within each
condition using Optuna, confirms the winning configuration on a larger seed
batch, tracks every run in MLflow, and produces a full statistical and visual
report.

The full design rationale, including two independent expert reviews and the
later switch from doit to Prefect, is documented in
`EXPERIMENT_PIPELINE_PLAN.md`. This README covers how to actually run it.

## Requirements

* Python 3.12 (pinned in `.python-version`; do not use 3.13, torch 2.4.1 has
  no wheel for it)
* uv (https://docs.astral.sh/uv/), used for environment and dependency
  management, and for running all commands below

If uv is not installed, follow the installation instructions on that page
before continuing.

## Installation

From the project root:

```bash
uv sync
```

This creates a `.venv`, installs Python 3.12 automatically if it is not
already available, and installs every dependency pinned in
`pyproject.toml` and `uv.lock`.

Verify the install:

```bash
bash scripts/check.sh
```

This runs linting and format checking (ruff), type checking (ty), and the
unit test suite. It should pass on a fresh checkout with no data or MLflow
runs present yet.

## Project structure

```
verb_classifier_x/
    EXPERIMENT_PIPELINE_PLAN.md   full design document and review history
    pyproject.toml                dependencies, ruff, ty, pytest configuration
    scripts/
        check.sh                  lint, format check, typecheck, tests (see "Testing and code quality")
    src/
        config.py                 hyperparameter constants, the 36 condition grid, pydantic config models
        data.py                   synthetic lexicon and training token generation
        model.py                  the MLP, with deterministic, parallel safe initialization
        pruning.py                magnitude based pruning (ported from the source notebook)
        metrics.py                wug generalization metrics, including the continuous primary metric
        train.py                  the training loop for one configuration and seed
        objective.py               the Optuna objective for one linguistic condition
        sweep.py                  runs one condition's Optuna search phase
        confirm.py                re runs the winning configuration on a larger seed batch
        pipeline.py               the Prefect flow that runs sweep, then confirm, then report, in parallel across conditions
        stats.py                  the four pre registered statistical comparisons
        leaderboard.py            queries MLflow for the current best results
        report.py                 generates every plot, table, and the final report
        logging_config.py         pipeline level logging (separate from MLflow)
    tests/                        unit tests for the modules above
    mlruns/                       MLflow's local tracking store, created on first run
    optuna_journal.log            Optuna's storage file, created on first run
    logs/                         rotating pipeline logs, created on first run
    reports/                      leaderboard, report, and figures, created by the pipeline
```

## How the pipeline works

The hyperparameter space is split into two tiers.

Linguistic factors define what the model is asked to learn: the
dimensionality of the input radical (D), the number of irregular verb types,
the token frequency skew toward irregulars, and whether magnitude pruning is
enabled. These four factors form a full grid of 36 conditions, each treated
as its own scientific comparison, not something to optimize away.

Network factors define how well the model can learn a given condition:
hidden layer width and depth, optimizer choice, learning rate, momentum, and
(when pruning is enabled) the pruning interval and target sparsity. These are
searched with Optuna inside each of the 36 conditions.

Each condition proceeds in two phases:

1. Search phase. Optuna runs 35 trials (60 for pruning enabled conditions,
   since they require a two objective search) against a cheap budget of 5
   seeds per trial, purely to rank candidate network configurations.
2. Confirmation phase. The configuration Optuna selected as best (or, for
   pruning enabled conditions, every point on the accuracy versus sparsity
   Pareto front) is re run on 20 fresh seeds. This confirmation run is the
   only thing any leaderboard or best result comparison reads from. The
   noisy search phase never determines a reported number directly.

Everything is logged to a local MLflow tracking store under
`./mlruns`, organized as: one root run per condition, nested trial runs
during search, nested seed runs under each trial, and separate confirmation
runs once search completes.

## How the pipeline is orchestrated

The experiment pipeline (sweep, confirm, report) is defined as a Prefect
flow in `src/pipeline.py`, not a plain script. Each of the 36 conditions
becomes two chained tasks, `sweep-condition` then `confirm-condition`, and
the report only runs once every condition has finished confirming.

Conditions run in parallel with each other. Passing one task's result into
the next task's input is what tells Prefect there is a dependency between
them, the same way a downstream task in Airflow depends on an upstream one;
Prefect infers the graph from that data flow rather than from a separately
declared edge list. Parallelism is controlled by the task runner attached to
the flow: this project uses `ProcessPoolTaskRunner`, so each condition's
training work runs in its own process, which matches the reason
`JournalStorage` (Optuna's storage backend here) was chosen in the first
place, since it is built for concurrent multi process writes.

No Prefect server, scheduler, or database is required to run any of this.
`python -m src.pipeline` executes the flow directly in the current process.
A server is only needed if the interactive Prefect UI is wanted, which this
project does not depend on.

Resumability does not depend on Prefect at all. Each task checks for a
completion marker file under `reports/.done/` before doing any work, and
writes that marker only after succeeding. Interrupting the pipeline and
running it again re executes only the conditions that had not finished.

## Running the pipeline

Run the quality gate (lint, format check, type check, unit tests):

```bash
bash scripts/check.sh
```

Run the full pipeline (sweep, then confirm, then report) across all 36
conditions, with up to 8 running in parallel:

```bash
uv run python -m src.pipeline --max-workers 8
```

This is the most expensive step. It can be interrupted at any point (Ctrl+C,
a crash, a killed process) and resumed safely, as described above. Running
the same command again will skip conditions that already finished and
resume the rest.

Run only the search and confirmation phases, without building the report:

```bash
uv run python -m src.pipeline --stage sweep-and-confirm --max-workers 8
```

Rebuild only the report from whatever runs already exist, without touching
the sweep or confirmation phases:

```bash
uv run python -m src.pipeline --stage report
```

To iterate on the pipeline code itself without waiting on the full sweep,
run a single condition directly, bypassing Prefect entirely:

```bash
uv run python -c "
from src.config import generate_condition_grid
from src.sweep import run_condition_study
run_condition_study(generate_condition_grid()[0])
"
```

Browse every run interactively at any point, including while the pipeline
is still in progress:

```bash
uv run mlflow ui --backend-store-uri ./mlruns
```

Then open the printed local address in a browser.

Optional: browse a single condition's Optuna study (parameter importance,
optimization history) in its own interactive dashboard:

```bash
uv run optuna-dashboard sqlite:///optuna_journal.log
```

Note that this specific dashboard renders with plotly internally. It is a
separate, optional, interactive tool. Nothing it produces is included in the
generated report, which uses matplotlib and seaborn only.

## Understanding the output

`reports/leaderboard.csv` lists the top confirmation runs ranked by
`wug_regular_prob`, the primary metric (mean softmax probability assigned to
the regular class across all wug items).

`reports/best_config.json` records the single best configuration found, in
a format that can be loaded directly into `LinguisticConfig` and
`NetworkConfig`.

`reports/report.md` is the main entry point for interpreting results. It
separates two kinds of findings:

* Confirmatory: the four hypotheses declared in advance
  (`EXPERIMENT_PIPELINE_PLAN.md`, section 4a), tested with Kruskal Wallis
  and pairwise Mann Whitney U, corrected for multiple comparisons with the
  Benjamini Hochberg procedure.
* Exploratory: every other breakdown of the 36 condition grid, shown in the
  figures but explicitly not corrected for multiple comparisons. Treat these
  as descriptive, not as confirmed effects.

`reports/figures/` contains, among others, an overregularization confusion
matrix (which irregular verbs get systematically misclassified as regular),
the wug response distribution plotted against the untrained model's baseline
response, per condition hyperparameter importance and correlation plots, and
the accuracy versus sparsity Pareto front for pruning enabled conditions.

## Declarative experiments (JSON)

Besides the fixed Optuna grid (`python -m src.pipeline`), experiments can be
described in a JSON file and run as a DAG:

```bash
uv run python -m src validate experiments/example_smoke.json   # check + plan, writes nothing
uv run python -m src run experiments/example_smoke.json        # Prefect, parallel
uv run python -m src run experiments/example_smoke.json --engine local   # sequential, no server
uv run python -m src report experiments/example_smoke.json     # rebuild aggregate + report
```

```
config.json -> validate -> build_dataset -> train -> evaluate -> aggregate -> report
                           (per dataset)   (per dataset x network x seed)
```

Every node reads/writes only the disk, keyed by the **content hash** of its
spec. Re-running skips what exists (`--force` ignores the cache), datasets are
shared across networks and across experiments, and a failed run
(`diverged`/`error`) is recorded in its manifest without stopping the DAG.

### File format

* `defaults.dataset` / `defaults.network`: base values for every entry.
* `datasets[]` / `networks[]`: `id`, optional `extends` (inherit another entry),
  `spec` (overrides) and optional `matrix` (cartesian product; ids become
  `base.hidden=[8]`...). A dataset can also be `{"id": ..., "path": "<dir>"}` to
  reuse a dataset directory that already exists.
* `runs[]`: `datasets`/`networks` (ids, a matrix group id, or `"*"`) and `seeds`.
  The run seed fills `init_seed` and `data_order_seed` unless the network sets them.
* `tracking`: `{"mlflow": true, "uri": null, "experiment": null}`. Default store is a
  SQLite file inside `output_dir`.
* `output_dir` (default `runs`, overridable with `--output-dir`).

Dataset spec (`src/specs.py::DatasetSpec`): `D`, `alphabet`, `encoding`
(`raw`/`one_hot`), `lexicon_size`, `p_irregular_types`, `p_token_irregular`,
`n_wug`, `freq_distribution` (`uniform`/`zipf` + `zipf_exponent`), `class_scheme`
(`unique_per_irregular`/`shared_k`+`k_classes`/`by_cluster`), `irregular_placement`
(`random`/`clustered`+`n_clusters`), `wug_strategy`
(`random`/`near_irregular`/`far_from_lexicon`), `sampling`
(`stream`/`finite`+`train_size`), `seed`. Infeasible combinations
(`|alphabet|**D < lexicon_size + n_wug`, ...) are rejected at validation time.

Network spec (`NetworkSpec`): `arch` (`mlp`/`transformer`), `hidden`, `activation`,
`dropout`, `norm`, `init`, `bias`, transformer `d_model`/`n_heads`/`n_layers`/`ff_dim`
(each radical position is a token), `optimizer`, `lr`, `momentum`, `weight_decay`,
`lr_schedule`, `grad_clip`, `batch_size`, `init_seed`/`data_order_seed`,
`max_tokens`, `eval_every_tokens`, early-stop settings, pruning settings.

### Output layout

```
runs/datasets/<dataset_hash>/        spec.json data.pt stats.json
runs/<experiment>/config.json        your file, verbatim
runs/<experiment>/resolved_config.json   after defaults/extends/matrix, with all hashes
runs/<experiment>/events.jsonl       experiment-level event stream (JSONL)
runs/<experiment>/trainings/<run_hash>/
    spec.json manifest.json (status, git sha, versions, timing) events.jsonl
    history.jsonl model.pt metrics.json predictions.npz
runs/<experiment>/{runs.csv,aggregate.csv,report/report.md}
```

Every event line carries `experiment`, `stage`, `dataset_id`, `network_id`,
`seed`, `run_hash` and `dataset_hash`, so any line can be traced to its
inputs. `VERB_TORCH_THREADS` (default 1) sets torch threads per training task.
Under Prefect's temporary SQLite API server, many workers can log
"database is locked" noise; results are unaffected (use a real Prefect server
to avoid it).

## Changing the hyperparameter space

The 36 condition grid, the fixed constants (lexicon size, number of wugs,
trial budgets, seed lists), and the network search ranges are not spread
across the codebase. They live in two places:

* `src/config.py`: linguistic factors, fixed constants, and
  `generate_condition_grid()`.
* `src/objective.py`: the `_suggest_network_config` function, which defines
  the Optuna search space for network hyperparameters.

Every radical must be unique (otherwise two verbs could share an input but
have different output classes), so each condition needs
`LEXICON_SIZE + N_WUG <= 3 ** D`. This is enforced by a pydantic validator on
`LinguisticConfig`; `generate_condition_grid()` skips infeasible
`(D, LEXICON_SIZE)` pairs (e.g. 10000 verbs need `D >= 9`).

Training samples tokens on the fly rather than cycling a fixed dataset: each
step draws regular vs. irregular with probability `P_TOKEN_IRREGULAR`, then a
verb uniformly within that group (`src.data.sample_token_indices`). Progress
is measured in cumulative `tokens_seen`; `BATCH_SIZE` (default 1) groups
draws into batches. Each irregular verb has its own class, so a condition has
`N_IRREGULAR + 1` output classes.

## Changing the level of parallelism

`--max-workers` on `python -m src.pipeline` controls how many conditions run
at once, each in its own process. There is no benefit to setting this above
the number of CPU cores available, and setting it to 1 runs every condition
sequentially, which is useful for debugging.

If process based parallelism ever causes issues, for example if some object
being passed between tasks turns out not to be picklable, `ProcessPoolTaskRunner`
in `src/pipeline.py`'s `main()` can be swapped for `ThreadPoolTaskRunner`
from the same `prefect.task_runners` module as a one line change. Threads
will not parallelize the CPU bound training work as effectively as separate
processes, but will still parallelize any waiting on disk or MLflow I/O.

## Testing and code quality

```bash
uv run ruff check src tests           # lint
uv run ruff format --check src tests  # format check
uv run ty check src tests             # type check
uv run pytest                         # tests, with coverage
bash scripts/check.sh                 # all four of the above, in order
```

These are independent, flat commands with no dependency graph between them,
so they are not part of the Prefect pipeline in `src/pipeline.py`. That
pipeline is reserved for the actual experiment DAG: sweep, confirm, report.

## Known limitations

Two issues raised during design review were deliberately not resolved in
this version, and are documented in `EXPERIMENT_PIPELINE_PLAN.md`, section
1.1:

* Irregular verb classes are assigned arbitrary, unstructured labels with no
  featural relationship to each other. A high regularization rate on wugs
  may partly reflect majority class dominance rather than genuine rule like
  generalization. The pipeline mitigates this by reporting results relative
  to an untrained baseline, but a full resolution would require redesigning
  the underlying model's label generation scheme, which is outside the scope
  of this pipeline.
* An Optuna trial that is still running when a process is killed is not
  automatically retried on resume. It is left orphaned in the storage file
  and a new trial is started in its place. This has no effect on
  correctness, only on wasted compute for that one trial.

## Troubleshooting

If `uv sync` reports that torch has no wheel for the current interpreter,
confirm that `.python-version` is present and set to `3.12`, then run
`uv sync` again. uv will download and manage a 3.12 interpreter
automatically if none is available locally.

If the pipeline is interrupted, simply run the same
`uv run python -m src.pipeline` command again. It is resumable by design
(see "How the pipeline is orchestrated" above).

If `mlflow ui` shows no runs, confirm it is pointed at the correct store
with `--backend-store-uri ./mlruns`, run from the project root.
