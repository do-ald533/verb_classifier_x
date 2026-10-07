# Verb Classifier X

A small PyTorch testbed for **verb past-tense regularization** (a Rumelhart and
McClelland style connectionist model, probed with the Marcus et al. wug
paradigm). A network learns a synthetic lexicon of verbs, most of which map to
one shared "regular" class while a minority are "irregular" (each with its own
class by default). We then ask how it treats **wugs**: novel inputs it has never
seen. Does it regularize them, and how does that depend on the lexicon, on how
often irregulars occur, on the network, and on the training budget?

The repository contains two ways to run experiments:

| | Declarative experiments (**new**) | Optuna grid (legacy) |
|---|---|---|
| Entry point | `python -m src {validate,run,report} file.json` | `python -m src.pipeline` |
| You describe | datasets, networks, seeds in a JSON file | nothing: a fixed grid of 198 conditions |
| Network hyperparameters | written by you, any number of networks | searched by Optuna per condition |
| Artifacts | content-addressed directories under `runs/` | MLflow + Optuna journal + `reports/` |
| Best for | controlled comparisons, ablations, anything new | the original pre-registered study |

The full design rationale of the legacy pipeline (two independent expert
reviews, the switch from doit to Prefect) is in `EXPERIMENT_PIPELINE_PLAN.md`.

## Contents

1. [Requirements and installation](#requirements-and-installation)
2. [Quick start](#quick-start)
3. [Declarative experiments](#declarative-experiments)
   * [The DAG](#the-dag)
   * [CLI reference](#cli-reference)
   * [Experiment file format](#experiment-file-format)
   * [Dataset spec reference](#dataset-spec-reference)
   * [Network spec reference](#network-spec-reference)
   * [How training works](#how-training-works)
   * [Hashing, caching and resuming](#hashing-caching-and-resuming)
   * [Output layout and file formats](#output-layout-and-file-formats)
   * [Observability](#observability)
   * [The report](#the-report)
   * [Recipes](#recipes)
4. [The legacy Optuna pipeline](#the-legacy-optuna-pipeline)
5. [Performance](#performance)
6. [Project structure](#project-structure)
7. [Testing and code quality](#testing-and-code-quality)
8. [Known limitations](#known-limitations)
9. [Troubleshooting](#troubleshooting)

---

## Requirements and installation

* Python 3.12 (pinned in `.python-version`; do not use 3.13, torch 2.4.1 has no
  wheel for it)
* [uv](https://docs.astral.sh/uv/) for the environment and for running every
  command below

```bash
uv sync
```

This creates `.venv`, installs Python 3.12 if needed, and installs everything
pinned in `pyproject.toml` / `uv.lock`: torch, prefect, mlflow, optuna, pydantic,
pandas, matplotlib, seaborn, scipy, orjson (fast JSON for events and
histories), tabulate and rich.

## Quick start

```bash
# 1. check the config and see the plan; nothing is written
uv run python -m src validate experiments/example_smoke.json

# 2. run it (12 tiny runs, about a minute); sequential, no server needed
uv run python -m src run experiments/example_smoke.json --engine local

# 3. read the report
$EDITOR runs/smoke/report/report.md
```

`experiments/example_smoke.json` crosses 2 datasets, 3 networks (an MLP, a deeper
tanh MLP and a small Transformer) and 2 seeds, with a tiny token budget so it
finishes quickly. Run the same command again and every node is skipped (cached);
add `--force` to recompute.

For parallel execution use the default engine, which runs the same nodes through
Prefect, one process per task:

```bash
uv run python -m src run experiments/example_smoke.json --max-workers 4
```

---

## Declarative experiments

An **experiment** is one JSON file. It declares *datasets*, *networks* and which
of them to cross with which *seeds*. Everything is validated before any compute
is spent (typos, impossible combinations, duplicate configurations all fail at
load time).

### The DAG

```
experiment.json
     │
 [validate]        pydantic, extra="forbid"; resolves defaults / extends / matrix / runs
     │
 [build_dataset]   one per distinct dataset spec   ── content-addressed, shared
     │
     ├─ [train]    one per (dataset, network, seed)  ── content-addressed, parallel
     │      │
     │  [evaluate]  re-loads the saved model, computes final metrics, mirrors to MLflow
     │
 [aggregate]       runs.csv, aggregate.csv (mean / std over seeds)
     │
 [report]          report.md + figures
```

Rules that make this a proper DAG:

* Every node reads and writes **only the disk**, in a directory named by the
  **hash of its spec**. Nothing flows through memory except small dicts
  (paths, manifests).
* A node whose output already exists is skipped, so any run is **resumable**.
* `train` of one run depends only on its own dataset, so all runs are
  independent and run in parallel. N networks over one dataset build that
  dataset **once**.
* A run-level failure never stops the DAG: `train` records `status =
  diverged | error` (with the message and traceback) in the manifest and
  returns normally; `evaluate` is skipped for that run; the report lists it
  under "Not-ok runs". Only infrastructure errors (disk, Prefect) propagate.
* Two engines run the same node functions (`src/nodes.py`):
  * `prefect` (default): tasks `build-dataset`, `train`, `evaluate`, wired by
    passing futures, executed by `ProcessPoolTaskRunner(max_workers=N)`. Tasks
    receive a light `ExperimentContext` (name, output dir, tracking, config hash)
    instead of the whole experiment, so submitting thousands of runs does not
    pickle thousands of copies of the run list.
  * `local`: plain sequential loops, no Prefect server or API. Used by the tests
    and for debugging.

  Both engines produce **identical hashes and bit-identical metrics** (each
  training uses its own explicit generators and one torch thread).

### CLI reference

```
python -m src validate CONFIG [--output-dir DIR]
python -m src run      CONFIG [--output-dir DIR] [--engine prefect|local]
                              [--max-workers N] [--force] [--no-report]
python -m src report   CONFIG [--output-dir DIR]
```

| Command | What it does |
|---|---|
| `validate` | Loads and resolves the file, prints the experiment name and config hash, every dataset and network with its hash, whether each dataset is `cached` or `to build`, and the count of runs by state (`pending`, `ok`, `diverged`, `error`). **Writes nothing.** |
| `run` | Executes the DAG. `--force` ignores cached datasets, trainings and evaluations. `--no-report` stops after aggregation. `--max-workers` (default 4) only applies to the `prefect` engine. |
| `report` | Re-aggregates and rebuilds the report from whatever is on disk. Does not train. |

`--output-dir` overrides the `output_dir` field of the file (default `runs`).

Exit codes: `0` all runs ok, `1` at least one run is not ok (they are listed),
`2` the config is invalid (message on stderr).

Environment: `VERB_TORCH_THREADS` (default `1`) sets the torch thread count
inside each training task. Leave it at 1 when running many workers.

### Experiment file format

```json
{
  "experiment": "smoke",
  "description": "Tiny end-to-end example: 2 datasets x 3 networks x 2 seeds.",
  "tracking": {"mlflow": false},
  "defaults": {
    "network": {"max_tokens": 6000, "eval_every_tokens": 2000, "early_stop": false}
  },
  "datasets": [
    {"id": "base", "spec": {"D": 6, "lexicon_size": 100, "p_irregular_types": 0.1,
                            "p_token_irregular": 0.7, "seed": 0}},
    {"id": "zipf_clustered", "extends": "base",
     "spec": {"freq_distribution": "zipf", "irregular_placement": "clustered",
              "n_clusters": 2, "class_scheme": "by_cluster", "wug_strategy": "near_irregular"}}
  ],
  "networks": [
    {"id": "mlp", "spec": {"hidden": [16], "optimizer": "adam", "lr": 0.01}},
    {"id": "mlp_deep_tanh", "extends": "mlp", "spec": {"hidden": [16, 16], "activation": "tanh"}},
    {"id": "transformer", "spec": {"arch": "transformer", "d_model": 16, "n_heads": 2,
                                    "n_layers": 1, "ff_dim": 32, "lr": 0.003}}
  ],
  "runs": [{"datasets": "*", "networks": "*", "seeds": [0, 1]}]
}
```

Top-level keys (anything else is rejected):

| Key | Type | Meaning |
|---|---|---|
| `experiment` | string, required | Name; becomes the directory `output_dir/<name>/`. Letters, digits, `_`, `.`, `-`. |
| `description` | string | Free text, copied into the report. |
| `output_dir` | string, default `runs` | Root for all artifacts. |
| `tracking` | object | `{"mlflow": true, "uri": null, "experiment": null}`. See [Observability](#observability). |
| `evaluation` | object | `{"accuracy_thresholds": [0.5, 0.9, 0.99]}`: thresholds for the `tokens_to_acc_*` learning-speed metrics. |
| `defaults` | object | `{"dataset": {...}, "network": {...}}`: base values applied to every entry that does not inherit through `extends`. |
| `datasets` | list, required | Dataset entries. |
| `networks` | list, required | Network entries. |
| `runs` | list, required | What to cross. |

**Entries** (`datasets[]` and `networks[]`) have:

* `id`: unique label (same character rule as `experiment`). Ids are only labels;
  renaming one does not change any hash.
* `extends`: id of another (non-matrix) entry to inherit from. Chains work;
  cycles are detected.
* `spec`: field overrides.
* `matrix`: `{"field": [values...]}`. Expands to the cartesian product. Each
  expansion gets the id `<id>.<k>=<v>[,<k>=<v>...]`, for example
  `mlp.hidden=[8],lr=0.1`, and remembers `<id>` as its **group**.
* `path` (datasets only): reuse an existing dataset directory (one that contains
  `spec.json`, `data.pt`, `stats.json`) instead of generating. Cannot be
  combined with `spec`, `matrix` or `extends`.

**Resolution order** (later wins): `defaults` < `extends` chain < the entry's own
`spec` < the `matrix` combination.

**`runs[]`** entries have `datasets`, `networks` (lists of ids, **group ids**, or
`"*"` for all) and `seeds` (non-empty list of ints). All listed combinations are
crossed with all seeds; the same (dataset, network, seed) listed twice is
deduplicated. Several `runs` entries let one file hold different "batteries":

```jsonc
"runs": [
  {"datasets": ["small"], "networks": ["mlp"],               "seeds": [0, 1, 2]},
  {"datasets": ["large"], "networks": ["mlp", "transformer"], "seeds": [0, 1, 2, 3, 4]}
]
```

**Seeds.** The run seed is written into `init_seed` and `data_order_seed` of the
network unless the network spec sets them explicitly. The dataset has its own
`seed` inside its spec and is *not* affected by the run seed. So by default all
seeds of a run share one dataset and differ in weight initialization and data
order. To resample the dataset too, add a `matrix` over the dataset `seed`.

**Validation errors** point at the offending entry, e.g.
`dataset 'big': D=8 with |alphabet|=3 gives only 6561 unique radicals, but
lexicon_size+n_wug=10030 are needed`. Two runs that resolve to the same content
hash (for example two networks with identical specs under different ids) are
rejected as duplicates.

### Dataset spec reference

A dataset is a **lexicon** of verbs plus a set of held-out **wugs**. Every verb
is a vector of `D` values from `alphabet` (a "radical"). Radicals are drawn
without replacement, so **no two verbs (or a verb and a wug) share an input**;
this is why the space must be big enough (`|alphabet|**D >= lexicon_size +
n_wug`). The first `n_regular` verbs are regular (class 0), the rest irregular.

| Field | Default | Meaning |
|---|---|---|
| `D` | required | Number of positions in a radical. |
| `alphabet` | `[-1, 0, 1]` | Values a position can take (at least 2 distinct). The radical space is `|alphabet|**D`, capped at 10,000,000. |
| `encoding` | `raw` | `raw`: the model input is the `D` values. `one_hot`: each position becomes a one-hot vector over the alphabet (input size `D*|alphabet|`). |
| `lexicon_size` | required | Number of verb types. |
| `p_irregular_types` | required | Fraction of types that are irregular, in (0, 1). `n_irregular = round(lexicon_size * p_irregular_types)`; both groups must be non-empty. |
| `n_wug` | `30` | Number of held-out novel inputs. |
| `p_token_irregular` | required | Probability that a sampled *token* is irregular, in [0, 1]. |
| `freq_distribution` | `uniform` | How tokens spread inside each group: `uniform`, or `zipf` (weight `1/rank**zipf_exponent`, ranks assigned at random per group). |
| `zipf_exponent` | `1.0` | Zipf exponent, used when `freq_distribution = zipf`. |
| `class_scheme` | `unique_per_irregular` | Labels of irregular verbs. `unique_per_irregular`: each its own class (`n_classes = n_irregular + 1`). `shared_k`: classes `1..k_classes`, assigned in balanced random fashion. `by_cluster`: one class per cluster (needs `irregular_placement = clustered`; `n_classes = n_clusters + 1`). |
| `k_classes` | none | Required (and only valid) with `shared_k`; `1 <= k <= n_irregular`. |
| `irregular_placement` | `random` | `random`: irregulars are arbitrary lexicon items. `clustered`: irregulars are grouped by Hamming similarity (see below). |
| `n_clusters` | `1` | Number of clusters, at most `n_irregular`. |
| `wug_strategy` | `random` | Where wugs sit relative to the lexicon: `random`, `near_irregular` (smallest Hamming distance to the nearest irregular), `far_from_lexicon` (largest distance to the nearest lexicon item). |
| `sampling` | `stream` | `stream`: every training token is a fresh i.i.d. draw. `finite`: `train_size` tokens are drawn once and recycled. |
| `train_size` | none | Required (and only valid) with `sampling = finite`. |
| `seed` | `0` | Seed of everything the dataset generates. |
| `schema_version` | current | Bumped when a change alters generated data; part of the hash. |

Details worth knowing:

* **Token sampling is two-stage.** A token is irregular with probability
  `p_token_irregular`; then a verb is drawn inside that group by its
  within-group weights (uniform or Zipf). The per-type token probability is
  stored as `p_types` and defines the *expected loss* used for convergence.
* **Clustered placement.** `n_clusters` random lexicon items are chosen as
  centers; each takes its nearest (Hamming distance, random tie-break) unused
  lexicon items until its share of the irregular budget is met. The effect is
  only as strong as the lexicon is dense: at `D=6` with 100 verbs the lexicon
  covers 14% of the space and clusters barely tighten. Use a larger `D` or a
  smaller lexicon for a visible effect (the dataset `stats.json` reports
  `irregular_mean_pairwise_hamming` so you can check).
* **Wug pool.** For `near_irregular` and `far_from_lexicon` the candidates are
  the `n_wug` random wugs plus up to 5,000 further unused radicals; the best
  `n_wug` by the strategy's distance are kept. Wugs are always disjoint from the
  lexicon.
* **`finite` sampling** cycles the stored tokens in a reshuffled order each
  pass, using the network's `data_order_seed`. Convergence is still judged by the
  population expected loss, not by loss on the finite sample.

### Network spec reference

| Field | Default | Meaning |
|---|---|---|
| `arch` | `mlp` | `mlp` or `transformer`. |
| `hidden` | `[16]` | MLP layer widths. `[]` is a plain linear classifier. Ignored by the transformer. |
| `activation` | `relu` | `relu`, `tanh`, `gelu`, `sigmoid`, `leaky_relu`, `silu`. |
| `dropout` | `0.0` | Dropout probability in [0, 1). |
| `norm` | `none` | `none`, `layernorm`, `batchnorm` (MLP only; requires `batch_size >= 2`). |
| `init` | `kaiming` | `kaiming` (the `nn.Linear` default), `xavier`, `normal` (std `1/sqrt(fan_in)`). Always driven by an explicit generator. |
| `bias` | `true` | Bias terms in hidden layers / transformer. |
| `d_model`, `n_heads`, `n_layers`, `ff_dim` | `32, 2, 1, 64` | Transformer size. `d_model` must be divisible by `n_heads`. |
| `optimizer` | `adam` | `sgd`, `adam`, `adamw`. |
| `lr` | `0.01` | Learning rate. |
| `momentum` | `0.0` | SGD momentum. |
| `weight_decay` | `0.0` | L2 / decoupled decay (for `adamw`). |
| `lr_schedule` | `const` | `const`, `cosine` (to 0 over `max_tokens`), `linear_decay` (to 0 over `max_tokens`). |
| `grad_clip` | none | Global gradient-norm clip. |
| `batch_size` | `1` | Tokens per optimizer step. `1` is pure token-by-token learning. |
| `init_seed` | run seed | Seeds weight initialization. |
| `data_order_seed` | run seed | Seeds the token stream (and dropout masks). |
| `max_tokens` | `5,000,000` | Budget in tokens (steps = `max_tokens // batch_size`). |
| `eval_every_tokens` | `10,000` | Checkpoint interval (metrics are computed and logged here). |
| `early_stop` | `true` | Stop when the expected loss has not improved by `early_stop_min_delta` for `early_stop_patience` consecutive checkpoints. |
| `early_stop_patience`, `early_stop_min_delta` | `5`, `1e-4` | Early-stop settings. |
| `prune_interval_steps`, `max_prune_pct` | none | Magnitude pruning: every `prune_interval_steps` optimizer steps, zero the smallest-magnitude still-active parameter permanently, until `max_prune_pct` percent are pruned. Set both or neither. |

The **transformer** treats each of the `D` radical positions as a token: its
value features (1 for `raw`, `|alphabet|` for `one_hot`) go through a linear
embedding, a learned position embedding is added, then `n_layers` pre-norm
encoder blocks (each has its own LayerNorm), mean pooling over positions, and a
linear head. `norm = layernorm` adds one extra LayerNorm before the head.

### How training works

* **Streaming tokens.** There is no epoch and no fixed dataset size. Each
  checkpoint interval draws `eval_every_tokens` tokens (two-stage, see above),
  gathers their inputs and labels once, then steps through them. Progress is
  always reported against **cumulative tokens seen**.
* **Two independent generators.** One for initialization (`init_seed`), one for
  the token stream (`data_order_seed`). This lets you isolate sources of
  variance: fix `init_seed` and vary the run seed to measure data-order noise
  only, or the reverse. If `dropout > 0`, the global torch RNG is also seeded
  with `data_order_seed`, because dropout masks draw from it.
* **Convergence is judged on the population.** Each checkpoint computes the
  *expected loss*: per-type cross entropy weighted by each type's token
  probability. Single-token losses are far too noisy for early stopping.
  `train_loss` (mean of the minibatch losses since the last checkpoint) is
  logged as well.
* **Metrics at every checkpoint** (`history.jsonl`): `tokens_seen`, `train_loss`,
  `lr`, `loss` (expected), `acc_types`, `regular_acc`, `irregular_acc`,
  `overregularization_rate` (fraction of irregular types predicted as the
  regular class), `param_norm`, the four wug metrics (`wug_regular_prob`,
  `wug_regular_rate`, `wug_entropy`, `wug_top2_margin`) and `active_pct`
  (percent of parameters not pruned). The primary metric is
  **`wug_regular_prob`**: mean softmax mass on the regular class over the wugs.
  It is continuous, unlike the discrete `wug_regular_rate`, so it is much less
  noisy with few wugs.
* **Baseline.** The wug metrics of the *untrained* network are stored as
  `baseline_*`, so results can be read relative to chance behavior.
* **Divergence.** Non-finite training loss is detected (checked every 256
  steps and at each checkpoint) and recorded as `status = diverged`.
* **Final evaluation** (`evaluate` node) reloads the saved model and recomputes
  every metric from scratch. It warns (`evaluate_mismatch`) if the recomputed
  loss disagrees with the last checkpoint. It adds `tokens_to_acc_<pct>` (first
  checkpoint where `acc_types` reached each threshold, null if never) and saves
  `predictions.npz` (predicted and true class per type, wug predictions and
  probabilities).

### Hashing, caching and resuming

Every spec (`DatasetSpec`, `NetworkSpec`) is a frozen pydantic model whose
`hash` is the first 16 hex characters of the SHA-256 of its canonical JSON
(sorted keys, all defaults explicit, includes `schema_version`). Three hashes
matter:

| Hash | Of | Used as |
|---|---|---|
| `dataset_hash` | the dataset spec (including its seed) | `runs/datasets/<dataset_hash>/` |
| `run_hash` | dataset hash + the network spec **with resolved seeds** | `runs/<experiment>/trainings/<run_hash>/` |
| `config_hash` | all dataset hashes, network hashes and run hashes | printed by `validate`, stored in the report and in every MLflow run |

Consequences:

* Same content gives the same directory, whatever you named it. Datasets are
  therefore shared across networks, across seeds and across **experiments**.
* Changing any field changes the hash and therefore creates a new directory.
  Old directories are left in place (there is no garbage collection; delete
  them by hand when you want the space back).
* Rerunning skips: `build_dataset` if the dataset directory is complete;
  `train` if its manifest has status `ok` or `diverged` (deterministic
  outcomes; `error` runs are retried); `evaluate` if `metrics.json` exists.
  Look for `*_reused` events in `events.jsonl`.
* Dataset directories are written to a temp directory and renamed into place, so
  a crash or a concurrent reader never sees a half-written dataset. JSON files
  are written the same way.
* Every `train` verifies that the dataset it loads has the hash it expects.
* If you change the code in a way that alters results (not just the config),
  the hashes do not know. Use `--force`, or bump `SCHEMA_VERSION` in
  `src/specs.py`, which changes every hash.

### Output layout and file formats

```
runs/                                      (output_dir)
├── datasets/<dataset_hash>/               shared, content-addressed
│   ├── spec.json                          the DatasetSpec
│   ├── data.pt                            radicals, labels, group weights (+ train_indices if finite)
│   └── stats.json                         descriptive statistics (see below)
├── mlflow.db, mlflow_artifacts/           default MLflow store (if tracking is on)
└── <experiment>/
    ├── config.json                        your file, verbatim
    ├── resolved_config.json               after defaults/extends/matrix, with every hash
    ├── events.jsonl                       experiment-level event stream
    ├── runs.csv                           one row per planned run
    ├── aggregate.csv                      mean/std per (dataset, network)
    ├── report/report.md, report/figures/
    └── trainings/<run_hash>/
        ├── spec.json                      ids, hashes, dataset dir, full network spec, seed
        ├── manifest.json                  status, timing, environment
        ├── events.jsonl                   this run's events (also copied to the experiment stream)
        ├── history.jsonl                  one line per checkpoint
        ├── train_summary.json             baseline, last checkpoint, stop reason, n_params
        ├── model.pt                       state_dict + network spec
        ├── metrics.json                   final metrics
        └── predictions.npz
```

**`stats.json`** (per dataset): `n_types`, `n_regular`, `n_irregular`,
`n_classes`, `input_dim`, `space_size`, `lexicon_density`, `token_entropy_nats`,
`max_type_token_prob`, `irregular_mean_pairwise_hamming`,
`wug_mean_min_hamming_to_lexicon`, `wug_mean_min_hamming_to_irregular`,
`class_count_min`, `class_count_max`, `n_train_tokens_finite`.

**`manifest.json`** (per run): `status` (`ok`, `diverged`, `error`), `stop_reason`
(`max_tokens` or `early_stopped`), `tokens_seen`, `n_params`, `started_at`,
`finished_at`, `wall_clock_seconds`, the ids and hashes, `mlflow_run_id` when
mirrored, and for failures `error` (+ `traceback`). Its `environment` block holds
git sha, branch and dirty flag, Python / torch / numpy versions, platform, CPU
count, torch threads and CUDA availability.

**`runs.csv`** columns: `dataset_id`, `network_id`, `seed`, the three hashes,
`status` (`ok`, `diverged`, `error`, `missing`), `error`, every dataset field as
`dataset.<field>`, every network field as `network.<field>` (lists are JSON
strings), and, for ok runs, every final metric. **`aggregate.csv`** has
`n_runs`, `n_ok` and `<metric>_mean` / `<metric>_std` over the ok seeds of each
(dataset, network).

### Observability

Everything needed to explain a number is on disk, and every artifact points back
to its inputs.

**Structured events.** `events.jsonl` (per run and per experiment) is JSON Lines:
one event per line, each with `ts`, `level`, `event` and the bound context
`experiment`, `stage`, `dataset_id`, `network_id`, `seed`, `run_hash`,
`dataset_hash`. Lines are written with a single `write` call, so concurrent
workers can append to the shared experiment stream safely. Events:

| Stage | Events |
|---|---|
| setup / done | `experiment_start` (config hash, counts, environment), `experiment_done` |
| dataset | `dataset_build_start` (full spec), `dataset_built` (seconds + stats), `dataset_reused` |
| train | `train_start` (network spec), `checkpoint` (every metric), `train_done`, `train_failed` (WARNING for divergence, ERROR otherwise), `train_reused` |
| evaluate | `evaluate_done` (metrics), `evaluate_reused`, `evaluate_skipped` (training not ok), `evaluate_mismatch` |
| aggregate / report | `aggregate_done` (status counts), `report_done` |

Handy queries:

```bash
# everything that went wrong in an experiment
jq -c 'select(.level != "INFO")' runs/smoke/events.jsonl

# learning curve of one run
jq -c 'select(.event == "checkpoint") | [.tokens_seen, .loss, .wug_regular_prob]' \
   runs/smoke/trainings/<run_hash>/events.jsonl
```

The console shows the same events (without the chatty `checkpoint`), and the
pipeline logger also writes `logs/pipeline.log`.

**Manifests and lineage.** From any report row you can get to its `run_hash`,
then to `trainings/<run_hash>/spec.json` (exact network, seed, dataset
directory), `manifest.json` (git sha and whether the tree was dirty, library
versions) and the dataset's `spec.json` / `stats.json`.

**MLflow mirror.** With `tracking.mlflow = true` (the default), the `evaluate`
node logs one MLflow run per training: params `dataset.*`, `network.*`,
`dataset_id`, `network_id`, `seed`; tags `experiment`, `run_hash`,
`dataset_hash`, `network_hash`, `config_hash`, `git_sha`; every checkpoint metric
with `step = tokens_seen`; `final_*` metrics; and `manifest.json`, `metrics.json`
and `spec.json` as artifacts. The default store is SQLite inside the output
directory (`<output_dir>/mlflow.db`, artifacts in `<output_dir>/mlflow_artifacts`)
because MLflow's file-store backend is deprecated. Override with
`tracking.uri` / `tracking.experiment`. MLflow is a **mirror**: if it is
unreachable a warning is logged and the run is unaffected; the disk artifacts
are the source of truth. The mirror happens once, after evaluation, not live.

```bash
uv run mlflow ui --backend-store-uri sqlite:///runs/mlflow.db
```

**Prefect.** Each node is a Prefect task, so with a Prefect server the DAG,
task states and logs are visible in its UI. No server is required; Prefect uses
a temporary local API otherwise.

### The report

`runs/<experiment>/report/report.md` contains: the description, generation time,
`config_hash`, git sha / branch / dirty flag, run counts, the command that
reproduces it, and then

1. a **Datasets** table (id, hash, D, alphabet size, lexicon, irregular type and
   token shares, frequency distribution, classes, placement, wug strategy,
   sampling, token entropy, density),
2. a **Networks** table (id, hash, architecture and shape, activation,
   optimizer, batch, dropout, norm, pruning),
3. a **Results** table of mean ± sd over seeds: `wug_regular_prob`, type
   accuracy, irregular accuracy, overregularization, tokens seen, tokens to 90%
   accuracy, and `ok/runs` per (dataset, network),
4. the **not-ok runs** with their status and error,
5. figures: `learning_curves_<dataset>.png` (loss, `acc_types`, `irregular_acc`,
   `wug_regular_prob` against tokens seen, one line per network, band = sd over
   seeds) and `final_metrics.png` (final `wug_regular_prob` and `irregular_acc`
   by dataset and network).

The tables are plain markdown and `runs.csv` / `aggregate.csv` have everything
for your own analysis (for example with pandas).

### Recipes

Every `json` block in this README that contains an `"experiment"` key is
resolved by `tests/test_readme.py`, so these examples stay valid.

**Sweep the lexicon size** (the number of verbs; larger lexicons need larger `D`):

```json
{
  "experiment": "lexicon_size",
  "tracking": {"mlflow": false},
  "defaults": {"network": {"max_tokens": 2000000}},
  "datasets": [
    {"id": "lex",
     "spec": {"D": 10, "p_irregular_types": 0.05, "p_token_irregular": 0.7},
     "matrix": {"lexicon_size": [100, 500, 1000, 5000]}}
  ],
  "networks": [{"id": "mlp", "spec": {"hidden": [32], "lr": 0.003}}],
  "runs": [{"datasets": ["lex"], "networks": ["mlp"], "seeds": [0, 1, 2, 3, 4]}]
}
```

`"datasets": ["lex"]` selects every expansion through the group id. Each of the 4
datasets is built once and trained with 5 seeds.

**Token frequency: skew and shape** (2 x 2 datasets):

```json
{
  "experiment": "frequency",
  "tracking": {"mlflow": false},
  "datasets": [
    {"id": "freq",
     "spec": {"D": 8, "lexicon_size": 500, "p_irregular_types": 0.05},
     "matrix": {"p_token_irregular": [0.5, 0.9],
                "freq_distribution": ["uniform", "zipf"]}}
  ],
  "networks": [{"id": "mlp", "spec": {}}],
  "runs": [{"datasets": "*", "networks": "*", "seeds": [0, 1, 2]}]
}
```

**Structured irregulars**: do irregulars that resemble each other (and share a
class) behave differently from arbitrary ones?

```json
{
  "experiment": "structure",
  "tracking": {"mlflow": false},
  "defaults": {"dataset": {"D": 8, "lexicon_size": 300, "p_irregular_types": 0.1,
                           "p_token_irregular": 0.7}},
  "datasets": [
    {"id": "arbitrary", "spec": {}},
    {"id": "shared_classes", "spec": {"class_scheme": "shared_k", "k_classes": 3}},
    {"id": "clustered", "spec": {"irregular_placement": "clustered", "n_clusters": 3,
                                  "class_scheme": "by_cluster"}}
  ],
  "networks": [{"id": "mlp", "spec": {"hidden": [32]}}],
  "runs": [{"datasets": "*", "networks": "*", "seeds": [0, 1, 2]}]
}
```

**Compare architectures and activations** on one dataset:

```json
{
  "experiment": "architectures",
  "tracking": {"mlflow": false},
  "datasets": [{"id": "d", "spec": {"D": 8, "lexicon_size": 500,
                                     "p_irregular_types": 0.05, "p_token_irregular": 0.7}}],
  "networks": [
    {"id": "mlp", "spec": {"hidden": [32], "lr": 0.003},
     "matrix": {"activation": ["relu", "tanh", "gelu"]}},
    {"id": "deep", "spec": {"hidden": [32, 32, 32], "norm": "layernorm", "lr": 0.003}},
    {"id": "transformer", "spec": {"arch": "transformer", "d_model": 32, "n_heads": 4,
                                    "n_layers": 2, "ff_dim": 64, "lr": 0.001}}
  ],
  "runs": [{"datasets": ["d"], "networks": "*", "seeds": [0, 1, 2]}]
}
```

**Isolate a source of variance.** Pinning `init_seed` makes the run seed control
only the data order; pinning `data_order_seed` does the reverse:

```json
{
  "experiment": "variance",
  "tracking": {"mlflow": false},
  "datasets": [{"id": "d", "spec": {"D": 6, "lexicon_size": 200,
                                     "p_irregular_types": 0.1, "p_token_irregular": 0.7}}],
  "networks": [
    {"id": "only_order_varies", "spec": {"init_seed": 0}},
    {"id": "only_init_varies", "spec": {"data_order_seed": 0}}
  ],
  "runs": [{"datasets": ["d"], "networks": "*", "seeds": [1, 2, 3, 4, 5]}]
}
```

The seeds start at 1 on purpose: with run seed 0 both networks would resolve to
`init_seed = 0` and `data_order_seed = 0`, i.e. the same run, and the loader
rejects that as a duplicate.

**Reuse a dataset you already generated** (path to a directory containing
`spec.json`, `data.pt`, `stats.json`, for instance one from another experiment):

```jsonc
"datasets": [{"id": "reused", "path": "runs/datasets/7f23bd2f5fbea625"}]
```

Nothing is regenerated, and its hash is the one in its `spec.json`.

**Tips.** `validate` first, always: it is free and shows what is cached. Keep
`max_tokens` small while designing, then raise it. Larger `batch_size` is the
biggest speed lever (see [Performance](#performance)). `runs.csv` and the report
list only the runs currently in the file: if you edit a spec and rerun, the new
runs get new hashes and directories, and the old ones stay on disk but drop out
of the report.

---

## The legacy Optuna pipeline

The original study: a full grid of linguistic **conditions**, a hyperparameter
search per condition with Optuna, a confirmation phase on fresh seeds, then a
statistical report. It still works and now runs on the same data generation and
training code as the declarative experiments (through adapters
`LinguisticConfig.to_dataset_spec` and `NetworkConfig.to_network_spec`).

### The grid

`src/config.py` defines linguistic factors, whose full cross product is
**198 conditions** (99 single-objective, 99 with pruning):

| Factor | Values |
|---|---|
| `D` | 6, 8, 10 |
| `LEXICON_SIZE` | 100, 500, 1000, 5000, 10000 |
| `P_IRREGULAR_TYPES` | 0.02, 0.05, 0.10 |
| `P_TOKEN_IRREGULAR` | 0.5, 0.7, 0.9 |
| pruning | off, on |

Infeasible `(D, LEXICON_SIZE)` pairs (`3**D < LEXICON_SIZE + 30`) are skipped, so
only 11 of the 15 pairs exist; for example 10,000 verbs need `D = 10`. Each
irregular verb has its own class (`N_IRREGULAR + 1` classes per condition) and
`N_IRREGULAR = round(LEXICON_SIZE * P_IRREGULAR_TYPES)`.

Network factors are searched by Optuna inside each condition
(`src/objective.py::_suggest_network_config`): optimizer (SGD/Adam), hidden width
4 to 32 (log), 1 to 3 hidden layers, learning rate 1e-3 to 0.3 (log), momentum 0
to 0.9 (SGD), and with pruning the interval (100 to 20,000 steps, log) and the
target sparsity (1% to 50%).

### Phases

1. **Search.** 35 Optuna trials per condition (60 with pruning, a two-objective
   search), each averaged over 5 seeds, purely to rank configurations.
2. **Confirmation.** The best configuration (for pruning conditions, every point
   of the accuracy versus sparsity Pareto front) is rerun on 20 fresh seeds. Only
   confirmation runs feed any leaderboard or comparison.

Training uses the streaming token sampler described above with a budget of
5,000,000 tokens, evaluated every 10,000 tokens (`MAX_TOKENS`,
`EVAL_EVERY_TOKENS`, `BATCH_SIZE` in `src/config.py`).

### Orchestration

`src/pipeline.py` is a Prefect flow: each condition becomes `sweep-condition`
then `confirm-condition`; conditions run in parallel in separate processes
(`ProcessPoolTaskRunner`, matching Optuna's multi-process-safe `JournalStorage`);
the report runs once everything has confirmed. Resumability does not depend on
Prefect: each task checks a marker under `reports/.done/` and writes it only on
success.

```bash
uv run python -m src.pipeline --max-workers 8                       # sweep + confirm + report
uv run python -m src.pipeline --stage sweep-and-confirm --max-workers 8
uv run python -m src.pipeline --stage report                        # rebuild the report only
```

Run a single condition directly while developing:

```bash
uv run python -c "
from src.config import generate_condition_grid
from src.sweep import run_condition_study
run_condition_study(generate_condition_grid()[0])
"
```

Set `--max-workers` to at most the number of cores; `1` runs sequentially.
`ProcessPoolTaskRunner` can be swapped for `ThreadPoolTaskRunner` in
`pipeline.py::main()` if something turns out to be unpicklable (at the cost of
CPU parallelism).

### Where results go

* **MLflow**, in MLflow's default store, which in the installed version is
  `./mlflow.db` (SQLite) in the directory you launch from. One root run per
  condition, nested trial runs, nested seed runs, separate confirmation runs.

  ```bash
  uv run mlflow ui --backend-store-uri sqlite:///mlflow.db
  ```

* **Optuna**: `optuna_journal.log` (a journal file).
* **`reports/`**: `leaderboard.csv` (top confirmation runs by `wug_regular_prob`),
  `best_config.json` (loadable into `LinguisticConfig` and `NetworkConfig`),
  `report.md`, `figures/`.

`reports/report.md` separates **confirmatory** findings (four hypotheses declared
in advance, `EXPERIMENT_PIPELINE_PLAN.md` section 4a: frequency skew
`P_TOKEN_IRREGULAR`, irregular set size `N_IRREGULAR`, dimensionality `D`,
pruning; Kruskal-Wallis plus pairwise Mann-Whitney U, Benjamini-Hochberg
corrected) from **exploratory** breakdowns (every other cut, not corrected, to
be read as descriptive). The figures include the overregularization confusion
matrix, the wug response distribution against the untrained baseline,
per-condition hyperparameter importance and correlation, and the accuracy versus
sparsity Pareto front.

Note that `N_IRREGULAR` (H2) now varies with both `LEXICON_SIZE` and
`P_IRREGULAR_TYPES`, so H2 pools lexicon sizes; the stats module was not changed
when those factors were added.

---

## Performance

Token-by-token training (`batch_size = 1`) is dominated by Python and autograd
overhead per step, not by arithmetic, so optimizations target the loop:

* **Fused Adam / AdamW** (one kernel over all parameters): about 30% faster
  steps on CPU for these model sizes.
* **No per-step finite check.** The loss is accumulated on-device and checked
  every 256 steps and at every checkpoint, instead of forcing a check each step.
* **Inputs and labels gathered once per checkpoint interval**, then sliced.
* **Mask handling** for pruning uses precomputed parameter/mask pairs.
* **One forward pass** for wug probabilities and predictions; `inference_mode`
  for evaluation.
* **Hashes are computed once** (cached on frozen specs, on the run spec and the
  experiment), the environment probe (three `git` calls) is cached per process,
  and tasks receive a light context instead of the whole experiment.
* **orjson** for events, histories and manifests.

Measured on one CPU thread (40,000 tokens, 1000-verb lexicon, D=8; before then
after):

| Case | Before | After |
|---|---|---|
| MLP `[16]`, batch 1 | 2,800 tok/s | 4,160 tok/s |
| MLP `[16]`, batch 1, pruning | 2,650 tok/s | 3,880 tok/s |
| MLP `[16]`, batch 32 | 90,000 tok/s | 132,000 tok/s |
| Transformer d=16, batch 1 | 806 tok/s | 1,189 tok/s |

What to do about it in practice:

* The biggest lever is `batch_size`: 32 is roughly 30x more tokens per second
  than 1. Keep `batch_size = 1` only when token-by-token dynamics are the point.
* Use as many workers as cores and leave `VERB_TORCH_THREADS = 1`.
* Datasets are built once and shared, so cost scales with the number of runs,
  not with networks per dataset.
* Raise `eval_every_tokens` if checkpoints (each evaluates the whole lexicon) are
  a noticeable fraction of the time, for example with 10,000-verb lexicons.

Fused Adam changes floating point rounding, which batch-1 training amplifies over
tens of thousands of steps. Results are deterministic and identical across
engines, but not bit-identical to runs made with the unfused optimizer. With
`fused=False` the pre-optimization numbers are reproduced exactly, so the other
changes do not alter results.

---

## Project structure

```
verb_classifier_x/
├── README.md                      this file
├── EXPERIMENT_PIPELINE_PLAN.md    design document and review history of the legacy pipeline
├── pyproject.toml, uv.lock        dependencies; ruff, ty, pytest configuration
├── experiments/
│   └── example_smoke.json         small end-to-end example
├── scripts/check.sh               lint, format check, type check, tests
├── src/
│   │  declarative experiments
│   ├── specs.py                   DatasetSpec / NetworkSpec: validation, hashing
│   ├── config_loader.py           JSON -> resolved experiment (defaults/extends/matrix/runs)
│   ├── data.py                    dataset generation, token sampling, persistence
│   ├── model.py                   MLP and transformer factories, explicit-generator init
│   ├── train.py                   the training loop (+ legacy adapter train_one_run)
│   ├── evaluation.py              point-in-time metrics, predictions, tokens-to-accuracy
│   ├── nodes.py                   the DAG nodes and the on-disk layout
│   ├── flow.py                    Prefect flow and the local engine
│   ├── obs.py                     JSONL events, manifests, MLflow mirror, orjson helpers
│   ├── exp_report.py              aggregate -> report.md and figures
│   ├── cli.py, __main__.py        python -m src {validate,run,report}
│   │  shared
│   ├── metrics.py                 wug metrics (primary: wug_regular_prob)
│   ├── pruning.py                 magnitude pruning
│   ├── logging_config.py          console + rotating file logger
│   │  legacy Optuna pipeline
│   ├── config.py                  198-condition grid, LinguisticConfig / NetworkConfig
│   ├── objective.py, sweep.py     Optuna objective and per-condition study
│   ├── confirm.py                 confirmation on fresh seeds
│   ├── pipeline.py                Prefect flow: sweep -> confirm -> report
│   ├── stats.py, leaderboard.py, report.py   hypothesis tests, ranking, report
└── tests/                         config, data, model, train, metrics, pruning, stats, experiment
```

Created at run time (git-ignored): `runs/`, `mlruns/`, `mlflow.db`,
`optuna_journal.log`, `logs/`, `reports/.done/`.

## Testing and code quality

```bash
uv run pytest                          # tests (with coverage)
uv run ruff check src tests            # lint
uv run ruff format --check src tests   # format check
uv run ty check src tests              # type check
bash scripts/check.sh                  # all four, in order
```

The tests cover spec validation and hashing; dataset properties (uniqueness of
radicals, group sizes, class schemes, clustering, wug strategies, Zipf sampling
against its declared probabilities, round-trip persistence); the model factories;
training options (every optimizer, schedule, norm, dropout, transformer, finite
sampling, pruning, early stop, divergence, seed independence, determinism); and
the experiment layer end to end through the local engine (artifacts, traceable
events, caching, evaluation consistency, failure recording, dataset reuse, CLI).

The Prefect engine is exercised manually rather than in the test suite because it
starts a temporary API server.

These checks are independent of each other, so they are not part of any DAG.

**Current state of the gate:** `pytest` passes, but `ruff check`, `ruff format
--check` and `ty check` still report findings (mostly style and typing issues in
the legacy modules, plus some style nits in the new tests). The gate does not yet
exit cleanly; this is known and unrelated to behavior.

## Known limitations

* **Labels carry little structure by default.** With `class_scheme =
  unique_per_irregular` an irregular verb's class has no relationship to its
  input. High wug regularization may partly reflect majority-class dominance
  rather than rule-like generalization. Results are best read against the
  `baseline_*` metrics and across the `shared_k` / `by_cluster` / `clustered`
  variants. Only `clustered` + `by_cluster` ties classes to input similarity.
* **Clustering is weak in sparse regimes** (see the dataset notes); check
  `irregular_mean_pairwise_hamming` in `stats.json`.
* **The Optuna pipeline is not part of the declarative layer.** There is no
  `search` node yet; networks in the JSON are written by hand. An Optuna trial
  killed mid-run is left orphaned in the journal and a new one starts in its
  place (wasted compute only).
* **The legacy statistics are fixed** to the four pre-registered hypotheses and
  were not updated for the new lexicon factors.
* **No garbage collection** of `runs/`; old hashes stay until you delete them.
* **MLflow is mirrored after evaluation**, so there are no live curves in the
  MLflow UI while a run trains (use `events.jsonl` for that).
* **Prefect temporary server noise**: with several workers the ephemeral SQLite
  API logs "database is locked"; results are unaffected. A real Prefect server
  avoids it.
* **The code-change caveat of content hashing**: hashes cover specs, not source
  code (see [Hashing](#hashing-caching-and-resuming)).
* **Quality gate** is not green yet (see above).

## Troubleshooting

* **`uv sync` says torch has no wheel for this interpreter**: confirm
  `.python-version` is `3.12` and run `uv sync` again; uv fetches a 3.12
  interpreter if needed.
* **`config error: ...`** (exit code 2): the message names the entry and field.
  Run `validate` after every edit.
* **"radicals ... are needed"**: the radical space is too small for the lexicon.
  Raise `D` or the alphabet size, or lower `lexicon_size` / `n_wug`.
* **"two runs resolve to the same content hash"**: two entries are identical
  after resolution. Remove one, or make them differ.
* **A run shows `diverged`**: the learning rate is too high for the optimizer.
  The error in the manifest names the step. Lower `lr` or add `grad_clip`.
* **A run shows `error`**: see `error` and `traceback` in its `manifest.json` and
  the `train_failed` event. Rerunning retries `error` runs only.
* **Nothing reruns after I changed the code**: hashes cover specs, not code.
  Use `--force`.
* **MLflow shows no runs**: point the UI at the right store. Declarative
  experiments: `sqlite:///<output_dir>/mlflow.db`. Legacy pipeline:
  `sqlite:///mlflow.db`, from the directory the pipeline ran in. MLflow
  mirroring failures appear as warnings in the console and in `pipeline.log`.
* **Many `database is locked` lines from Prefect**: harmless, see above.
* **Interrupted pipeline** (Ctrl+C, crash, kill): run the same command again; it
  resumes (declarative experiments through the hash cache, the legacy pipeline
  through `reports/.done/` markers).
