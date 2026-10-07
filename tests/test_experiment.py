import copy
import json

import pandas as pd
import pytest
from src.cli import main as cli_main
from src.config_loader import ConfigError, resolve_experiment
from src.flow import run_local
from src.nodes import Layout, collect_runs
from src.obs import read_json, read_jsonl

BASE = {
    "experiment": "t",
    "tracking": {"mlflow": False},
    "defaults": {"network": {"max_tokens": 400, "eval_every_tokens": 200, "early_stop": False}},
    "datasets": [
        {
            "id": "base",
            "spec": {
                "D": 6,
                "lexicon_size": 100,
                "p_irregular_types": 0.1,
                "p_token_irregular": 0.7,
            },
        },
        {"id": "zipf", "extends": "base", "spec": {"freq_distribution": "zipf"}},
    ],
    "networks": [
        {"id": "mlp", "spec": {"hidden": [8]}},
        {"id": "wide", "extends": "mlp", "spec": {"hidden": [16]}},
    ],
    "runs": [{"datasets": "*", "networks": "*", "seeds": [0, 1]}],
}


def _cfg(**changes):
    cfg = copy.deepcopy(BASE)
    cfg.update(changes)
    return cfg


# ----------------------------------------------------------------- resolution
def test_extends_defaults_and_cross_product():
    exp = resolve_experiment(_cfg())
    assert len(exp.runs) == 2 * 2 * 2
    assert exp.datasets["zipf"].spec.D == 6  # inherited
    assert exp.datasets["zipf"].spec.freq_distribution == "zipf"  # overridden
    assert exp.networks["wide"].spec.max_tokens == 400  # from defaults
    assert exp.networks["wide"].spec.hidden == (16,)


def test_matrix_expansion_and_group_selection():
    cfg = _cfg(
        networks=[
            {
                "id": "mlp",
                "spec": {"activation": "relu"},
                "matrix": {"hidden": [[8], [16, 16]], "lr": [0.1, 0.01]},
            }
        ],
        runs=[{"datasets": ["base"], "networks": ["mlp"], "seeds": [0]}],
    )
    exp = resolve_experiment(cfg)
    assert len(exp.networks) == 4  # group id selects every expansion
    assert len(exp.runs) == 4
    assert {n.group for n in exp.networks.values()} == {"mlp"}


def test_hashes_are_stable_and_content_addressed():
    a, b = resolve_experiment(_cfg()), resolve_experiment(_cfg())
    assert a.config_hash == b.config_hash
    assert [r.run_hash for r in a.runs] == [r.run_hash for r in b.runs]
    changed = _cfg()
    changed["datasets"][0]["spec"]["lexicon_size"] = 120
    assert resolve_experiment(changed).config_hash != a.config_hash
    # ids are labels only: renaming a dataset does not change its hash
    renamed = _cfg()
    renamed["datasets"][0]["id"] = "renamed"
    renamed["datasets"][1]["extends"] = "renamed"
    renamed["runs"][0]["datasets"] = "*"
    assert resolve_experiment(renamed).datasets["renamed"].hash == a.datasets["base"].hash


def test_seed_resolution_and_explicit_seed_override():
    exp = resolve_experiment(_cfg())
    run = next(r for r in exp.runs if r.seed == 1)
    assert run.net.init_seed == run.net.data_order_seed == 1
    cfg = _cfg()
    cfg["networks"][0]["spec"]["init_seed"] = 7
    exp = resolve_experiment(cfg)
    run = next(r for r in exp.runs if r.seed == 1 and r.network_id == "mlp")
    assert run.net.init_seed == 7 and run.net.data_order_seed == 1


@pytest.mark.parametrize(
    "mutate, fragment",
    [
        (lambda c: c["datasets"][0]["spec"].update(D=3), "radicals"),
        (lambda c: c["datasets"][0]["spec"].update(typo_field=1), "typo_field"),
        (lambda c: c["networks"][0]["spec"].update(activation="swish"), "activation"),
        (lambda c: c["runs"][0].update(datasets=["nope"]), "unknown dataset"),
        (lambda c: c["datasets"][1].update(extends="ghost"), "unknown id"),
        (lambda c: c["datasets"].append({"id": "base", "spec": {}}), "duplicate"),
        (lambda c: c.update(surprise=1), "surprise"),
        (lambda c: c["runs"][0].update(seeds=[]), "seeds"),
    ],
)
def test_invalid_configs_fail_before_any_compute(mutate, fragment):
    cfg = _cfg()
    mutate(cfg)
    with pytest.raises(ConfigError, match=fragment):
        resolve_experiment(cfg)


def test_extends_cycle_detected():
    cfg = _cfg()
    cfg["datasets"][0]["extends"] = "zipf"
    with pytest.raises(ConfigError, match="cycle"):
        resolve_experiment(cfg)


def test_duplicate_configurations_rejected():
    cfg = _cfg()
    cfg["networks"].append({"id": "mlp_again", "spec": {"hidden": [8]}})
    with pytest.raises(ConfigError, match="same content hash"):
        resolve_experiment(cfg)


# ------------------------------------------------------------------- end-to-end
@pytest.fixture
def experiment(tmp_path):
    return resolve_experiment(_cfg(), output_dir=str(tmp_path))


def test_local_run_produces_all_artifacts_and_report(experiment):
    run_local(experiment)
    layout = Layout.of(experiment)
    assert (layout.exp_dir / "config.json").exists()
    assert (layout.exp_dir / "resolved_config.json").exists()
    assert (layout.exp_dir / "report" / "report.md").exists()
    assert len(list((layout.root / "datasets").iterdir())) == 2

    for run in experiment.runs:
        run_dir = layout.training_dir(run.run_hash)
        for name in (
            "spec.json", "manifest.json", "events.jsonl", "history.jsonl",
            "model.pt", "metrics.json", "predictions.npz",
        ):  # fmt: skip
            assert (run_dir / name).exists(), (run.label, name)
        manifest = read_json(run_dir / "manifest.json")
        assert manifest["status"] == "ok"
        assert manifest["dataset_hash"] == run.dataset_hash
        assert "git_sha" in manifest["environment"]
        events = read_jsonl(run_dir / "events.jsonl")
        assert {"train_start", "checkpoint", "train_done", "evaluate_done"} <= {
            e["event"] for e in events
        }
        assert all(e["run_hash"] == run.run_hash for e in events)  # traceable lines

    runs = pd.read_csv(layout.exp_dir / "runs.csv")
    assert (runs["status"] == "ok").all() and len(runs) == 8
    aggregate = pd.read_csv(layout.exp_dir / "aggregate.csv")
    assert len(aggregate) == 4 and (aggregate["n_ok"] == 2).all()
    assert "wug_regular_prob_mean" in aggregate and "tokens_to_acc_90_mean" in aggregate


def test_rerun_reuses_everything(experiment):
    run_local(experiment)
    layout = Layout.of(experiment)
    before = (layout.training_dir(experiment.runs[0].run_hash) / "model.pt").stat().st_mtime_ns
    run_local(experiment)
    after = (layout.training_dir(experiment.runs[0].run_hash) / "model.pt").stat().st_mtime_ns
    assert before == after
    events = read_jsonl(layout.events_path)
    assert sum(e["event"] == "train_reused" for e in events) == 8
    assert sum(e["event"] == "dataset_reused" for e in events) == 2


def test_evaluation_matches_last_training_checkpoint(experiment):
    run_local(experiment)
    layout = Layout.of(experiment)
    run = experiment.runs[0]
    history = read_jsonl(layout.training_dir(run.run_hash) / "history.jsonl")
    metrics = read_json(layout.training_dir(run.run_hash) / "metrics.json")["metrics"]
    assert metrics["loss"] == pytest.approx(history[-1]["loss"], abs=1e-6)
    events = read_jsonl(layout.training_dir(run.run_hash) / "events.jsonl")
    assert not any(e["event"] == "evaluate_mismatch" for e in events)


def test_diverging_run_is_recorded_not_fatal(tmp_path):
    cfg = _cfg()
    cfg["networks"] = [
        {"id": "ok", "spec": {"hidden": [8]}},
        {"id": "boom", "spec": {"hidden": [8], "optimizer": "sgd", "lr": 1e12, "max_tokens": 5000}},
    ]
    cfg["runs"] = [{"datasets": ["base"], "networks": "*", "seeds": [0]}]
    exp = resolve_experiment(cfg, output_dir=str(tmp_path))
    run_local(exp)
    runs = collect_runs(exp).set_index("network_id")
    assert runs.loc["ok", "status"] == "ok"
    assert runs.loc["boom", "status"] == "diverged"
    report = (Layout.of(exp).exp_dir / "report" / "report.md").read_text()
    assert "diverged" in report


def test_existing_dataset_path_can_be_reused(tmp_path):
    first = resolve_experiment(_cfg(), output_dir=str(tmp_path / "a"))
    run_local(first)
    directory = tmp_path / "a" / "datasets" / first.datasets["base"].hash
    cfg = _cfg(
        datasets=[{"id": "existing", "path": str(directory)}],
        runs=[{"datasets": ["existing"], "networks": ["mlp"], "seeds": [0]}],
    )
    second = resolve_experiment(cfg, output_dir=str(tmp_path / "b"))
    assert second.datasets["existing"].hash == first.datasets["base"].hash
    run_local(second)
    assert (collect_runs(second)["status"] == "ok").all()
    assert not (tmp_path / "b" / "datasets").exists()  # nothing regenerated


def test_cli_validate_and_bad_config(tmp_path, capsys):
    good = tmp_path / "good.json"
    good.write_text(json.dumps(_cfg()))
    assert cli_main(["validate", str(good), "--output-dir", str(tmp_path / "out")]) == 0
    assert "8 runs" in capsys.readouterr().out
    assert not (tmp_path / "out").exists()  # validate writes nothing

    bad = tmp_path / "bad.json"
    broken = _cfg()
    broken["datasets"][0]["spec"]["D"] = 2
    bad.write_text(json.dumps(broken))
    assert cli_main(["validate", str(bad)]) == 2


def test_cached_hashes_are_not_carried_over_by_with_seed():
    from src.specs import NetworkSpec

    base = NetworkSpec(hidden=(8,))
    assert base.hash == base.hash  # cached, stable
    seeded = base.with_seed(3)
    assert seeded.hash != base.hash  # rebuilt, not a stale copy
    assert seeded.with_seed(99).init_seed == 3  # explicit seeds are never overwritten
    assert base.with_seed(3).hash == seeded.hash


def test_tasks_receive_a_light_context_not_the_whole_experiment():
    import pickle

    exp = resolve_experiment(_cfg())
    assert len(pickle.dumps(exp.context)) < len(pickle.dumps(exp)) / 3
    assert exp.context.config_hash == exp.config_hash
