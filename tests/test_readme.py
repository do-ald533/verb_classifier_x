"""Every experiment JSON shown in the README must resolve, so the docs cannot
drift away from the schema."""

import json
import re
from pathlib import Path

import pytest

from src.config_loader import resolve_experiment

README = Path(__file__).resolve().parent.parent / "README.md"
BLOCKS = re.findall(r"```json\n(.*?)\n```", README.read_text(), flags=re.DOTALL)
EXPERIMENTS = [json.loads(b) for b in BLOCKS if '"experiment"' in b]


def test_readme_contains_experiment_examples():
    assert len(EXPERIMENTS) >= 6


@pytest.mark.parametrize("config", EXPERIMENTS, ids=[c["experiment"] for c in EXPERIMENTS])
def test_readme_experiment_resolves(config, tmp_path):
    exp = resolve_experiment(config, output_dir=str(tmp_path))
    assert exp.runs


def test_example_file_matches_readme_smoke_example():
    shown = next(c for c in EXPERIMENTS if c["experiment"] == "smoke")
    on_disk = json.loads((README.parent / "experiments" / "example_smoke.json").read_text())
    assert shown == on_disk
