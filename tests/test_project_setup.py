"""Pemeriksaan kontrak setup dan perlindungan file privat."""

import tomllib
from pathlib import Path

import pathspec
import pytest

from llmname.config import validate_config

ROOT = Path(__file__).resolve().parents[1]


def test_dependencies_and_private_paths():
    """Memastikan dependensi inti dan seluruh jalur privat diabaikan Git."""
    project = tomllib.loads((ROOT / "pyproject.toml").read_text())
    assert any(name.startswith("scikit-learn") for name in project["project"]["dependencies"])
    spec = pathspec.PathSpec.from_lines("gitignore", (ROOT / ".gitignore").read_text().splitlines())
    for name in [
        "docs/a.md",
        "AGENTS.md",
        ".agents/a",
        "artifacts/x",
        "name-that-llm/train.csv",
        "configs/kaggle.local.toml",
        ".env",
        "console-guide.md",
    ]:
        assert spec.match_file(name)
    assert not spec.match_file("llmname/config.py")


def test_config_rejects_unknown_options(config):
    """Menolak parameter salah eja dan fold yang tidak masuk akal."""
    config["experiment"]["n_splits"] = 1
    with pytest.raises(ValueError):
        validate_config(config)
    config["experiment"]["n_splits"] = 5
    config["model"]["typo"] = True
    with pytest.raises(ValueError):
        validate_config(config)
