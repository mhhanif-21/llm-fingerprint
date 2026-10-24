"""Smoke E2E pada API Kaggle nyata dengan verifikasi versi dan hasil."""

import os
from pathlib import Path

import pytest

from llmname.artifacts import read_json
from llmname.execution.package import validate_result
from llmname.execution.workflow import load_target, submit, wait_and_fetch

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.e2e
@pytest.mark.kaggle
def test_kaggle_real_run_and_matching_manifest(config):
    """Mengirim smoke run atau memverifikasi ulang run nyata yang sudah diunduh."""
    if os.environ.get("LLMNAME_KAGGLE_E2E") != "1":
        pytest.skip("Aktifkan melalui make test-e2e TARGET=kaggle")
    existing = os.environ.get("LLMNAME_KAGGLE_RUN_DIR")
    if existing:
        directory = Path(existing)
        state = read_json(directory / "remote.json")
        manifest = validate_result(directory / "output", state["request"])
    else:
        target = load_target(ROOT / "configs/kaggle.local.toml")
        directory = submit(ROOT, config, target, smoke=True)
        print(f"Kaggle run directory: {directory}", flush=True)
        manifest = wait_and_fetch(directory)
    assert manifest["mode"] == "smoke"
    assert manifest["n_splits"] == 2
    assert manifest["status"] == "complete"
    assert read_json(directory / "remote.json")["kernel_version"] > 0
