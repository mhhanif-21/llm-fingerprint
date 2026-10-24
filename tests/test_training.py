"""Pemeriksaan manifest run sukses maupun kegagalan pembacaan data."""

from pathlib import Path

import pytest

from llmname.artifacts import read_json
from llmname.modeling.training import train_run


def test_smoke_manifest_records_real_artifacts(trained_run):
    """Memastikan run kecil memiliki metrik, fold, versi, dan hash model nyata."""
    manifest = read_json(trained_run / "manifest.json")
    assert manifest["status"] == "complete"
    assert manifest["mode"] == "smoke"
    assert manifest["train_rows"] > 0
    assert "model.joblib" in manifest["files"]
    assert manifest["seconds"] > 0


def test_missing_data_records_failed_run(config, tmp_path):
    """Mencatat status gagal ketika dataset tidak tersedia tanpa hasil palsu."""
    root = Path(__file__).resolve().parents[1]
    with pytest.raises(FileNotFoundError):
        train_run(
            config,
            tmp_path / "absent",
            root,
            smoke=True,
            run_id="missing-data",
            output_dir=tmp_path,
        )
    manifest = read_json(tmp_path / "missing-data/manifest.json")
    assert manifest["status"] == "failed"
    assert manifest["error_type"] == "FileNotFoundError"
