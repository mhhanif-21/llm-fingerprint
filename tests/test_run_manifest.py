"""Pemeriksaan integritas manifest dan fingerprint konfigurasi."""

import pytest

from llmname.artifacts import file_hash, json_hash, verify_files


def test_hashes_detect_modified_artifact(tmp_path, config):
    """Mendeteksi perubahan file dan parameter tanpa memerlukan hasil buatan."""
    from pathlib import Path

    original = Path("configs/baseline_svc.toml").read_bytes()
    path = tmp_path / "config.toml"
    path.write_bytes(original)
    hashes = {path.name: file_hash(path)}
    verify_files(tmp_path, hashes)
    path.write_bytes(original + b"\n")
    with pytest.raises(ValueError, match="Fingerprint"):
        verify_files(tmp_path, hashes)
    first = json_hash(config)
    config["model"]["params"]["C"] = 2
    assert first != json_hash(config)


def test_path_traversal_is_rejected(tmp_path):
    """Menolak path manifest yang keluar dari direktori hasil."""
    with pytest.raises(ValueError, match="path"):
        verify_files(tmp_path, {"../pyproject.toml": ""})
