"""Pemeriksaan batas kepercayaan hasil remote dan sensor log."""

import pytest

from llmname.artifacts import read_json
from llmname.execution.client import redact
from llmname.execution.package import validate_result


def test_missing_remote_output_does_not_report_success(tmp_path):
    """Menolak hasil yang belum tersedia alih-alih menganggap run berhasil."""
    with pytest.raises(FileNotFoundError):
        validate_result(tmp_path, {"run_id": "missing-run"})


def test_log_redaction_removes_credential_patterns():
    """Menyensor pola kredensial tanpa memakai nilai secret sungguhan."""
    value = "Bearer TEST_VALUE token=TEST_VALUE https://user:TEST_VALUE@example.com"
    assert "TEST_VALUE" not in redact(value)


def test_wrong_code_or_dataset_fingerprint_is_rejected(trained_run):
    """Menolak manifest nyata ketika request mengharapkan kode atau data berbeda."""
    manifest = read_json(trained_run / "manifest.json")
    request = {key: manifest[key] for key in ["run_id", "code_hash", "config_hash", "mode"]}
    validate_result(trained_run.parent, request)
    request["code_hash"] = "incorrect"
    with pytest.raises(ValueError, match="code_hash"):
        validate_result(trained_run.parent, request)
    request["code_hash"] = manifest["code_hash"]
    request["expected_train_hash"] = "incorrect"
    with pytest.raises(ValueError, match="Dataset"):
        validate_result(trained_run.parent, request)
