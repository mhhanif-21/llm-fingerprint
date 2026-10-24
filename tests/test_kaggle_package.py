"""Audit arsip kode Kaggle tanpa akses jaringan atau mock API."""

import zipfile
from pathlib import Path

import pytest

from llmname.execution.package import build_notebook, build_package, validate_handle

ROOT = Path(__file__).resolve().parents[1]


def test_package_contains_only_code(config, tmp_path):
    """Memastikan arsip hanya membawa sumber yang diperlukan dan fingerprint valid."""
    request = build_package(ROOT, tmp_path / "package", config, "package-check", True)
    with zipfile.ZipFile(tmp_path / "package/llmname_source.zip") as archive:
        names = archive.namelist()
    assert "llmname/modeling/training.py" in names
    assert not any(
        any(part in name for part in ["docs/", ".env", "train.csv", "artifacts/", "AGENTS"])
        for name in names
    )
    build_notebook(tmp_path / "run.ipynb", request, tmp_path / "package/llmname_source.zip")
    import json

    notebook = json.loads((tmp_path / "run.ipynb").read_text())
    assert len(notebook["cells"]) == 3
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            compile("".join(cell["source"]), "notebook", "exec")


def test_target_handle_validation():
    """Menolak target kosong atau path yang bukan referensi Kaggle."""
    with pytest.raises(ValueError):
        validate_handle("../private")
    with pytest.raises(ValueError):
        validate_handle("")


def test_notebook_archive_is_deterministic(config, tmp_path):
    """Membandingkan snapshot yang isinya sama dan memeriksa bootstrap aktual."""
    import json
    import shutil
    import sys

    from llmname.artifacts import code_hash

    first = build_package(ROOT, tmp_path / "first", config, "first", True)
    second = build_package(ROOT, tmp_path / "second", config, "second", True)
    assert first["archive_hash"] == second["archive_hash"]
    build_notebook(tmp_path / "run.ipynb", first, tmp_path / "first/llmname_source.zip")
    notebook = json.loads((tmp_path / "run.ipynb").read_text())
    namespace = {}
    original_paths = sys.path.copy()
    try:
        exec("".join(notebook["cells"][1]["source"]), namespace)
        assert code_hash(namespace["source_dir"]) == first["code_hash"]
    finally:
        sys.path[:] = original_paths
        if "source_dir" in namespace:
            shutil.rmtree(namespace["source_dir"])
