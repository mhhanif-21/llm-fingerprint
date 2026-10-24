"""Pengemasan kode minimum dengan inventaris dan notebook pemanggil tipis."""

import base64
import json
import re
import zipfile
from pathlib import Path

from llmname.artifacts import code_hash, file_hash, json_hash, read_json, source_files, write_json
from llmname.config import validate_config


def validate_handle(handle: str) -> str:
    """Memastikan target Kaggle berbentuk pemilik dan slug yang eksplisit."""
    if not re.fullmatch(r"[A-Za-z0-9_-]+/[A-Za-z0-9_-]+", handle):
        raise ValueError("Target Kaggle harus berbentuk owner/slug")
    return handle


def build_package(root: Path, destination: Path, config: dict, run_id: str, smoke: bool) -> dict:
    """Membuat arsip kode yang tidak mengandung data maupun secret."""
    validate_config(config)
    destination.mkdir(parents=True, exist_ok=False)
    archive = destination / "llmname_source.zip"
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as stream:
        for path in source_files(root):
            if path.is_symlink():
                raise ValueError("Symlink tidak diperbolehkan dalam paket")
            info = zipfile.ZipInfo(path.relative_to(root).as_posix())
            info.compress_type = zipfile.ZIP_DEFLATED
            stream.writestr(info, path.read_bytes())
    request = {
        "run_id": run_id,
        "code_hash": code_hash(root),
        "archive_hash": file_hash(archive),
        "config": config,
        "config_hash": json_hash(config),
        "mode": "smoke" if smoke else "full",
        "expected_train_hash": file_hash(root / "name-that-llm/train.csv")
        if (root / "name-that-llm/train.csv").is_file()
        else None,
    }
    write_json(destination / "request.json", request)
    return request


def validate_result(directory: Path, request: dict) -> dict:
    """Memastikan hasil remote berasal dari run, kode, konfigurasi, dan mode yang diminta."""
    from llmname.artifacts import verify_files

    manifest = read_json(directory / request["run_id"] / "manifest.json")
    for field in ["run_id", "code_hash", "config_hash", "mode"]:
        if manifest.get(field) != request[field]:
            raise ValueError(f"Hasil Kaggle tidak cocok pada {field}")
    if manifest.get("status") != "complete":
        raise ValueError("Run remote belum berhasil")
    if (
        request.get("expected_train_hash")
        and manifest.get("train_file_hash") != request["expected_train_hash"]
    ):
        raise ValueError("Dataset training remote berbeda dari data lokal")
    verify_files(directory / request["run_id"], manifest["files"])
    return manifest


def build_notebook(path: Path, request: dict, archive_path: Path) -> None:
    """Membuat notebook tanpa duplikasi logika fitur, evaluasi, dan training."""
    archive_base64 = base64.b64encode(archive_path.read_bytes()).decode("ascii")
    if file_hash(archive_path) != request["archive_hash"]:
        raise ValueError("Fingerprint arsip berubah sebelum notebook dibuat")
    first = """import base64
import hashlib
import io
import json
from pathlib import Path
import sys
import tempfile
import zipfile

archive_bytes = base64.b64decode(ARCHIVE_BASE64, validate=True)
if hashlib.sha256(archive_bytes).hexdigest() != ARCHIVE_HASH:
    raise ValueError("Fingerprint arsip kode tidak cocok")
source_dir = Path(tempfile.mkdtemp(prefix="llmname-source-"))
with zipfile.ZipFile(io.BytesIO(archive_bytes)) as package:
    for name in package.namelist():
        if not (source_dir / name).resolve().is_relative_to(source_dir.resolve()):
            raise ValueError("Path arsip tidak valid")
    package.extractall(source_dir)
sys.path.insert(0, str(source_dir))
from llmname.execution.remote import run_remote
"""
    first = (
        f"ARCHIVE_BASE64 = {archive_base64!r}\nARCHIVE_HASH = {request['archive_hash']!r}\n" + first
    )
    second = "request = json.loads(" + repr(json.dumps(request)) + ")\n"
    second += "run_remote(request, source_dir)\n"
    cells = [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "# llmname\n",
                "Eksekusi paket kode yang fingerprint-nya telah ditentukan.\n",
            ],
        }
    ]
    for source in [first, second]:
        cells.append(
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": source.splitlines(keepends=True),
            }
        )
    write_json(
        path,
        {
            "nbformat": 4,
            "nbformat_minor": 5,
            "cells": cells,
            "metadata": {
                "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}
            },
        },
    )
