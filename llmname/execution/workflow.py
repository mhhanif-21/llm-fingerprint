"""Siklus package, upload, run, status, dan fetch versi Kaggle yang tepat."""

import re
import time
import tomllib
import uuid
from datetime import datetime, timezone
from pathlib import Path

from llmname.artifacts import read_json, write_json

from .client import kaggle_command
from .package import build_notebook, build_package, validate_handle, validate_result


def load_target(path: Path) -> dict:
    """Membaca target lokal dan menolak metadata Kaggle yang tidak lengkap."""
    with path.open("rb") as stream:
        target = tomllib.load(stream)
    if set(target) != {
        "kernel",
        "competition",
        "title",
        "poll_seconds",
        "timeout_seconds",
    }:
        raise ValueError("Konfigurasi target Kaggle tidak lengkap atau memiliki opsi asing")
    validate_handle(target["kernel"])
    if not re.fullmatch(r"[a-z0-9-]+", target["competition"]):
        raise ValueError("Slug kompetisi tidak valid")
    if target["poll_seconds"] < 5 or target["timeout_seconds"] <= target["poll_seconds"]:
        raise ValueError("Batas polling tidak valid")
    return target


def submit(root: Path, config: dict, target: dict, smoke: bool) -> Path:
    """Mengirim snapshot paket dan notebook baru tanpa membawa artefak lokal."""
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:8]
    directory = root / "artifacts" / "kaggle" / run_id
    directory.mkdir(parents=True)
    request = build_package(root, directory / "package", config, run_id, smoke)
    state = {"request": request, "target": target, "status": "packaged"}
    write_json(directory / "remote.json", state)
    log = directory / "kaggle.log"
    notebook_dir = directory / "notebook"
    notebook_dir.mkdir()
    build_notebook(
        notebook_dir / "llmname.ipynb", request, directory / "package/llmname_source.zip"
    )
    kernel = {
        "id": target["kernel"],
        "title": target["title"],
        "code_file": "llmname.ipynb",
        "language": "python",
        "kernel_type": "notebook",
        "is_private": True,
        "enable_gpu": False,
        "enable_internet": False,
        "dataset_sources": [],
        "competition_sources": [target["competition"]],
        "kernel_sources": [],
    }
    write_json(notebook_dir / "kernel-metadata.json", kernel)
    try:
        pushed = kaggle_command(["kernels", "push", "-p", str(notebook_dir)], log)
    except Exception as error:
        state.update({"status": "submission_failed", "error_type": type(error).__name__})
        write_json(directory / "remote.json", state)
        raise
    version = re.search(r"[Vv]ersion\s+(\d+)", pushed)
    if not version:
        raise RuntimeError("Upload tidak mengembalikan versi notebook yang dapat diverifikasi")
    state.update({"kernel_version": int(version.group(1)), "status": "submitted"})
    write_json(directory / "remote.json", state)
    return directory


def status(directory: Path) -> str:
    """Membaca status terbaru tanpa mengubah run yang sudah diverifikasi."""
    state = read_json(directory / "remote.json")
    if state["status"] == "verified":
        return "VERIFIED"
    output = kaggle_command(
        ["kernels", "status", state["target"]["kernel"]], directory / "kaggle.log"
    )
    match = re.search(r"KernelWorkerStatus\.([A-Z_]+)", output)
    if not match:
        raise RuntimeError(f"Status Kaggle tidak dikenal: {output.strip()}")
    return match.group(1)


def fetch(directory: Path) -> dict:
    """Mengunduh versi spesifik dan memvalidasi seluruh manifest serta hash hasil."""
    state = read_json(directory / "remote.json")
    reference = f"{state['target']['kernel']}/{state['kernel_version']}"
    output_dir = directory / "output"
    kaggle_command(
        [
            "kernels",
            "output",
            reference,
            "-p",
            str(output_dir),
            "--force",
            "--file-pattern",
            "^" + re.escape(state["request"]["run_id"]) + "/",
        ],
        directory / "kaggle.log",
        timeout=300,
    )
    manifest = validate_result(output_dir, state["request"])
    state["status"] = "verified"
    write_json(directory / "remote.json", state)
    return manifest


def wait_and_fetch(directory: Path) -> dict:
    """Menunggu dengan batas waktu tanpa mengirim ulang training atau paket."""
    target = read_json(directory / "remote.json")["target"]
    deadline = time.monotonic() + target["timeout_seconds"]
    while time.monotonic() < deadline:
        current = status(directory)
        print(f"Kaggle status: {current}; run: {directory.name}", flush=True)
        if current in {"COMPLETE", "VERIFIED"}:
            return fetch(directory)
        if current not in {"RUNNING", "QUEUED", "PENDING"}:
            raise RuntimeError(f"Eksekusi Kaggle gagal: {current}; jalankan make diagnose")
        time.sleep(target["poll_seconds"])
    raise TimeoutError(
        "Batas waktu pemantauan tercapai; lanjutkan dengan kaggle-status/kaggle-fetch"
    )
