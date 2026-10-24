"""Fingerprint dan manifest artefak eksperimen yang dapat diverifikasi."""

import hashlib
import importlib.metadata
import json
import platform
from pathlib import Path


def file_hash(path: Path) -> str:
    """Menghitung SHA-256 secara bertahap tanpa memuat seluruh file ke memori."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def json_hash(payload: dict) -> str:
    """Menghasilkan fingerprint JSON kanonik untuk konfigurasi."""
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def source_files(root: Path) -> list[Path]:
    """Memilih kode dan metadata paket tanpa data, kredensial, atau artefak."""
    paths = [root / "pyproject.toml"]
    for package in ["llmname", "llm_eda"]:
        paths += [
            p
            for p in (root / package).rglob("*")
            if p.suffix in {".py", ".html"} and "__pycache__" not in p.parts
        ]
    return sorted(paths)


def code_hash(root: Path) -> str:
    """Menggabungkan nama relatif dan isi seluruh berkas kode paket."""
    digest = hashlib.sha256()
    for path in source_files(root):
        digest.update(path.relative_to(root).as_posix().encode())
        digest.update(bytes.fromhex(file_hash(path)))
    return digest.hexdigest()


def write_json(path: Path, payload: dict) -> None:
    """Menulis JSON valid tanpa nilai NaN dengan penggantian file atomik."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    temporary.replace(path)


def read_json(path: Path) -> dict:
    """Membaca objek JSON dari artefak yang tersedia."""
    return json.loads(path.read_text())


def collect_versions() -> dict:
    """Mencatat versi runtime dan dependensi yang memengaruhi hasil eksperimen."""
    packages = ["numpy", "pandas", "scikit-learn", "scipy", "joblib", "pyarrow"]
    return {
        "python": platform.python_version(),
        **{name: importlib.metadata.version(name) for name in packages},
    }


def verify_files(directory: Path, hashes: dict[str, str]) -> None:
    """Memeriksa integritas file dan menolak path di luar direktori hasil."""
    for name, expected in hashes.items():
        path = (directory / name).resolve()
        if not path.is_relative_to(directory.resolve()) or not path.is_file():
            raise ValueError(f"Artefak tidak tersedia atau path tidak valid: {name}")
        if file_hash(path) != expected:
            raise ValueError(f"Fingerprint artefak tidak cocok: {name}")
