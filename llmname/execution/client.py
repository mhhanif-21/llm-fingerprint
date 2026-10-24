"""Akses CLI Kaggle dengan batas waktu dan log tersensor."""

import os
import re
import shutil
import subprocess
from pathlib import Path


def redact(text: str) -> str:
    """Menyensor pola token, header autentikasi, dan kredensial pada URL."""
    text = re.sub(r"(?i)(bearer\s+)\S+", r"\1[REDACTED]", text)
    text = re.sub(
        r"(?i)((?:token|password|api[_-]?key|secret)\s*[=:]\s*)[^\s,;]+", r"\1[REDACTED]", text
    )
    return re.sub(r"(https?://)[^/@\s]+:[^/@\s]+@", r"\1[REDACTED]@", text)


def kaggle_command(arguments: list[str], log_path: Path, timeout: int = 180) -> str:
    """Menjalankan Kaggle CLI yang tersedia tanpa membaca isi kredensial."""
    executable = os.environ.get("KAGGLE_CLI") or shutil.which("kaggle")
    if not executable:
        raise RuntimeError("Kaggle CLI tidak tersedia; jalankan make setup-kaggle")
    try:
        result = subprocess.run(
            [executable, *arguments], capture_output=True, text=True, timeout=timeout, check=False
        )
    except subprocess.TimeoutExpired as error:
        raise RuntimeError(f"Kaggle CLI melewati batas {timeout} detik") from error
    output = redact(result.stdout + result.stderr)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a") as stream:
        stream.write(f"\nCommand: kaggle {' '.join(arguments)}\n{output}\n")
    if result.returncode:
        raise RuntimeError(f"Kaggle CLI gagal ({result.returncode}): {output[-1600:]}")
    return output
