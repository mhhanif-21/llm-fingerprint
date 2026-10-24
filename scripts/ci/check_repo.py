"""Audit ignore dan metadata Git tanpa mengubah indeks atau konfigurasi global."""

import subprocess
from pathlib import Path

import pathspec

ROOT = Path(__file__).resolve().parents[2]
PROTECTED = [
    "docs/testing.md",
    "AGENTS.md",
    "AGENT.md",
    ".agents/rules.md",
    ".env",
    "configs/kaggle.local.toml",
    "name-that-llm/train.csv",
    "artifacts/run/model.joblib",
    "reports/eda/summary.json",
    "console/tool.py",
    "console-guide.md",
]


def ignored_paths() -> pathspec.PathSpec:
    """Membaca pola Gitignore menggunakan semantik gitwildmatch."""
    return pathspec.PathSpec.from_lines("gitignore", (ROOT / ".gitignore").read_text().splitlines())


def main() -> None:
    """Memastikan jalur privat diabaikan dan tidak dilacak bila Git tersedia."""
    spec = ignored_paths()
    missing = [path for path in PROTECTED if not spec.match_file(path)]
    if missing:
        raise SystemExit(f"Proteksi Git tidak lengkap: {missing}")
    result = subprocess.run(["git", "ls-files", "-z"], cwd=ROOT, capture_output=True, text=True)
    if result.returncode:
        print("Ignore: lulus; metadata .git tidak aktif/dapat ditulis pada lingkungan ini.")
        return
    tracked = [p for p in result.stdout.split("\0") if p and spec.match_file(p)]
    if tracked:
        raise SystemExit(f"File privat/generated sudah dilacak: {tracked}")
    print("Proteksi dan indeks Git: lulus")


if __name__ == "__main__":
    main()
