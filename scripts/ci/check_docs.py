"""Pemeriksaan referensi lokal dan inventaris test pada dokumentasi privat."""

import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    """Memastikan dokumen wajib, tautan lokal, dan daftar test tetap sinkron."""
    required = [
        "project-navigation.md",
        "testing.md",
        "prd.md",
        "contracts.md",
        "execution.md",
        "research/eda-research.md",
        "plan/model-execution-guide.md",
        "plan/setup-pipeline.md",
    ]
    for name in required:
        if not (ROOT / "docs" / name).is_file():
            raise SystemExit(f"Dokumen belum tersedia: {name}")
    testing = (ROOT / "docs/testing.md").read_text()
    for test in (ROOT / "tests").rglob("test_*.py"):
        if test.relative_to(ROOT).as_posix() not in testing:
            raise SystemExit(f"Test belum didokumentasikan: {test}")
    for path in (ROOT / "docs").rglob("*.md"):
        for link in re.findall(r"\]\(([^)]+)\)", path.read_text()):
            if "://" not in link and not link.startswith("#"):
                if not (path.parent / link.split("#")[0]).exists():
                    raise SystemExit(f"Tautan lokal tidak tersedia: {path}: {link}")
    result = subprocess.run(
        ["make", "-n", "help", "eda", "test", "lint"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=True,
    )
    if not result.stdout:
        raise SystemExit("Perintah Make terdokumentasi tidak tersedia")
    print("Dokumentasi: lulus")


if __name__ == "__main__":
    main()
