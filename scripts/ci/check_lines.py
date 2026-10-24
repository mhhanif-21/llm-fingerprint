"""Verifikasi batas panjang seluruh kode Python melalui wc -l."""

import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    """Menolak file Python proyek dengan lebih dari 350 baris fisik."""
    paths = sorted(
        p
        for directory in ["llmname", "llm_eda", "tests", "scripts"]
        for p in (ROOT / directory).rglob("*.py")
    )
    paths += [ROOT / "run_eda.py"]
    result = subprocess.run(
        ["wc", "-l", *map(str, paths)], text=True, capture_output=True, check=True
    )
    rows = result.stdout.splitlines()[:-1]
    failed = [line for line in rows if int(line.split()[0]) > 350]
    if failed:
        raise SystemExit("Batas 350 baris terlampaui:\n" + "\n".join(failed))
    print(
        f"Batas baris: {len(paths)} file lulus; maksimum {max(int(row.split()[0]) for row in rows)}"
    )


if __name__ == "__main__":
    main()
