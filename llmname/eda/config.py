"""Konfigurasi dan resolusi lokasi data untuk pipeline EDA."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

DEFAULT_OUTPUT_DIR = Path("reports/eda")
DATA_DIR_CANDIDATES = (
    Path("/kaggle/input/competitions/name-that-llm"),
    Path("/kaggle/input/name-that-llm"),
)


@dataclass(frozen=True)
class EDAConfig:
    """Parameter yang mengatur seluruh proses EDA."""

    data_dir: Path
    output_dir: Path = DEFAULT_OUTPUT_DIR
    seed: int = 42
    n_splits: int = 5
    top_n: int = 15
    near_duplicate_threshold: float = 0.90
    skip_modeling: bool = False
    skip_near_duplicates: bool = False
    log_level: str = "INFO"

    @property
    def tables_dir(self) -> Path:
        """Mengembalikan direktori tabel hasil analisis."""
        return self.output_dir / "tables"

    @property
    def figures_dir(self) -> Path:
        """Mengembalikan direktori gambar hasil analisis."""
        return self.output_dir / "figures"

    @property
    def predictions_dir(self) -> Path:
        """Mengembalikan direktori prediksi hasil evaluasi."""
        return self.output_dir / "predictions"

    @property
    def log_file(self) -> Path:
        """Mengembalikan jalur berkas log eksekusi."""
        return self.output_dir / "run.log"

    def prepare_directories(self) -> None:
        """Membuat seluruh direktori output yang diperlukan."""
        for path in (
            self.output_dir,
            self.tables_dir,
            self.figures_dir,
            self.predictions_dir,
        ):
            path.mkdir(parents=True, exist_ok=True)


def _candidate_data_dirs() -> list[Path]:
    """Mengurutkan kandidat direktori data tanpa jalur yang berulang."""
    project_root = Path(__file__).resolve().parents[1]
    candidates = [
        *DATA_DIR_CANDIDATES,
        Path.cwd() / "name-that-llm",
        project_root / "name-that-llm",
        Path.cwd(),
    ]

    unique: list[Path] = []
    seen: set[Path] = set()
    for candidate in candidates:
        resolved = candidate.expanduser().resolve()
        if resolved not in seen:
            seen.add(resolved)
            unique.append(resolved)
    return unique


def resolve_data_dir(explicit: Path | None = None) -> Path:
    """Menentukan folder data yang valid."""
    if explicit is not None:
        path = explicit.expanduser().resolve()
        if (path / "train.csv").is_file():
            return path
        raise FileNotFoundError(f"Folder data tidak valid: {path} (train.csv tidak ditemukan).")

    for path in _candidate_data_dirs():
        if (path / "train.csv").is_file():
            return path

    candidates = "\n".join(f"  - {path}" for path in _candidate_data_dirs())
    raise FileNotFoundError(
        "Folder data tidak ditemukan. Gunakan --data-dir atau siapkan salah satu "
        f"lokasi berikut:\n{candidates}"
    )
