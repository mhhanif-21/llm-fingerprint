"""Fungsi baca/tulis dataset dan artefak EDA."""

from __future__ import annotations

import json
import logging
import math
from datetime import date, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .artifacts import DatasetBundle
from .config import EDAConfig

LOGGER = logging.getLogger("llm_eda")


def _read_csv(path: Path) -> pd.DataFrame:
    """Membaca CSV dengan pelaporan kesalahan yang menyertakan nama berkas."""
    if not path.is_file():
        raise FileNotFoundError(f"Berkas tidak ditemukan: {path}")
    frame = pd.read_csv(path, dtype=str)
    LOGGER.info("Memuat %s: %d baris, %d kolom.", path.name, *frame.shape)
    return frame


def _read_sample_submission(path: Path) -> tuple[pd.DataFrame | None, bool, list[str]]:
    """Membaca sample submission dan mencatat ketersediaan dependensi parquet."""
    warnings: list[str] = []
    if not path.is_file():
        warnings.append(f"Berkas {path.name} tidak ditemukan.")
        return None, False, warnings

    try:
        frame = pd.read_parquet(path)
    except Exception as exc:  # noqa: BLE001 - engine parquet dapat gagal dengan tipe berbeda.
        # Engine parquet tidak tersedia atau berkas rusak.
        warnings.append(
            "sample_submission.parquet tidak dapat dibaca "
            f"({type(exc).__name__}); validasi submission dilewati."
        )
        LOGGER.warning(warnings[-1])
        return None, False, warnings

    frame = frame.astype(str)
    LOGGER.info("Memuat %s: %d baris, %d kolom.", path.name, *frame.shape)
    return frame, True, warnings


def load_datasets(config: EDAConfig) -> DatasetBundle:
    """Memuat train, test, dan sample submission dari folder data."""
    data_dir = config.data_dir
    train = _read_csv(data_dir / "train.csv")
    test = _read_csv(data_dir / "test.csv")
    sample_submission, available, warnings = _read_sample_submission(
        data_dir / "sample_submission.parquet"
    )

    return DatasetBundle(
        train=train,
        test=test,
        sample_submission=sample_submission,
        data_dir=data_dir,
        sample_submission_available=available,
        load_warnings=warnings,
    )


def to_jsonable(value: Any) -> Any:
    """Mengubah nilai numpy/pandas menjadi tipe yang aman untuk JSON."""
    if isinstance(value, dict):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return [to_jsonable(item) for item in value.tolist()]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if np.isnan(value) else float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, float) and math.isnan(value):
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    return value


def write_dataframe(frame: pd.DataFrame, path: Path, index: bool = False) -> None:
    """Menyimpan DataFrame sebagai CSV UTF-8."""
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=index, encoding="utf-8")
    LOGGER.debug("Tabel disimpan: %s", path)


def write_json(payload: Any, path: Path) -> None:
    """Menyimpan payload JSON dengan tipe data yang sudah dinormalisasi."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(to_jsonable(payload), handle, ensure_ascii=False, indent=2)
    LOGGER.debug("JSON disimpan: %s", path)


def compact_text(value: str, max_chars: int = 220) -> str:
    """Merapikan dan memotong teks untuk keperluan tampilan."""
    text = " ".join(str(value).split())
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 1].rstrip() + "…"
