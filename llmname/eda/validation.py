"""Validasi skema dan integritas dasar dataset."""

from __future__ import annotations

import logging
from collections.abc import Iterable

import pandas as pd

from .artifacts import DatasetBundle

LOGGER = logging.getLogger("llm_eda")

TRAIN_REQUIRED_COLUMNS = ("id", "prompt_id", "text", "model")
TEST_REQUIRED_COLUMNS = ("id", "text")
SUBMISSION_REQUIRED_COLUMNS = ("id", "model")


class DatasetValidationError(ValueError):
    """Kesalahan validasi dataset."""


def _require_columns(frame: pd.DataFrame, required: Iterable[str], dataset_name: str) -> None:
    """Memastikan kolom wajib tersedia pada dataset yang diperiksa."""
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise DatasetValidationError(
            f"Kolom wajib pada {dataset_name} tidak ditemukan: {', '.join(missing)}."
        )


def validate_datasets(bundle: DatasetBundle) -> None:
    """Memastikan dataset memiliki struktur minimum untuk EDA."""
    train, test = bundle.train, bundle.test
    _require_columns(train, TRAIN_REQUIRED_COLUMNS, "train.csv")
    _require_columns(test, TEST_REQUIRED_COLUMNS, "test.csv")

    if train.empty:
        raise DatasetValidationError("train.csv tidak memiliki baris.")
    if test.empty:
        raise DatasetValidationError("test.csv tidak memiliki baris.")
    if train["model"].nunique() < 2:
        raise DatasetValidationError("train.csv memerlukan minimal dua kelas model.")

    for column in TRAIN_REQUIRED_COLUMNS:
        null_count = int(train[column].isna().sum())
        if null_count:
            LOGGER.warning("Kolom %s pada train.csv memiliki %d nilai kosong.", column, null_count)

    if bundle.sample_submission is not None:
        sample = bundle.sample_submission
        _require_columns(sample, SUBMISSION_REQUIRED_COLUMNS, "sample_submission.parquet")
        missing_ids = set(test["id"]) - set(sample["id"])
        if missing_ids:
            LOGGER.warning(
                "sample_submission.parquet tidak memuat %d id dari test.csv.",
                len(missing_ids),
            )
    else:
        LOGGER.warning("sample_submission.parquet tidak tersedia; validasi submission dilewati.")

    LOGGER.info("Validasi skema selesai: train=%s, test=%s.", train.shape, test.shape)
