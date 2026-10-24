"""Pengujian inti untuk fitur, statistik, dan validasi dataset."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from llm_eda.artifacts import DatasetBundle
from llm_eda.features import (
    FEATURE_COLUMNS,
    add_features,
    build_feature_dictionary,
    infer_model_family,
)
from llm_eda.statistics import benjamini_hochberg, effect_size_label, eta_squared
from llm_eda.validation import DatasetValidationError, validate_datasets


@pytest.fixture()
def sample_frame(real_frame) -> pd.DataFrame:
    """Mengambil empat respons nyata dari fixture bersama."""
    return real_frame.head(4).copy()


def test_add_features_creates_all_feature_columns(sample_frame: pd.DataFrame) -> None:
    """Memverifikasi kelengkapan kolom fitur pada respons nyata."""
    enriched = add_features(sample_frame)

    assert set(FEATURE_COLUMNS).issubset(enriched.columns)
    assert enriched["n_chars"].tolist() == sample_frame["text"].str.len().tolist()
    assert enriched["em_dash_rate"].ge(0).all()
    assert enriched["family"].ne("").all()


def test_feature_dictionary_is_complete(sample_frame: pd.DataFrame) -> None:
    """Memverifikasi kelengkapan deskripsi dan kelompok fitur."""
    dictionary = build_feature_dictionary(add_features(sample_frame))

    assert len(dictionary) == len(FEATURE_COLUMNS)
    assert dictionary["description"].ne("").all()
    assert set(dictionary["group"]) == {
        "length",
        "case",
        "lexical",
        "punctuation",
        "format",
        "unicode",
    }


def test_infer_model_family() -> None:
    """Memverifikasi pemetaan nama model ke keluarga yang dikenal."""
    assert infer_model_family("gpt-6-astra") == "gpt"
    assert infer_model_family("claude-opus-5") == "claude"
    assert infer_model_family("unknown-model") == "other"


def test_eta_squared_detects_clear_separation() -> None:
    """Memverifikasi nilai eta kuadrat pada pemisahan matematis sempurna."""
    frame = pd.DataFrame(
        {
            "model": ["a", "a", "a", "b", "b", "b"],
            "value": [1.0, 1.0, 1.0, 10.0, 10.0, 10.0],
        }
    )

    assert eta_squared(frame, "value", "model") == pytest.approx(1.0)


def test_benjamini_hochberg_is_bounded_and_monotonic() -> None:
    """Memverifikasi batas dan monotonisitas koreksi pengujian berganda."""
    adjusted = benjamini_hochberg([0.01, 0.04, 0.03, 0.20])

    assert ((adjusted >= 0.0) & (adjusted <= 1.0)).all()
    ordered = adjusted[[0, 2, 1, 3]]
    assert (ordered[1:] >= ordered[:-1]).all()


def test_effect_size_label() -> None:
    """Memverifikasi pemetaan ukuran efek ke kategori interpretasi."""
    assert effect_size_label(0.005) == "dapat diabaikan"
    assert effect_size_label(0.03) == "kecil"
    assert effect_size_label(0.10) == "sedang"
    assert effect_size_label(0.20) == "besar"


def test_validate_datasets_accepts_valid_bundle(sample_frame: pd.DataFrame, data_dir) -> None:
    """Memverifikasi penerimaan bundle dataset asli dengan skema valid."""
    bundle = DatasetBundle(
        train=sample_frame,
        test=pd.read_csv(data_dir / "test.csv"),
        sample_submission=pd.read_parquet(data_dir / "sample_submission.parquet"),
        data_dir=Path("."),
        sample_submission_available=True,
    )

    validate_datasets(bundle)


def test_validate_datasets_rejects_missing_column(sample_frame: pd.DataFrame, data_dir) -> None:
    """Memverifikasi penolakan bundle yang kehilangan kolom wajib."""
    bundle = DatasetBundle(
        train=sample_frame.drop(columns=["text"]),
        test=pd.read_csv(data_dir / "test.csv"),
        sample_submission=None,
        data_dir=Path("."),
        sample_submission_available=False,
    )

    with pytest.raises(DatasetValidationError):
        validate_datasets(bundle)
