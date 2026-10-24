"""Perbandingan baseline dan analisis kesalahan klasifikasi."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import ComplementNB
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import MaxAbsScaler

from ..features import FEATURE_COLUMNS

LOGGER = logging.getLogger("llm_eda")

FEATURE_SETS = [
    "word_1_2",
    "char_wb_3_5",
    "word_char",
    "style_only",
    "word_char_style",
]


class TextColumnSelector(BaseEstimator, TransformerMixin):
    """Mengambil kolom teks dari DataFrame."""

    def __init__(self, column: str = "text"):
        """Menyimpan parameter transformer sesuai kontrak estimator scikit-learn."""
        self.column = column

    def fit(self, X: pd.DataFrame, y: Any = None) -> TextColumnSelector:
        """Memeriksa kontrak masukan sebelum transformasi fitur."""
        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        """Mengubah masukan menjadi representasi fitur yang ditentukan."""
        return X[self.column].fillna("").astype(str).to_numpy()


class NumericColumnSelector(BaseEstimator, TransformerMixin):
    """Mengambil kolom numerik dari DataFrame."""

    def __init__(self, columns: list[str]):
        """Menyimpan parameter transformer sesuai kontrak estimator scikit-learn."""
        self.columns = columns

    def fit(self, X: pd.DataFrame, y: Any = None) -> NumericColumnSelector:
        """Memeriksa kontrak masukan sebelum transformasi fitur."""
        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        """Mengubah masukan menjadi representasi fitur yang ditentukan."""
        return X[list(self.columns)].fillna(0.0).to_numpy(dtype=float)


def _make_word_vectorizer() -> TfidfVectorizer:
    """Membuat vectorizer untuk pola satu dan dua kata."""
    return TfidfVectorizer(
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.90,
        sublinear_tf=True,
    )


def _make_char_vectorizer() -> TfidfVectorizer:
    """Membuat vectorizer karakter pada batas kata."""
    return TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(3, 5),
        min_df=2,
        max_features=60000,
        sublinear_tf=True,
    )


def _text_branch(vectorizer: TfidfVectorizer) -> Pipeline:
    """Menyusun pemilih teks dan vectorizer menjadi satu cabang fitur."""
    return Pipeline(
        [
            ("select_text", TextColumnSelector()),
            ("vectorize", vectorizer),
        ]
    )


def _style_branch() -> Pipeline:
    """Menyusun pemilih numerik dan scaler fitur gaya."""
    return Pipeline(
        [
            ("select_numeric", NumericColumnSelector(FEATURE_COLUMNS)),
            ("scale", MaxAbsScaler()),
        ]
    )


def build_feature_transformer(feature_set: str) -> Any:
    """Membangun transformer fitur sesuai nama konfigurasi."""
    if feature_set == "word_1_2":
        return _text_branch(_make_word_vectorizer())
    if feature_set == "char_wb_3_5":
        return _text_branch(_make_char_vectorizer())
    if feature_set == "word_char":
        return FeatureUnion(
            [
                ("word", _text_branch(_make_word_vectorizer())),
                ("char", _text_branch(_make_char_vectorizer())),
            ]
        )
    if feature_set == "style_only":
        return _style_branch()
    if feature_set == "word_char_style":
        return FeatureUnion(
            [
                ("word", _text_branch(_make_word_vectorizer())),
                ("char", _text_branch(_make_char_vectorizer())),
                ("style", _style_branch()),
            ]
        )
    raise ValueError(f"Feature set tidak dikenal: {feature_set}")


def build_classifier(classifier_name: str, seed: int):
    """Membuat estimator pembanding melalui factory model bersama."""
    from llmname.modeling.factory import build_model

    if classifier_name == "complement_nb":
        return ComplementNB(alpha=1.0)
    params = {"C": 5.0, "max_iter": 3000}
    if classifier_name == "linear_svc":
        params["max_iter"] = 10000
    return build_model({"name": classifier_name, "params": params}, seed)


def build_pipeline(feature_set: str, classifier_name: str, seed: int) -> Pipeline:
    """Membangun pipeline lengkap untuk satu konfigurasi."""
    return Pipeline(
        [
            ("features", build_feature_transformer(feature_set)),
            ("classifier", build_classifier(classifier_name, seed)),
        ]
    )
