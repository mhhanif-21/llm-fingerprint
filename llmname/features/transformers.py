"""Transformer kompatibel scikit-learn dengan seleksi fitur eksplisit."""

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

from llmname.eda.features import FEATURE_COLUMNS, FEATURE_GROUPS, add_features

from .candidates import CANDIDATE_DESCRIPTIONS, candidate_frame


def select_style_columns(options: dict) -> list[str]:
    """Memilih kelompok atau fitur individual dalam urutan yang stabil."""
    allowed = {"groups", "include", "exclude", "candidates"}
    if set(options) - allowed:
        raise ValueError("Opsi style tidak dikenal")
    for value in options.values():
        if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
            raise ValueError("Pilihan fitur harus berupa daftar string")
    groups = options.get("groups", [])
    if set(groups) - set(FEATURE_GROUPS):
        raise ValueError("Kelompok style tidak dikenal")
    known = set(FEATURE_COLUMNS)
    if set(options.get("include", []) + options.get("exclude", [])) - known:
        raise ValueError("Fitur style tidak dikenal")
    candidates = options.get("candidates", [])
    if set(candidates) - set(CANDIDATE_DESCRIPTIONS):
        raise ValueError("Kandidat fitur tidak dikenal")
    columns = [column for group in groups for column in FEATURE_GROUPS[group]]
    columns += options.get("include", [])
    excluded = set(options.get("exclude", []))
    columns = list(dict.fromkeys(column for column in columns if column not in excluded))
    columns += list(dict.fromkeys(candidates))
    if not columns:
        raise ValueError("Kelompok style tidak memiliki fitur aktif")
    return columns


class TextSelector(TransformerMixin, BaseEstimator):
    """Mengambil teks saja sehingga metadata tidak menjadi fitur model."""

    def fit(self, X: pd.DataFrame, y=None):
        """Memeriksa keberadaan kolom teks tanpa mempelajari statistik global."""
        if "text" not in X:
            raise ValueError("Kolom text tidak tersedia")
        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        """Mengembalikan teks dengan tanda baca dan Unicode yang tetap utuh."""
        return X["text"].fillna("").astype(str).to_numpy()


class StyleTransformer(TransformerMixin, BaseEstimator):
    """Menghitung fitur gaya stateless berdasarkan pilihan konfigurasi."""

    def __init__(self, columns: tuple[str, ...]):
        """Menyimpan parameter tanpa mengubahnya agar estimator dapat di-clone."""
        self.columns = columns

    def fit(self, X: pd.DataFrame, y=None):
        """Menyimpan nama fitur setelah memeriksa kontrak masukan."""
        if not self.columns or "text" not in X:
            raise ValueError("Fitur style atau kolom text kosong")
        unknown = set(self.columns) - set(FEATURE_COLUMNS) - set(CANDIDATE_DESCRIPTIONS)
        if unknown:
            raise ValueError(f"Fitur tidak dikenal: {sorted(unknown)}")
        self.feature_names_out_ = np.asarray(self.columns, dtype=object)
        return self

    def transform(self, X: pd.DataFrame) -> csr_matrix:
        """Menghasilkan matriks sparse numerik tanpa menggunakan label."""
        check_is_fitted(self, "feature_names_out_")
        base = add_features(X[["text"]].reset_index(drop=True))
        candidates = [name for name in self.columns if name in CANDIDATE_DESCRIPTIONS]
        if candidates:
            extra = candidate_frame(base["text"].fillna("").astype(str))
            base[candidates] = extra[candidates]
        matrix = base[list(self.columns)].to_numpy(dtype=np.float64)
        if not np.isfinite(matrix).all():
            raise ValueError("Fitur mengandung nilai numerik tidak hingga")
        return csr_matrix(matrix)

    def get_feature_names_out(self, input_features=None) -> np.ndarray:
        """Mengembalikan nama fitur dalam urutan keluaran transformasi."""
        check_is_fitted(self, "feature_names_out_")
        return self.feature_names_out_.copy()
