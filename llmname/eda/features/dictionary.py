"""Komputasi fitur teks menggunakan definisi bersama."""

from __future__ import annotations

import pandas as pd

from .definitions import FEATURE_COLUMNS, FEATURE_DESCRIPTIONS, FEATURE_GROUPS


def build_feature_dictionary(frame: pd.DataFrame) -> pd.DataFrame:
    """Membuat tabel dokumentasi fitur beserta statistik dasar."""
    rows: list[dict[str, object]] = []
    for group, columns in FEATURE_GROUPS.items():
        for column in columns:
            series = frame[column]
            rows.append(
                {
                    "feature": column,
                    "group": group,
                    "description": FEATURE_DESCRIPTIONS.get(column, ""),
                    "non_null": int(series.notna().sum()),
                    "unique_values": int(series.nunique(dropna=True)),
                    "min": float(series.min()) if series.notna().any() else None,
                    "max": float(series.max()) if series.notna().any() else None,
                }
            )
    return pd.DataFrame(rows)


def numeric_feature_columns() -> list[str]:
    """Mengembalikan daftar fitur numerik yang digunakan untuk analisis."""
    return list(FEATURE_COLUMNS)
