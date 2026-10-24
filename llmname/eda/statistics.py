"""Fungsi statistik umum untuk analisis dataset dan fitur."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd
from scipy.stats import kruskal


def wilson_interval(successes: int, total: int, z: float = 1.96) -> tuple[float, float]:
    """Menghitung interval kepercayaan Wilson untuk sebuah proporsi."""
    if total <= 0:
        return 0.0, 1.0
    proportion = successes / total
    denominator = 1.0 + z**2 / total
    center = (proportion + z**2 / (2.0 * total)) / denominator
    margin = (
        z * np.sqrt(proportion * (1.0 - proportion) / total + z**2 / (4.0 * total**2)) / denominator
    )
    return max(0.0, center - margin), min(1.0, center + margin)


def shannon_entropy(counts: Iterable[float]) -> float:
    """Menghitung entropi Shannon (basis 2) dari distribusi kelas."""
    values = np.asarray(list(counts), dtype=float)
    total = values.sum()
    if total <= 0:
        return 0.0
    probabilities = values / total
    probabilities = probabilities[probabilities > 0]
    return float(-(probabilities * np.log2(probabilities)).sum())


def normalized_entropy(counts: Iterable[float]) -> float:
    """Menghitung entropi ternormalisasi terhadap jumlah kelas."""
    values = np.asarray(list(counts), dtype=float)
    if values.size <= 1:
        return 0.0
    maximum = np.log2(values.size)
    if maximum <= 0:
        return 0.0
    return shannon_entropy(values) / maximum


def imbalance_metrics(counts: pd.Series) -> dict[str, float]:
    """Merangkum tingkat ketidakseimbangan distribusi kelas."""
    values = counts.astype(float)
    if values.empty:
        return {
            "min_count": 0.0,
            "max_count": 0.0,
            "max_min_ratio": float("nan"),
            "coefficient_of_variation": float("nan"),
            "entropy": 0.0,
            "normalized_entropy": 0.0,
        }

    minimum = float(values.min())
    maximum = float(values.max())
    mean = float(values.mean())
    return {
        "min_count": minimum,
        "max_count": maximum,
        "max_min_ratio": maximum / minimum if minimum > 0 else float("nan"),
        "coefficient_of_variation": float(values.std(ddof=0) / mean) if mean > 0 else float("nan"),
        "entropy": shannon_entropy(values),
        "normalized_entropy": normalized_entropy(values),
    }


def eta_squared(frame: pd.DataFrame, feature: str, group_column: str) -> float:
    """Menghitung effect size eta-squared untuk fitur numerik."""
    subset = frame[[feature, group_column]].dropna()
    if subset.empty:
        return 0.0

    grand_mean = float(subset[feature].mean())
    ss_between = 0.0
    for _, group in subset.groupby(group_column, observed=True)[feature]:
        ss_between += len(group) * (float(group.mean()) - grand_mean) ** 2

    ss_total = float(((subset[feature] - grand_mean) ** 2).sum())
    if ss_total <= 0:
        return 0.0
    return float(ss_between / ss_total)


def kruskal_test(frame: pd.DataFrame, feature: str, group_column: str) -> tuple[float, float]:
    """Menjalankan uji Kruskal-Wallis untuk fitur numerik antar kelompok."""
    subset = frame[[feature, group_column]].dropna()
    groups = [
        group[feature].to_numpy(dtype=float)
        for _, group in subset.groupby(group_column, observed=True)
        if len(group) > 0
    ]
    if len(groups) < 2:
        return float("nan"), 1.0

    concatenated = np.concatenate(groups)
    if np.unique(concatenated).size < 2:
        return float("nan"), 1.0

    statistic, p_value = kruskal(*groups)
    return float(statistic), float(p_value)


def benjamini_hochberg(p_values: Iterable[float]) -> np.ndarray:
    """Mengoreksi p-value dengan prosedur Benjamini-Hochberg."""
    values = np.asarray(list(p_values), dtype=float)
    adjusted = np.ones_like(values)
    finite_mask = np.isfinite(values)
    if not finite_mask.any():
        return adjusted

    finite_values = values[finite_mask]
    order = np.argsort(finite_values)
    ranked = finite_values[order]
    count = len(ranked)
    scaled = ranked * count / np.arange(1, count + 1)
    monotonic = np.minimum.accumulate(scaled[::-1])[::-1]
    monotonic = np.clip(monotonic, 0.0, 1.0)

    adjusted_finite = np.empty_like(monotonic)
    adjusted_finite[order] = monotonic
    adjusted[finite_mask] = adjusted_finite
    return adjusted


def effect_size_label(eta_squared_value: float) -> str:
    """Memberi label interpretasi berdasarkan ambang Cohen."""
    if not np.isfinite(eta_squared_value):
        return "tidak tersedia"
    if eta_squared_value < 0.01:
        return "dapat diabaikan"
    if eta_squared_value < 0.06:
        return "kecil"
    if eta_squared_value < 0.14:
        return "sedang"
    return "besar"


def adjusted_r_squared(r_squared: float, n_observations: int, n_predictors: int) -> float:
    """Menghitung adjusted R-squared."""
    if not np.isfinite(r_squared):
        return float("nan")
    denominator = n_observations - n_predictors - 1
    if denominator <= 0:
        return float("nan")
    return float(1.0 - (1.0 - r_squared) * (n_observations - 1) / denominator)
