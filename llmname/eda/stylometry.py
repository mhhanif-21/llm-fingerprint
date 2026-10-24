"""Analisis stylometri dan diskriminabilitas fitur antar model."""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

from .artifacts import FeatureAnalysis
from .config import EDAConfig
from .features import (
    FEATURE_COLUMNS,
    FEATURE_DESCRIPTIONS,
    FEATURE_GROUPS,
    build_feature_dictionary,
)
from .statistics import (
    adjusted_r_squared,
    benjamini_hochberg,
    effect_size_label,
    eta_squared,
    kruskal_test,
)

LOGGER = logging.getLogger("llm_eda")

VARIANCE_TARGET_FEATURES = [
    "n_chars",
    "n_words",
    "em_dash_rate",
    "avg_word_length",
    "type_token_ratio",
    "comma_rate",
]


def _feature_group(column: str) -> str:
    """Menentukan kelompok fitur berdasarkan katalog yang tersedia."""
    for group, columns in FEATURE_GROUPS.items():
        if column in columns:
            return group
    return "lainnya"


def build_feature_discriminability(frame: pd.DataFrame, feature_columns: list[str]) -> pd.DataFrame:
    """Menghitung eta-squared, Kruskal-Wallis, dan koreksi Benjamini-Hochberg."""
    rows: list[dict[str, object]] = []
    p_values: list[float] = []

    for feature in feature_columns:
        series = frame[feature]
        means = frame.groupby("model")[feature].mean()
        n_unique = int(series.nunique(dropna=True))

        if n_unique <= 1:
            eta_value = 0.0
            statistic = float("nan")
            p_value = 1.0
        else:
            eta_value = eta_squared(frame, feature, "model")
            statistic, p_value = kruskal_test(frame, feature, "model")

        min_model = str(means.idxmin()) if not means.empty else ""
        max_model = str(means.idxmax()) if not means.empty else ""
        min_value = float(means.min()) if not means.empty else 0.0
        max_value = float(means.max()) if not means.empty else 0.0

        rows.append(
            {
                "feature": feature,
                "group": _feature_group(feature),
                "description": FEATURE_DESCRIPTIONS.get(feature, ""),
                "eta_squared": float(eta_value),
                "kruskal_h": statistic,
                "p_value": float(p_value),
                "min_model": min_model,
                "min_value": min_value,
                "max_model": max_model,
                "max_value": max_value,
                "mean_range": max_value - min_value,
                "n_unique": n_unique,
            }
        )
        p_values.append(float(p_value))

    adjusted = benjamini_hochberg(p_values)
    for row, p_adjusted in zip(rows, adjusted, strict=True):
        row["p_adjusted"] = float(p_adjusted)
        row["significant_bh"] = bool(p_adjusted < 0.05)
        row["effect_size"] = effect_size_label(float(row["eta_squared"]))

    result = pd.DataFrame(rows).sort_values("eta_squared", ascending=False)
    return result.reset_index(drop=True)


def build_feature_means_by_model(frame: pd.DataFrame, feature_columns: list[str]) -> pd.DataFrame:
    """Menghitung rata-rata setiap fitur untuk setiap model."""
    means = frame.groupby("model")[feature_columns].mean()
    families = frame.groupby("model")["family"].first()
    result = pd.concat([families.rename("family"), means], axis=1).reset_index()
    return result


def build_feature_zscores(frame: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    """Menstandarkan rata-rata fitur antar model untuk keperluan heatmap."""
    means = frame.groupby("model")[features].mean()
    zscores = (means - means.mean(axis=0)) / means.std(axis=0, ddof=0)
    zscores = zscores.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    families = frame.groupby("model")["family"].first()
    ordered_models = sorted(
        zscores.index,
        key=lambda model: (str(families.get(model, "other")), model),
    )
    return zscores.loc[ordered_models]


def build_feature_correlation(
    frame: pd.DataFrame, features: list[str], method: str = "spearman"
) -> pd.DataFrame:
    """Menghitung matriks korelasi antar fitur terpilih."""
    return frame[features].corr(method=method)


def build_punctuation_means(frame: pd.DataFrame) -> pd.DataFrame:
    """Menghitung rata-rata fitur tanda baca per model."""
    punctuation_features = FEATURE_GROUPS["punctuation"]
    means = frame.groupby("model")[punctuation_features].mean()
    families = frame.groupby("model")["family"].first()
    result = pd.concat([families.rename("family"), means], axis=1)
    return result.sort_values("em_dash_rate", ascending=False)


def _grouped_r_squared(
    frame: pd.DataFrame, target: str, group_columns: list[str]
) -> tuple[float, float, int]:
    """Menghitung daya jelas fitur untuk satu atau beberapa faktor kategori."""
    values = frame[target].to_numpy(dtype=float)
    dummies = pd.get_dummies(
        frame[group_columns],
        drop_first=True,
        dtype=float,
    )
    if dummies.shape[1] == 0:
        return 0.0, 0.0, 0

    model = LinearRegression()
    model.fit(dummies, values)
    predictions = model.predict(dummies)
    r_squared = float(r2_score(values, predictions))
    adjusted = adjusted_r_squared(r_squared, len(values), dummies.shape[1])
    return r_squared, adjusted, int(dummies.shape[1])


def build_variance_explained(frame: pd.DataFrame) -> pd.DataFrame:
    """Membandingkan variasi yang dijelaskan model, prompt, dan keduanya."""
    rows: list[dict[str, object]] = []
    for target in VARIANCE_TARGET_FEATURES:
        model_r2, model_adj, model_params = _grouped_r_squared(frame, target, ["model"])
        prompt_r2, prompt_adj, prompt_params = _grouped_r_squared(frame, target, ["prompt_id"])
        both_r2, both_adj, _ = _grouped_r_squared(frame, target, ["model", "prompt_id"])
        rows.append(
            {
                "feature": target,
                "description": FEATURE_DESCRIPTIONS.get(target, ""),
                "n_observations": len(frame),
                "model_r2": model_r2,
                "model_adjusted_r2": model_adj,
                "model_parameters": model_params,
                "prompt_r2": prompt_r2,
                "prompt_adjusted_r2": prompt_adj,
                "prompt_parameters": prompt_params,
                "both_r2": both_r2,
                "both_adjusted_r2": both_adj,
            }
        )
    return pd.DataFrame(rows)


def build_stylometry_analysis(frame: pd.DataFrame, config: EDAConfig) -> FeatureAnalysis:
    """Menjalankan seluruh analisis stylometri."""
    dictionary = build_feature_dictionary(frame)
    discriminability = build_feature_discriminability(frame, FEATURE_COLUMNS)
    means = build_feature_means_by_model(frame, FEATURE_COLUMNS)

    top_features = discriminability.head(config.top_n)["feature"].tolist()
    zscores = build_feature_zscores(frame, top_features)
    correlation = build_feature_correlation(frame, top_features)
    punctuation = build_punctuation_means(frame)
    variance = build_variance_explained(frame)

    LOGGER.info("Fitur paling diskriminatif: %s", ", ".join(top_features[:5]))
    return FeatureAnalysis(
        dictionary=dictionary,
        discriminability=discriminability,
        means_by_model=means,
        zscores_by_model=zscores,
        correlation=correlation,
        punctuation_means_by_model=punctuation,
        variance_explained=variance,
    )
