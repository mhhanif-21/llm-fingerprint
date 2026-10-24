"""Perbandingan baseline dan analisis kesalahan klasifikasi."""

from __future__ import annotations

import logging
import time
import warnings
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.model_selection import GroupKFold, StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline

from ..config import EDAConfig
from .builders import build_pipeline

LOGGER = logging.getLogger("llm_eda")

FEATURE_SETS = [
    "word_1_2",
    "char_wb_3_5",
    "word_char",
    "style_only",
    "word_char_style",
]


def _evaluate_pipeline(
    pipeline: Pipeline,
    frame: pd.DataFrame,
    cv: Any,
    groups: pd.Series | None,
) -> tuple[np.ndarray, dict[str, float]]:
    """Mengevaluasi pipeline pada pembagian fold yang diberikan."""
    start = time.perf_counter()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        predictions = cross_val_predict(
            pipeline,
            frame,
            frame["model"],
            cv=cv,
            groups=groups,
            n_jobs=1,
        )
    elapsed = time.perf_counter() - start

    y_true = frame["model"]
    metrics = {
        "accuracy": float(accuracy_score(y_true, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, predictions)),
        "macro_f1": float(f1_score(y_true, predictions, average="macro")),
        "weighted_f1": float(f1_score(y_true, predictions, average="weighted")),
        "fit_seconds": float(elapsed),
    }
    return predictions, metrics


def _comparison_row(
    feature_set: str,
    classifier: str,
    cv_scheme: str,
    metrics: dict[str, float] | None = None,
    error: str = "",
) -> dict[str, Any]:
    """Menyusun baris perbandingan metrik atau alasan kegagalan estimator."""
    row: dict[str, Any] = {
        "feature_set": feature_set,
        "classifier": classifier,
        "cv_scheme": cv_scheme,
        "accuracy": np.nan,
        "balanced_accuracy": np.nan,
        "macro_f1": np.nan,
        "weighted_f1": np.nan,
        "fit_seconds": np.nan,
        "error": error,
    }
    if metrics is not None:
        row.update(metrics)
    return row


def build_model_comparison(
    frame: pd.DataFrame, config: EDAConfig
) -> tuple[pd.DataFrame, dict[str, np.ndarray], dict[str, Any]]:
    """Membandingkan feature set, classifier, dan skema validasi."""
    group_scheme = f"group_{config.n_splits}fold"
    random_scheme = f"random_{config.n_splits}fold"
    group_cv = GroupKFold(n_splits=config.n_splits)
    random_cv = StratifiedKFold(
        n_splits=config.n_splits,
        shuffle=True,
        random_state=config.seed,
    )

    rows: list[dict[str, Any]] = []
    predictions_by_key: dict[str, np.ndarray] = {}

    for feature_set in FEATURE_SETS:
        key = f"{feature_set}__logistic_regression__{group_scheme}"
        LOGGER.info(
            "Evaluasi baseline: %s + logistic regression (%s).",
            feature_set,
            group_scheme,
        )
        try:
            pipeline = build_pipeline(feature_set, "logistic_regression", config.seed)
            predictions, metrics = _evaluate_pipeline(pipeline, frame, group_cv, frame["prompt_id"])
            rows.append(_comparison_row(feature_set, "logistic_regression", group_scheme, metrics))
            predictions_by_key[key] = predictions
        except Exception as exc:
            LOGGER.exception("Evaluasi %s gagal.", key)
            rows.append(
                _comparison_row(
                    feature_set,
                    "logistic_regression",
                    group_scheme,
                    error=str(exc),
                )
            )

    comparison = pd.DataFrame(rows)
    candidates = comparison[comparison["cv_scheme"].eq(group_scheme) & comparison["error"].eq("")]
    if candidates.empty:
        raise RuntimeError("Seluruh konfigurasi baseline gagal dievaluasi.")

    best_row = candidates.sort_values(["balanced_accuracy", "macro_f1"], ascending=False).iloc[0]
    best_feature_set = str(best_row["feature_set"])

    for classifier in ("linear_svc", "complement_nb"):
        key = f"{best_feature_set}__{classifier}__{group_scheme}"
        LOGGER.info("Evaluasi classifier tambahan: %s (%s).", classifier, best_feature_set)
        try:
            pipeline = build_pipeline(best_feature_set, classifier, config.seed)
            predictions, metrics = _evaluate_pipeline(pipeline, frame, group_cv, frame["prompt_id"])
            rows.append(_comparison_row(best_feature_set, classifier, group_scheme, metrics))
            predictions_by_key[key] = predictions
        except Exception as exc:
            LOGGER.exception("Evaluasi %s gagal.", key)
            rows.append(
                _comparison_row(
                    best_feature_set,
                    classifier,
                    group_scheme,
                    error=str(exc),
                )
            )

    for feature_set in ("word_1_2", "word_char"):
        key = f"{feature_set}__logistic_regression__{random_scheme}"
        LOGGER.info("Evaluasi pembanding: %s + random split.", feature_set)
        try:
            pipeline = build_pipeline(feature_set, "logistic_regression", config.seed)
            predictions, metrics = _evaluate_pipeline(pipeline, frame, random_cv, None)
            rows.append(_comparison_row(feature_set, "logistic_regression", random_scheme, metrics))
            predictions_by_key[key] = predictions
        except Exception as exc:
            LOGGER.exception("Evaluasi %s gagal.", key)
            rows.append(
                _comparison_row(
                    feature_set,
                    "logistic_regression",
                    random_scheme,
                    error=str(exc),
                )
            )

    comparison = pd.DataFrame(rows).sort_values(
        ["cv_scheme", "balanced_accuracy"], ascending=[True, False]
    )
    comparison = comparison.reset_index(drop=True)

    best_mask = (
        comparison["feature_set"].eq(best_feature_set)
        & comparison["cv_scheme"].eq(group_scheme)
        & comparison["error"].eq("")
    )
    best_candidates = comparison[best_mask].sort_values(
        ["balanced_accuracy", "macro_f1"], ascending=False
    )
    best_row = best_candidates.iloc[0]
    best_classifier = str(best_row["classifier"])
    best_key = f"{best_feature_set}__{best_classifier}__{group_scheme}"

    if best_key not in predictions_by_key:
        raise RuntimeError("Prediksi out-of-fold untuk konfigurasi terbaik tidak tersedia.")

    best_metadata = {
        "feature_set": best_feature_set,
        "classifier": best_classifier,
        "cv_scheme": group_scheme,
        "balanced_accuracy": float(best_row["balanced_accuracy"]),
        "accuracy": float(best_row["accuracy"]),
    }
    return comparison, predictions_by_key, best_metadata
