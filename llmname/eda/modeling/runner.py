"""Perbandingan baseline dan analisis kesalahan klasifikasi."""

from __future__ import annotations

import logging

import pandas as pd

from ..artifacts import ModelingResults
from ..config import EDAConfig
from ..features import infer_model_family
from .diagnostics import (
    build_class_metrics,
    build_confusion_pairs,
    build_distinctive_terms,
    build_family_confusion,
)
from .evaluation import build_model_comparison

LOGGER = logging.getLogger("llm_eda")

FEATURE_SETS = [
    "word_1_2",
    "char_wb_3_5",
    "word_char",
    "style_only",
    "word_char_style",
]


def run_modeling(frame: pd.DataFrame, config: EDAConfig) -> ModelingResults:
    """Menjalankan perbandingan baseline dan seluruh analisis kesalahan."""
    comparison, predictions_by_key, best_metadata = build_model_comparison(frame, config)
    best_key = (
        f"{best_metadata['feature_set']}__{best_metadata['classifier']}__"
        f"{best_metadata['cv_scheme']}"
    )
    best_predictions = predictions_by_key[best_key]

    labels = sorted(frame["model"].unique())
    class_metrics, matrix_frame = build_class_metrics(frame, best_predictions, labels)
    confusion_pairs = build_confusion_pairs(matrix_frame, labels)

    oof_predictions = frame[["id", "prompt_id", "model", "family"]].copy()
    oof_predictions["predicted_model"] = best_predictions
    oof_predictions["predicted_family"] = oof_predictions["predicted_model"].map(infer_model_family)
    oof_predictions["correct"] = oof_predictions["model"] == oof_predictions["predicted_model"]
    family_confusion = build_family_confusion(oof_predictions)

    distinctive_terms = build_distinctive_terms(frame, config.top_n, config.seed)

    LOGGER.info(
        "Baseline terbaik: %s + %s (%s), balanced accuracy %.4f.",
        best_metadata["feature_set"],
        best_metadata["classifier"],
        best_metadata["cv_scheme"],
        best_metadata["balanced_accuracy"],
    )
    if not confusion_pairs.empty:
        top = confusion_pairs.iloc[0]
        LOGGER.info(
            "Kesalahan terbanyak: %s -> %s (%d kasus).",
            top["true_model"],
            top["predicted_model"],
            int(top["count"]),
        )

    return ModelingResults(
        comparison=comparison,
        best_config=(
            f"{best_metadata['feature_set']} | {best_metadata['classifier']} | "
            f"{best_metadata['cv_scheme']}"
        ),
        best_feature_set=str(best_metadata["feature_set"]),
        best_classifier=str(best_metadata["classifier"]),
        best_balanced_accuracy=float(best_metadata["balanced_accuracy"]),
        best_accuracy=float(best_metadata["accuracy"]),
        oof_predictions=oof_predictions,
        class_metrics=class_metrics,
        confusion_matrix=matrix_frame,
        confusion_pairs=confusion_pairs,
        family_confusion=family_confusion,
        distinctive_terms=distinctive_terms,
    )
