"""Perbandingan baseline dan analisis kesalahan klasifikasi."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix

from ..features import infer_model_family

LOGGER = logging.getLogger("llm_eda")

FEATURE_SETS = [
    "word_1_2",
    "char_wb_3_5",
    "word_char",
    "style_only",
    "word_char_style",
]


def build_class_metrics(
    frame: pd.DataFrame, predictions: np.ndarray, labels: list[str]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Membuat metrik per kelas dan confusion matrix."""
    y_true = frame["model"]
    report = classification_report(
        y_true,
        predictions,
        labels=labels,
        output_dict=True,
        zero_division=0,
    )
    metrics = (
        pd.DataFrame(report)
        .T.loc[labels, ["precision", "recall", "f1-score", "support"]]
        .reset_index()
        .rename(columns={"index": "model"})
    )
    metrics["model"] = metrics["model"].astype(str)
    metrics["support"] = metrics["support"].astype(int)
    metrics["family"] = metrics["model"].map(infer_model_family)

    matrix = confusion_matrix(y_true, predictions, labels=labels)
    matrix_frame = pd.DataFrame(matrix, index=labels, columns=labels)
    matrix_frame.index.name = "true_model"
    matrix_frame.columns.name = "predicted_model"

    confused_with: list[str] = []
    confused_counts: list[int] = []
    confused_rates: list[float] = []
    for label in metrics["model"]:
        row = matrix_frame.loc[label].drop(labels=[label])
        if row.empty or row.max() == 0:
            confused_with.append("")
            confused_counts.append(0)
            confused_rates.append(0.0)
        else:
            confused_with.append(str(row.idxmax()))
            confused_counts.append(int(row.max()))
            denominator = int(matrix_frame.loc[label].sum())
            confused_rates.append(float(row.max() / denominator) if denominator else 0.0)

    metrics["most_confused_with"] = confused_with
    metrics["most_confused_count"] = confused_counts
    metrics["most_confused_rate"] = confused_rates
    metrics = metrics.sort_values(["recall", "model"], ascending=[True, True]).reset_index(
        drop=True
    )
    return metrics, matrix_frame


def build_confusion_pairs(matrix_frame: pd.DataFrame, labels: list[str]) -> pd.DataFrame:
    """Mendata pasangan kelas yang paling sering tertukar."""
    rows: list[dict[str, Any]] = []
    for true_model in labels:
        row = matrix_frame.loc[true_model]
        total = int(row.sum())
        for predicted_model in labels:
            if true_model == predicted_model:
                continue
            count = int(row[predicted_model])
            if count == 0:
                continue
            rows.append(
                {
                    "true_model": true_model,
                    "predicted_model": predicted_model,
                    "count": count,
                    "rate_of_true_class": count / total if total else 0.0,
                    "same_family": (
                        infer_model_family(true_model) == infer_model_family(predicted_model)
                    ),
                }
            )
    result = pd.DataFrame(rows)
    if result.empty:
        return pd.DataFrame(
            columns=[
                "true_model",
                "predicted_model",
                "count",
                "rate_of_true_class",
                "same_family",
            ]
        )
    return result.sort_values(
        ["count", "true_model", "predicted_model"], ascending=[False, True, True]
    ).reset_index(drop=True)


def build_family_confusion(oof_predictions: pd.DataFrame) -> pd.DataFrame:
    """Membuat confusion matrix pada tingkat keluarga model."""
    true_families = oof_predictions["model"].map(infer_model_family)
    predicted_families = oof_predictions["predicted_model"].map(infer_model_family)
    families = sorted(set(true_families) | set(predicted_families))
    matrix = pd.crosstab(true_families, predicted_families)
    matrix = matrix.reindex(index=families, columns=families, fill_value=0)
    matrix.index.name = "true_family"
    matrix.columns.name = "predicted_family"
    return matrix


def build_distinctive_terms(frame: pd.DataFrame, top_n: int, seed: int) -> pd.DataFrame:
    """Mengambil istilah TF-IDF paling positif untuk setiap kelas."""
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.90,
        sublinear_tf=True,
        stop_words="english",
    )
    matrix = vectorizer.fit_transform(frame["text"].fillna("").astype(str))
    classifier = LogisticRegression(
        max_iter=3000,
        C=5,
        random_state=seed,
    )
    classifier.fit(matrix, frame["model"])
    terms = vectorizer.get_feature_names_out()

    rows: list[dict[str, Any]] = []
    for class_index, label in enumerate(classifier.classes_):
        coefficients = classifier.coef_[class_index]
        top_indices = np.argsort(coefficients)[-top_n:][::-1]
        for rank, index in enumerate(top_indices, start=1):
            rows.append(
                {
                    "model": str(label),
                    "family": infer_model_family(str(label)),
                    "rank": rank,
                    "term": str(terms[index]),
                    "coefficient": float(coefficients[index]),
                }
            )

    return pd.DataFrame(rows).sort_values(["model", "rank"]).reset_index(drop=True)
