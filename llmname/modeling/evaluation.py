"""Evaluasi fold, OOF, dan laporan kesalahan berdasarkan prediksi nyata."""

import time

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score

from llmname.eda.modeling.diagnostics import build_class_metrics, build_confusion_pairs

from .pipeline import build_pipeline


def compute_metrics(labels, predictions) -> dict[str, float]:
    """Menghitung metrik klasifikasi tanpa mengasumsikan jumlah kelas tertentu."""
    return {
        "accuracy": float(accuracy_score(labels, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, predictions)),
        "macro_f1": float(f1_score(labels, predictions, average="macro", zero_division=0)),
    }


def evaluate_folds(frame: pd.DataFrame, folds: pd.DataFrame, config: dict, output_dir, cache_dir):
    """Menjalankan estimator baru untuk setiap fold dan menyimpan prediksi OOF."""
    predictions = np.full(len(frame), "", dtype=object)
    fold_metrics = []
    for fold in sorted(folds["fold"].unique()):
        valid_mask = folds["fold"].eq(fold).to_numpy()
        training = frame.loc[~valid_mask]
        validation = frame.loc[valid_mask]
        start = time.perf_counter()
        estimator = build_pipeline(config, cache_dir)
        estimator.fit(training[["text"]], training["model"])
        predicted = estimator.predict(validation[["text"]])
        predictions[valid_mask] = predicted
        metrics = compute_metrics(validation["model"], predicted)
        fold_metrics.append(
            {
                "fold": int(fold),
                "train_rows": len(training),
                "valid_rows": len(validation),
                "seconds": time.perf_counter() - start,
                **metrics,
            }
        )
        print(f"Fold {fold}: balanced_accuracy={metrics['balanced_accuracy']:.4f}", flush=True)
    if any(value == "" for value in predictions):
        raise ValueError("Prediksi OOF tidak lengkap")
    oof = folds.copy()
    oof["predicted_model"] = predictions
    oof["correct"] = oof["model"].eq(oof["predicted_model"])
    oof["n_chars"] = frame["text"].str.len().to_numpy()
    oof.to_csv(output_dir / "oof.csv", index=False)
    pd.DataFrame(fold_metrics).to_csv(output_dir / "fold_metrics.csv", index=False)
    labels = sorted(frame["model"].unique())
    per_class, confusion = build_class_metrics(frame, predictions, labels)
    per_class.to_csv(output_dir / "class_metrics.csv", index=False)
    confusion.to_csv(output_dir / "confusion_matrix.csv")
    build_confusion_pairs(confusion, labels).to_csv(output_dir / "confusion_pairs.csv", index=False)
    oof.assign(length_bin=pd.qcut(oof["n_chars"], 4, duplicates="drop")).groupby(
        "length_bin", observed=True
    )["correct"].agg(["mean", "count"]).to_csv(output_dir / "errors_by_length.csv")
    oof.groupby("prompt_id")["correct"].agg(["mean", "count"]).to_csv(
        output_dir / "errors_by_prompt.csv"
    )
    return compute_metrics(frame["model"], predictions), fold_metrics
