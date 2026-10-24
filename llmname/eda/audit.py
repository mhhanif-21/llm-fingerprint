"""Audit struktur dataset, distribusi kelas, dan cakupan prompt."""

from __future__ import annotations

import logging

import pandas as pd

from .artifacts import DatasetAudit, DatasetBundle
from .features import infer_model_family
from .io import compact_text
from .statistics import imbalance_metrics, wilson_interval

LOGGER = logging.getLogger("llm_eda")


def build_schema_table(bundle: DatasetBundle) -> pd.DataFrame:
    """Membuat tabel skema untuk train, test, dan sample submission."""
    datasets = [
        ("train", bundle.train),
        ("test", bundle.test),
        ("sample_submission", bundle.sample_submission),
    ]
    rows: list[dict[str, object]] = []

    for dataset_name, frame in datasets:
        if frame is None:
            continue
        for column in frame.columns:
            series = frame[column]
            first_value = series.dropna().iloc[0] if series.notna().any() else ""
            rows.append(
                {
                    "dataset": dataset_name,
                    "column": column,
                    "dtype": str(series.dtype),
                    "non_null": int(series.notna().sum()),
                    "missing": int(series.isna().sum()),
                    "unique": int(series.nunique(dropna=True)),
                    "example": compact_text(first_value, max_chars=120),
                }
            )
    return pd.DataFrame(rows)


def build_class_distribution(train: pd.DataFrame) -> pd.DataFrame:
    """Menghitung distribusi kelas beserta interval kepercayaan Wilson."""
    counts = train["model"].value_counts()
    total = len(train)
    expected = total / len(counts)
    rows: list[dict[str, object]] = []

    for model, count in counts.items():
        low, high = wilson_interval(int(count), total)
        rows.append(
            {
                "model": model,
                "family": infer_model_family(model),
                "count": int(count),
                "percentage": count / total * 100.0,
                "ci_low": low * 100.0,
                "ci_high": high * 100.0,
                "deviation_from_expected": count - expected,
            }
        )

    frame = pd.DataFrame(rows).sort_values(["count", "model"], ascending=[False, True])
    return frame.reset_index(drop=True)


def build_prompt_tables(
    train: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Membuat pivot cakupan prompt dan tabel ringkasan struktur prompt."""
    model_columns = sorted(train["model"].unique())
    coverage = (
        train.pivot_table(
            index="prompt_id",
            columns="model",
            values="id",
            aggfunc="count",
            fill_value=0,
        )
        .reindex(columns=model_columns, fill_value=0)
        .sort_index()
    )
    coverage.index.name = "prompt_id"
    coverage.columns.name = None

    rows_per_prompt = (
        train.groupby("prompt_id")
        .size()
        .value_counts()
        .sort_index()
        .rename_axis("responses_per_prompt")
        .reset_index(name="prompt_count")
    )

    models_per_prompt = (
        train.groupby("prompt_id")["model"]
        .nunique()
        .value_counts()
        .sort_index()
        .rename_axis("models_per_prompt")
        .reset_index(name="prompt_count")
    )

    missing_rows: list[dict[str, object]] = []
    for prompt_id, row in coverage.iterrows():
        for model, count in row.items():
            if int(count) == 0:
                missing_rows.append(
                    {
                        "prompt_id": prompt_id,
                        "model": model,
                        "family": infer_model_family(model),
                        "reason": "Tidak ada respons untuk kombinasi prompt-model.",
                    }
                )
    missing_cells = pd.DataFrame(
        missing_rows,
        columns=["prompt_id", "model", "family", "reason"],
    )

    return coverage, missing_cells, rows_per_prompt, models_per_prompt


def build_dataset_audit(bundle: DatasetBundle) -> DatasetAudit:
    """Menyusun ringkasan menyeluruh tentang dataset."""
    train = bundle.train
    test = bundle.test
    sample = bundle.sample_submission

    class_counts = train["model"].value_counts()
    n_classes = int(train["model"].nunique())
    n_prompts = int(train["prompt_id"].nunique())
    expected_cells = n_prompts * n_classes

    summary = {
        "train_rows": len(train),
        "train_columns": list(train.columns),
        "test_rows": len(test),
        "test_columns": list(test.columns),
        "sample_submission_rows": len(sample) if sample is not None else None,
        "sample_submission_columns": list(sample.columns) if sample is not None else None,
        "n_classes": n_classes,
        "n_prompts": n_prompts,
        "expected_prompt_model_cells": int(expected_cells),
        "observed_prompt_model_cells": len(train),
        "prompt_model_coverage_pct": (
            len(train) / expected_cells * 100.0 if expected_cells else 0.0
        ),
        "test_id_overlap_with_train": len(set(test["id"]) & set(train["id"])),
        "test_text_overlap_with_train": len(
            set(test["text"].dropna()) & set(train["text"].dropna())
        ),
    }

    integrity = {
        "missing_values": {column: int(train[column].isna().sum()) for column in train.columns},
        "duplicate_train_ids": int(train["id"].duplicated().sum()),
        "duplicate_train_texts": int(train["text"].duplicated().sum()),
        "duplicate_prompt_model_pairs": int(train.duplicated(subset=["prompt_id", "model"]).sum()),
        "empty_train_texts": int(train["text"].fillna("").str.strip().eq("").sum()),
        "duplicate_test_ids": int(test["id"].duplicated().sum()),
        "duplicate_test_texts": int(test["text"].duplicated().sum()),
        "test_ids_missing_from_sample": (
            len(set(test["id"]) - set(sample["id"])) if sample is not None else None
        ),
    }

    imbalance = imbalance_metrics(class_counts)
    schema = build_schema_table(bundle)
    class_distribution = build_class_distribution(train)
    coverage, missing_cells, rows_per_prompt, models_per_prompt = build_prompt_tables(train)

    LOGGER.info(
        "Audit dataset: %d baris, %d kelas, %d prompt, cakupan %.2f%%.",
        len(train),
        n_classes,
        n_prompts,
        summary["prompt_model_coverage_pct"],
    )
    if not missing_cells.empty:
        LOGGER.info("Ditemukan %d sel prompt-model yang kosong.", len(missing_cells))

    return DatasetAudit(
        summary=summary,
        integrity=integrity,
        imbalance=imbalance,
        schema=schema,
        class_distribution=class_distribution,
        prompt_coverage=coverage,
        missing_cells=missing_cells,
        rows_per_prompt=rows_per_prompt,
        models_per_prompt=models_per_prompt,
    )
