"""Struktur data yang digunakan antar-modul pipeline EDA."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd


@dataclass
class DatasetBundle:
    """Kumpulan dataset yang dimuat dari folder data."""

    train: pd.DataFrame
    test: pd.DataFrame
    sample_submission: pd.DataFrame | None
    data_dir: Path
    sample_submission_available: bool
    load_warnings: list[str] = field(default_factory=list)


@dataclass
class DatasetAudit:
    """Ringkasan struktur, integritas, dan keseimbangan dataset."""

    summary: dict[str, Any]
    integrity: dict[str, Any]
    imbalance: dict[str, float]
    schema: pd.DataFrame
    class_distribution: pd.DataFrame
    prompt_coverage: pd.DataFrame
    missing_cells: pd.DataFrame
    rows_per_prompt: pd.DataFrame
    models_per_prompt: pd.DataFrame


@dataclass
class FeatureAnalysis:
    """Hasil analisis fitur gaya dan diskriminabilitasnya."""

    dictionary: pd.DataFrame
    discriminability: pd.DataFrame
    means_by_model: pd.DataFrame
    zscores_by_model: pd.DataFrame
    correlation: pd.DataFrame
    punctuation_means_by_model: pd.DataFrame
    variance_explained: pd.DataFrame


@dataclass
class ModelingResults:
    """Hasil perbandingan baseline dan analisis kesalahan."""

    comparison: pd.DataFrame
    best_config: str
    best_feature_set: str
    best_classifier: str
    best_balanced_accuracy: float
    best_accuracy: float
    oof_predictions: pd.DataFrame
    class_metrics: pd.DataFrame
    confusion_matrix: pd.DataFrame
    confusion_pairs: pd.DataFrame
    family_confusion: pd.DataFrame
    distinctive_terms: pd.DataFrame


@dataclass
class AnalysisArtifacts:
    """Seluruh artefak yang dihasilkan oleh pipeline EDA."""

    audit: DatasetAudit
    feature_table: pd.DataFrame
    feature_analysis: FeatureAnalysis
    text_statistics_overall: pd.DataFrame
    text_statistics_by_model: pd.DataFrame
    prompt_text_statistics: pd.DataFrame
    non_ascii_characters: pd.DataFrame
    top_words_overall: pd.DataFrame
    top_words_by_model: pd.DataFrame
    opening_phrases: pd.DataFrame
    near_duplicate_pairs: pd.DataFrame
    near_duplicate_summary: dict[str, Any]
    modeling: ModelingResults | None = None
    warnings: list[str] = field(default_factory=list)
