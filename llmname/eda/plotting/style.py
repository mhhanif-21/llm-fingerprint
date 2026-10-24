"""Pembuatan grafik EDA dengan gaya visual yang konsisten."""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from .base import _save_figure

matplotlib.use("Agg")


LOGGER = logging.getLogger("llm_eda")


def plot_punctuation_heatmap(punctuation_means: pd.DataFrame, path: Path) -> None:
    """Membuat heatmap profil tanda baca per model."""
    features = [
        "em_dash_rate",
        "curly_apostrophe_rate",
        "curly_quote_rate",
        "colon_rate",
        "semicolon_rate",
        "comma_rate",
        "question_rate",
        "exclamation_rate",
    ]
    features = [feature for feature in features if feature in punctuation_means.columns]
    raw = punctuation_means[features].copy()
    zscores = (raw - raw.mean(axis=0)) / raw.std(axis=0, ddof=0)
    zscores = zscores.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        zscores,
        annot=raw.round(2),
        fmt="",
        cmap="RdBu_r",
        center=0,
        linewidths=0.5,
        linecolor="white",
        annot_kws={"fontsize": 7},
        cbar_kws={"label": "Z-score rata-rata"},
        ax=ax,
    )
    ax.set_title("Profil tanda baca per model")
    ax.set_xlabel("Frekuensi per 1.000 karakter")
    ax.set_ylabel("")
    ax.tick_params(axis="y", labelsize=8)
    plt.setp(ax.get_xticklabels(), rotation=40, ha="right")
    fig.tight_layout()
    _save_figure(fig, path)


def plot_stylometry_heatmap(zscores: pd.DataFrame, path: Path) -> None:
    """Membuat heatmap z-score fitur gaya per model."""
    fig, ax = plt.subplots(figsize=(14, 8))
    sns.heatmap(
        zscores,
        annot=True,
        fmt=".1f",
        cmap="RdBu_r",
        center=0,
        linewidths=0.5,
        linecolor="white",
        annot_kws={"fontsize": 7},
        cbar_kws={"label": "Z-score rata-rata"},
        ax=ax,
    )
    ax.set_title("Profil gaya per model (z-score antar model)")
    ax.set_xlabel("Fitur")
    ax.set_ylabel("")
    ax.tick_params(axis="y", labelsize=8)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    fig.tight_layout()
    _save_figure(fig, path)


def plot_feature_correlation(correlation: pd.DataFrame, path: Path) -> None:
    """Membuat heatmap korelasi antar fitur terpilih."""
    data = correlation.fillna(0.0)
    mask = np.triu(np.ones_like(data, dtype=bool))

    fig, ax = plt.subplots(figsize=(12, 9))
    sns.heatmap(
        data,
        mask=mask,
        cmap="vlag",
        center=0,
        square=True,
        linewidths=0.5,
        linecolor="white",
        cbar_kws={"label": "Korelasi Spearman", "shrink": 0.8},
        ax=ax,
    )
    ax.set_title("Korelasi Spearman antar fitur gaya terpilih")
    ax.tick_params(axis="x", labelsize=8)
    ax.tick_params(axis="y", labelsize=8)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    fig.tight_layout()
    _save_figure(fig, path)


def plot_prompt_variability(prompt_stats: pd.DataFrame, path: Path) -> None:
    """Membuat scatter variasi panjang teks antarrespons per prompt."""
    data = prompt_stats.dropna(subset=["mean_chars", "std_chars"]).copy()
    if data.empty:
        LOGGER.warning("Tidak ada data variasi prompt untuk digrafikkan.")
        return

    fig, ax = plt.subplots(figsize=(10, 7))
    scatter = ax.scatter(
        data["mean_chars"],
        data["std_chars"],
        c=data["mean_em_dash"],
        cmap="viridis",
        s=34,
        alpha=0.85,
        edgecolor="white",
        linewidth=0.4,
    )
    colorbar = fig.colorbar(scatter, ax=ax)
    colorbar.set_label("Rata-rata em dash per 1.000 karakter")

    top_variability = data.nlargest(5, "std_chars")
    for _, row in top_variability.iterrows():
        ax.annotate(
            str(row["prompt_id"]),
            (row["mean_chars"], row["std_chars"]),
            xytext=(5, 4),
            textcoords="offset points",
            fontsize=8,
            color="#333333",
        )

    ax.set_xlabel("Rata-rata panjang teks (karakter)")
    ax.set_ylabel("Simpangan baku panjang teks (karakter)")
    ax.set_title("Variasi panjang teks antarrespons dalam satu prompt")
    fig.tight_layout()
    _save_figure(fig, path)


def plot_variance_explained(variance: pd.DataFrame, path: Path) -> None:
    """Membuat grafik R-squared model versus prompt."""
    data = variance.set_index("feature")[["model_r2", "prompt_r2", "both_r2"]] * 100.0
    fig, ax = plt.subplots(figsize=(10, 6))
    data.plot(
        kind="barh",
        ax=ax,
        color=["#4C72B0", "#DD8452", "#55A868"],
        width=0.72,
    )
    ax.set_xlabel("R² in-sample (%)")
    ax.set_ylabel("")
    ax.set_title("Proporsi variasi fitur yang dijelaskan model dan prompt")
    ax.legend(["Model", "Prompt", "Model + prompt"], loc="lower right")
    ax.tick_params(axis="y", labelsize=8)
    fig.tight_layout()
    _save_figure(fig, path)
