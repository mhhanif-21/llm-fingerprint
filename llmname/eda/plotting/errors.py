"""Pembuatan grafik EDA dengan gaya visual yang konsisten."""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from .base import FAMILY_COLORS, _family_legend, _save_figure

matplotlib.use("Agg")


LOGGER = logging.getLogger("llm_eda")


def plot_class_recall(class_metrics: pd.DataFrame, path: Path) -> None:
    """Membuat grafik recall per kelas."""
    data = class_metrics.sort_values("recall").reset_index(drop=True)
    colors = [FAMILY_COLORS.get(family, FAMILY_COLORS["other"]) for family in data["family"]]
    chance = 1.0 / max(len(data), 1)

    fig, ax = plt.subplots(figsize=(10, 7))
    bars = ax.barh(data["model"], data["recall"], color=colors, height=0.72)
    ax.axvline(
        chance,
        color="#333333",
        linestyle="--",
        linewidth=1.1,
        label=f"Chance: {chance:.3f}",
    )
    for bar, value in zip(bars, data["recall"], strict=True):
        ax.text(
            bar.get_width() + 0.01,
            bar.get_y() + bar.get_height() / 2,
            f"{value:.2f}",
            va="center",
            fontsize=8,
        )
    ax.set_xlim(0, max(1.0, data["recall"].max() * 1.12))
    ax.set_xlabel("Recall")
    ax.set_ylabel("")
    ax.set_title("Recall per kelas pada validasi GroupKFold")
    ax.tick_params(axis="y", labelsize=8)

    families = sorted(data["family"].unique())
    family_legend = ax.legend(
        handles=_family_legend(families),
        title="Keluarga model",
        loc="lower right",
    )
    ax.add_artist(family_legend)
    ax.legend(loc="upper right")
    fig.tight_layout()
    _save_figure(fig, path)


def plot_confusion_matrix(confusion: pd.DataFrame, path: Path) -> None:
    """Membuat heatmap confusion matrix ternormalisasi."""
    row_sums = confusion.sum(axis=1).replace(0, 1)
    normalized = confusion.div(row_sums, axis=0)
    upper = max(float(normalized.to_numpy().max()), 0.05)

    fig, ax = plt.subplots(figsize=(14, 12))
    sns.heatmap(
        normalized,
        annot=True,
        fmt=".2f",
        cmap="Blues",
        vmin=0,
        vmax=upper,
        linewidths=0.4,
        linecolor="white",
        annot_kws={"fontsize": 6.5},
        cbar_kws={"label": "Proporsi per kelas aktual"},
        ax=ax,
    )
    ax.set_title("Confusion matrix ternormalisasi (GroupKFold)")
    ax.set_xlabel("Prediksi")
    ax.set_ylabel("Aktual")
    ax.tick_params(axis="both", labelsize=8)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    plt.setp(ax.get_yticklabels(), rotation=0)
    fig.tight_layout()
    _save_figure(fig, path)


def plot_family_confusion(family_confusion: pd.DataFrame, path: Path) -> None:
    """Membuat heatmap confusion matrix antar keluarga model."""
    row_sums = family_confusion.sum(axis=1).replace(0, 1)
    normalized = family_confusion.div(row_sums, axis=0)

    annotations = np.empty(family_confusion.shape, dtype=object)
    for row_index, row_label in enumerate(family_confusion.index):
        for column_index, column_label in enumerate(family_confusion.columns):
            count = int(family_confusion.loc[row_label, column_label])
            rate = float(normalized.loc[row_label, column_label])
            annotations[row_index, column_index] = f"{count}\n({rate:.2f})"

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        normalized,
        annot=annotations,
        fmt="",
        cmap="Blues",
        vmin=0,
        vmax=max(float(normalized.to_numpy().max()), 0.05),
        linewidths=0.5,
        linecolor="white",
        annot_kws={"fontsize": 8},
        cbar_kws={"label": "Proporsi per keluarga aktual"},
        ax=ax,
    )
    ax.set_title("Confusion matrix antar keluarga model")
    ax.set_xlabel("Prediksi")
    ax.set_ylabel("Aktual")
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    fig.tight_layout()
    _save_figure(fig, path)
