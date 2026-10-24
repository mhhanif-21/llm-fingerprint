"""Pembuatan grafik EDA dengan gaya visual yang konsisten."""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib.patches import Patch

from .base import EFFECT_COLORS, FAMILY_COLORS, _save_figure

matplotlib.use("Agg")


LOGGER = logging.getLogger("llm_eda")


def plot_class_distribution(class_distribution: pd.DataFrame, path: Path) -> None:
    """Membuat grafik batang distribusi kelas."""
    data = class_distribution.sort_values("count").reset_index(drop=True)
    colors = [FAMILY_COLORS.get(family, FAMILY_COLORS["other"]) for family in data["family"]]
    mean_count = float(data["count"].mean())

    fig, ax = plt.subplots(figsize=(10, 7))
    bars = ax.barh(data["model"], data["count"], color=colors, height=0.72)
    ax.axvline(
        mean_count,
        color="#333333",
        linestyle="--",
        linewidth=1.1,
        label=f"Rata-rata: {mean_count:.1f}",
    )
    for bar, count in zip(bars, data["count"], strict=True):
        ax.text(
            bar.get_width() + 0.3,
            bar.get_y() + bar.get_height() / 2,
            f"{int(count)}",
            va="center",
            fontsize=8,
            color="#333333",
        )
    ax.set_xlim(0, data["count"].max() * 1.12)
    ax.set_xlabel("Jumlah contoh")
    ax.set_ylabel("")
    ax.set_title("Distribusi kelas model pada data latih")
    ax.legend(loc="lower right")
    ax.tick_params(axis="y", labelsize=8)
    fig.tight_layout()
    _save_figure(fig, path)


def plot_text_length_by_model(feature_table: pd.DataFrame, path: Path) -> None:
    """Membuat histogram panjang teks dan boxplot per model."""
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(15, 7),
        gridspec_kw={"width_ratios": [1, 1.55]},
    )

    median_length = float(feature_table["n_chars"].median())
    axes[0].hist(
        feature_table["n_chars"],
        bins=40,
        color="#4C72B0",
        edgecolor="white",
        linewidth=0.5,
    )
    axes[0].axvline(
        median_length,
        color="#C44E52",
        linestyle="--",
        linewidth=1.2,
        label=f"Median: {median_length:.0f}",
    )
    axes[0].set_xlabel("Jumlah karakter")
    axes[0].set_ylabel("Frekuensi")
    axes[0].set_title("Distribusi panjang teks")
    axes[0].legend()

    order = feature_table.groupby("model")["n_chars"].median().sort_values().index.tolist()
    sns.boxplot(
        data=feature_table,
        x="n_chars",
        y="model",
        order=order,
        color="#7AA6C2",
        fliersize=2,
        linewidth=0.8,
        ax=axes[1],
    )
    axes[1].set_xlabel("Jumlah karakter")
    axes[1].set_ylabel("")
    axes[1].set_title("Panjang teks per model")
    axes[1].tick_params(axis="y", labelsize=8)

    fig.tight_layout()
    _save_figure(fig, path)


def plot_feature_effect_sizes(discriminability: pd.DataFrame, path: Path, top_n: int = 15) -> None:
    """Membuat grafik effect size fitur paling diskriminatif."""
    data = discriminability.head(top_n).sort_values("eta_squared").reset_index(drop=True)
    colors = [EFFECT_COLORS.get(label, "#BDBDBD") for label in data["effect_size"]]

    fig, ax = plt.subplots(figsize=(10, 7))
    bars = ax.barh(data["feature"], data["eta_squared"], color=colors, height=0.72)
    for bar, value in zip(bars, data["eta_squared"], strict=True):
        ax.text(
            bar.get_width() + max(data["eta_squared"].max() * 0.015, 0.001),
            bar.get_y() + bar.get_height() / 2,
            f"{value:.3f}",
            va="center",
            fontsize=8,
        )
    ax.axvline(0.01, color="#9E9E9E", linestyle=":", linewidth=1)
    ax.axvline(0.06, color="#757575", linestyle=":", linewidth=1)
    ax.axvline(0.14, color="#424242", linestyle=":", linewidth=1)
    ax.set_xlim(0, max(data["eta_squared"].max() * 1.15, 0.05))
    ax.set_xlabel("Eta-squared")
    ax.set_ylabel("")
    ax.set_title("Fitur gaya paling diskriminatif antar model")
    ax.tick_params(axis="y", labelsize=8)

    present_effects = [label for label in EFFECT_COLORS if label in set(data["effect_size"])]
    ax.legend(
        handles=[Patch(facecolor=EFFECT_COLORS[label], label=label) for label in present_effects],
        title="Ukuran efek",
        loc="lower right",
    )
    fig.tight_layout()
    _save_figure(fig, path)


def plot_non_ascii_characters(non_ascii: pd.DataFrame, path: Path) -> None:
    """Membuat grafik karakter non-ASCII yang paling sering muncul."""
    if non_ascii.empty:
        LOGGER.warning("Tidak ada karakter non-ASCII untuk digrafikkan.")
        return

    data = non_ascii.head(12).sort_values("count").reset_index(drop=True)
    labels = [f"{row.character}  ({row.codepoint})" for row in data.itertuples(index=False)]

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.barh(labels, data["count"], color="#8172B2", height=0.7)
    for bar, value in zip(bars, data["count"], strict=True):
        ax.text(
            bar.get_width() + data["count"].max() * 0.01,
            bar.get_y() + bar.get_height() / 2,
            f"{int(value)}",
            va="center",
            fontsize=8,
        )
    ax.set_xlim(0, data["count"].max() * 1.12)
    ax.set_xlabel("Jumlah kemunculan")
    ax.set_ylabel("")
    ax.set_title("Karakter non-ASCII yang paling sering muncul")
    ax.tick_params(axis="y", labelsize=8)
    fig.tight_layout()
    _save_figure(fig, path)
