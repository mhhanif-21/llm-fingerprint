"""Pembuatan grafik EDA dengan gaya visual yang konsisten."""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Patch

matplotlib.use("Agg")


LOGGER = logging.getLogger("llm_eda")

FAMILY_COLORS = {
    "claude": "#D55E00",
    "gemini": "#0072B2",
    "glm": "#009E73",
    "gpt": "#CC79A7",
    "kimi": "#E69F00",
    "qwen": "#56B4E9",
    "other": "#7F7F7F",
}

EFFECT_COLORS = {
    "dapat diabaikan": "#BDBDBD",
    "kecil": "#90CAF9",
    "sedang": "#42A5F5",
    "besar": "#1565C0",
    "tidak tersedia": "#E0E0E0",
}


def configure_plots() -> None:
    """Mengatur tema dan parameter global grafik."""
    sns.set_theme(style="whitegrid", context="notebook", palette="deep")
    plt.rcParams.update(
        {
            "figure.dpi": 120,
            "savefig.dpi": 150,
            "font.family": "DejaVu Sans",
            "axes.titlesize": 13,
            "axes.titleweight": "bold",
            "axes.labelsize": 10,
            "axes.edgecolor": "#B0B0B0",
            "axes.linewidth": 0.8,
            "grid.color": "#E5E5E5",
            "grid.linewidth": 0.7,
            "legend.frameon": False,
        }
    )


def _save_figure(fig: plt.Figure, path: Path) -> None:
    """Menyimpan gambar dan menutup objek figure setelah penulisan."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    LOGGER.debug("Grafik disimpan: %s", path)


def _family_legend(families: list[str]) -> list[Patch]:
    """Menyusun legenda warna berdasarkan keluarga model."""
    return [
        Patch(facecolor=FAMILY_COLORS.get(family, FAMILY_COLORS["other"]), label=family)
        for family in families
    ]
