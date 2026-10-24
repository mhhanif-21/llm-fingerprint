"""Pembuatan laporan Markdown, HTML, dan ringkasan JSON."""

from __future__ import annotations

import importlib.metadata
import logging
import platform
from typing import Any

import pandas as pd

from ..config import EDAConfig

try:
    import markdown as markdown_lib
except ImportError:  # Paket markdown tidak selalu tersedia di runtime notebook.
    markdown_lib = None


LOGGER = logging.getLogger("llm_eda")


def _format_int(value: Any) -> str:
    """Memformat nilai bilangan bulat untuk laporan."""
    try:
        return f"{int(value):,}".replace(",", ".")
    except (TypeError, ValueError):
        return str(value)


def _format_decimal(value: Any, digits: int = 3) -> str:
    """Memformat nilai desimal dengan presisi yang ditentukan."""
    try:
        return f"{float(value):.{digits}f}".replace(".", ",")
    except (TypeError, ValueError):
        return str(value)


def markdown_table(
    frame: pd.DataFrame,
    columns: list[str] | None = None,
    max_rows: int | None = None,
    float_format: str | dict[str, str] = ".3f",
    max_cell_chars: int = 180,
) -> str:
    """Mengubah DataFrame menjadi tabel Markdown tanpa dependensi tambahan."""
    if frame is None or frame.empty:
        return "_Tidak ada data._"

    data = frame.copy()
    if columns is not None:
        available = [column for column in columns if column in data.columns]
        data = data[available]
    if max_rows is not None:
        data = data.head(max_rows)

    if isinstance(float_format, str):
        formats = {
            column: float_format
            for column in data.columns
            if pd.api.types.is_float_dtype(data[column])
        }
    else:
        formats = {
            column: float_format.get(column, ".3f")
            for column in data.columns
            if pd.api.types.is_float_dtype(data[column])
        }

    for column, fmt in formats.items():
        data[column] = data[column].map(
            lambda value, format_spec=fmt: "" if pd.isna(value) else f"{value:{format_spec}}"
        )

    header = "| " + " | ".join(str(column) for column in data.columns) + " |"
    separator = "|" + "|".join(["---"] * len(data.columns)) + "|"
    rows: list[str] = []
    for _, row in data.iterrows():
        values: list[str] = []
        for value in row:
            text = "" if pd.api.types.is_scalar(value) and pd.isna(value) else str(value)
            text = " ".join(text.split())
            if len(text) > max_cell_chars:
                text = text[: max_cell_chars - 1].rstrip() + "…"
            values.append(text.replace("|", "\\|"))
        rows.append("| " + " | ".join(values) + " |")

    return "\n".join([header, separator, *rows])


def _figure_markdown(config: EDAConfig, filename: str, caption: str) -> str:
    """Membuat referensi Markdown untuk gambar laporan."""
    path = config.figures_dir / filename
    if not path.exists():
        return ""
    return f"![{caption}](figures/{filename})\n\n*{caption}.*"


def _compact_terms(terms: pd.DataFrame, top_k: int = 6) -> pd.DataFrame:
    """Meringkas istilah pembeda teratas pada setiap kelas."""
    if terms.empty:
        return pd.DataFrame(columns=["model", "family", "top_terms"])
    selected = terms.groupby("model", sort=False).head(top_k)
    rows: list[dict[str, str]] = []
    for model, group in selected.groupby("model", sort=False):
        rows.append(
            {
                "model": str(model),
                "family": str(group["family"].iloc[0]),
                "top_terms": ", ".join(group["term"].astype(str)),
            }
        )
    return pd.DataFrame(rows)


def _compact_openings(openings: pd.DataFrame, top_k: int = 2) -> pd.DataFrame:
    """Meringkas frasa pembuka yang paling sering muncul per kelas."""
    if openings.empty:
        return pd.DataFrame(columns=["model", "top_openings"])
    selected = openings.groupby("model", sort=False).head(top_k)
    rows: list[dict[str, str]] = []
    for model, group in selected.groupby("model", sort=False):
        labels = [
            f"{row.opening_phrase} ({int(row.count)})" for row in group.itertuples(index=False)
        ]
        rows.append({"model": str(model), "top_openings": "; ".join(labels)})
    return pd.DataFrame(rows)


def _top_feature_bullets(discriminability: pd.DataFrame, count: int = 5) -> str:
    """Menyusun ringkasan fitur dengan diskriminabilitas tertinggi."""
    lines: list[str] = []
    for row in discriminability.head(count).itertuples(index=False):
        lines.append(
            f"- **{row.feature}** ({row.description}) — η² = "
            f"{_format_decimal(row.eta_squared)}; rata-rata terendah "
            f"**{row.min_model}** ({_format_decimal(row.min_value, 2)}), "
            f"tertinggi **{row.max_model}** ({_format_decimal(row.max_value, 2)})."
        )
    return "\n".join(lines)


def _integrity_table(integrity: dict[str, Any]) -> pd.DataFrame:
    """Mengubah hasil audit integritas menjadi tabel laporan."""
    rows = [
        ("ID duplikat (train)", integrity["duplicate_train_ids"]),
        ("Teks duplikat (train)", integrity["duplicate_train_texts"]),
        ("Pasangan prompt-model duplikat", integrity["duplicate_prompt_model_pairs"]),
        ("Teks kosong (train)", integrity["empty_train_texts"]),
        ("ID duplikat (test)", integrity["duplicate_test_ids"]),
        ("Teks duplikat (test)", integrity["duplicate_test_texts"]),
        (
            "ID test yang tidak ada di sample submission",
            integrity["test_ids_missing_from_sample"],
        ),
    ]
    return pd.DataFrame(
        [(label, "tidak tersedia" if value is None else str(value)) for label, value in rows],
        columns=["Pemeriksaan", "Nilai"],
    )


def collect_versions() -> dict[str, str]:
    """Mengumpulkan versi Python dan pustaka utama."""
    versions = {"python": platform.python_version()}
    for package in (
        "pandas",
        "numpy",
        "scikit-learn",
        "scipy",
        "matplotlib",
        "seaborn",
        "markdown",
        "pyarrow",
    ):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = "tidak terpasang"
    return versions
