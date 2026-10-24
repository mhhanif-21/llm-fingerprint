"""Pembuatan laporan Markdown, HTML, dan ringkasan JSON."""

from __future__ import annotations

import logging
from datetime import datetime
from importlib.resources import files
from string import Template
from typing import Any

from ..artifacts import AnalysisArtifacts
from ..config import EDAConfig

try:
    import markdown as markdown_lib
except ImportError:  # Paket markdown tidak selalu tersedia di runtime notebook.
    markdown_lib = None


LOGGER = logging.getLogger("llm_eda")


HTML_TEMPLATE = Template(files(__package__).joinpath("report.html").read_text(encoding="utf-8"))


def build_summary(
    artifacts: AnalysisArtifacts,
    config: EDAConfig,
    runtime_seconds: float,
    versions: dict[str, str],
) -> dict[str, Any]:
    """Menyusun ringkasan JSON yang mudah dikonsumsi program lain."""
    audit = artifacts.audit
    summary: dict[str, Any] = {
        "generated_at": datetime.now().astimezone().isoformat(),
        "runtime_seconds": float(runtime_seconds),
        "data_dir": str(config.data_dir),
        "output_dir": str(config.output_dir),
        "dataset": audit.summary,
        "integrity": audit.integrity,
        "imbalance": audit.imbalance,
        "near_duplicate": artifacts.near_duplicate_summary,
        "top_features": artifacts.feature_analysis.discriminability.head(10)[
            ["feature", "eta_squared", "p_adjusted", "effect_size"]
        ].to_dict(orient="records"),
        "class_distribution": audit.class_distribution.to_dict(orient="records"),
        "versions": versions,
        "warnings": artifacts.warnings,
        "tables": sorted(path.name for path in config.tables_dir.glob("*.csv")),
        "figures": sorted(path.name for path in config.figures_dir.glob("*.png")),
        "report": {
            "markdown": str(config.output_dir / "report.md"),
            "html": str(config.output_dir / "report.html"),
        },
    }

    if artifacts.modeling is not None:
        summary["modeling"] = {
            "best_config": artifacts.modeling.best_config,
            "best_feature_set": artifacts.modeling.best_feature_set,
            "best_classifier": artifacts.modeling.best_classifier,
            "balanced_accuracy": artifacts.modeling.best_balanced_accuracy,
            "accuracy": artifacts.modeling.best_accuracy,
            "comparison": artifacts.modeling.comparison.to_dict(orient="records"),
            "top_confusion_pairs": artifacts.modeling.confusion_pairs.head(10).to_dict(
                orient="records"
            ),
        }
    else:
        summary["modeling"] = {"skipped": True}

    return summary
