"""Pembuatan laporan Markdown, HTML, dan ringkasan JSON."""

from __future__ import annotations

import logging
from datetime import datetime
from importlib.resources import files
from pathlib import Path
from string import Template

from ..config import EDAConfig

try:
    import markdown as markdown_lib
except ImportError:  # Paket markdown tidak selalu tersedia di runtime notebook.
    markdown_lib = None


LOGGER = logging.getLogger("llm_eda")


HTML_TEMPLATE = Template(files(__package__).joinpath("report.html").read_text(encoding="utf-8"))


def write_reports(report_markdown: str, config: EDAConfig) -> dict[str, Path | None]:
    """Menulis laporan Markdown dan HTML bila dependensi tersedia."""
    markdown_path = config.output_dir / "report.md"
    html_path = config.output_dir / "report.html"

    markdown_path.write_text(report_markdown, encoding="utf-8")
    if markdown_lib is None:
        LOGGER.warning(
            "Paket markdown tidak tersedia; laporan HTML dilewati dan hanya "
            "laporan Markdown yang ditulis."
        )
        return {"markdown": markdown_path, "html": None}

    body = markdown_lib.markdown(
        report_markdown,
        extensions=["tables", "toc", "sane_lists"],
    )
    generated_at = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S %Z")
    html = HTML_TEMPLATE.substitute(
        title="EDA — Name That LLM",
        body=body,
        generated_at=generated_at,
    )
    html_path.write_text(html, encoding="utf-8")
    LOGGER.info("Laporan ditulis: %s dan %s", markdown_path, html_path)
    return {"markdown": markdown_path, "html": html_path}
