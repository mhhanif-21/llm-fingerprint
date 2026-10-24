"""Pengujian utilitas pelaporan."""

from __future__ import annotations

import pandas as pd

from llm_eda.reporting import markdown_table


def test_markdown_table_formats_floats_and_escapes_pipes() -> None:
    """Memverifikasi format desimal dan pelolosan karakter tabel Markdown."""
    frame = pd.DataFrame(
        {
            "name": ["alpha|beta"],
            "score": [0.12345],
        }
    )

    table = markdown_table(frame, float_format={"score": ".2f"})

    assert "| name | score |" in table
    assert "alpha\\|beta" in table
    assert "0.12" in table


def test_markdown_table_handles_empty_frame() -> None:
    """Memverifikasi perilaku format laporan pada tabel kosong."""
    table = markdown_table(pd.DataFrame())

    assert table == "_Tidak ada data._"
