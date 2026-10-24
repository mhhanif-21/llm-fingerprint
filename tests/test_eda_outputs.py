"""Verifikasi artefak nyata EDA tanpa menjalankan training."""

import json

from llmname.eda.config import EDAConfig
from llmname.eda.pipeline import run_eda


def test_quick_eda_writes_reports_and_figures(data_dir, tmp_path):
    """Menjalankan paket EDA lengkap pada data nyata dengan modeling dimatikan."""
    config = EDAConfig(
        data_dir=data_dir, output_dir=tmp_path, skip_modeling=True, skip_near_duplicates=True
    )
    run_eda(config)
    summary = json.loads((tmp_path / "summary.json").read_text())
    assert summary["dataset"]["train_rows"] > 0
    assert summary["modeling"]["skipped"] is True
    assert (tmp_path / "report.html").stat().st_size > 1000
    report = (tmp_path / "report.md").read_text()
    assert "Modeling tidak dijalankan" in report
    for name in summary["figures"]:
        assert (tmp_path / "figures" / name).stat().st_size > 100
