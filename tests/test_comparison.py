"""Pemeriksaan tabel perbandingan yang berasal dari run nyata."""

from llmname.modeling.comparison import compare_runs


def test_comparison_keeps_smoke_mode_visible(trained_run, tmp_path):
    """Menyertakan mode dan kelompok pembanding agar skor smoke tidak disalahartikan."""
    table = compare_runs(trained_run.parent, tmp_path / "comparison.csv")
    assert table["mode"].eq("smoke").all()
    assert table["comparison_group"].str.len().gt(100).all()
    assert (tmp_path / "comparison.csv").is_file()
