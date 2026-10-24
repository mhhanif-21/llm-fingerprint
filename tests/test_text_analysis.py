"""Pengujian pola teks dan perhitungan diagnostik tambahan."""

import pandas as pd

from llmname.eda.extra import analyze_extra
from llmname.features.candidates import extract_candidates


def test_candidates_are_deterministic(real_frame):
    """Memastikan fitur struktur dapat dihitung berulang tanpa state dataset."""
    text = real_frame["text"].iloc[0]
    assert extract_candidates(text) == extract_candidates(text)
    assert all(value >= 0 for value in extract_candidates(text).values())


def test_relative_length_matches_prompt_medians(real_frame, tmp_path):
    """Memeriksa panjang relatif terhadap perhitungan independen pada prompt nyata."""
    summary = analyze_extra(real_frame, tmp_path)
    result = pd.read_csv(tmp_path / "prompt_normalized_length.csv").set_index("model")
    from llmname.eda.features import add_features

    frame = add_features(real_frame)
    frame["ratio"] = frame["n_words"] / frame.groupby("prompt_id")["n_words"].transform("median")
    expected = frame.groupby("model")["ratio"].median()
    pd.testing.assert_series_equal(
        result["median_relative_length"].sort_index(), expected.sort_index(), check_names=False
    )
    assert summary["training_executed"] is False
