"""Pemilihan kelompok, fitur individual, dan kandidat melalui konfigurasi."""

import pytest

from llmname.features.transformers import select_style_columns


def test_select_include_exclude_and_candidates():
    """Memilih fitur individual dan kandidat tanpa mengubah runner eksperimen."""
    selected = select_style_columns(
        {
            "groups": ["case"],
            "include": ["em_dash_rate"],
            "exclude": ["uppercase_ratio"],
            "candidates": ["sentence_length_cv"],
        }
    )
    assert selected == [
        "all_caps_word_ratio",
        "title_case_word_ratio",
        "em_dash_rate",
        "sentence_length_cv",
    ]


@pytest.mark.parametrize(
    "options",
    [{"groups": ["unknown"]}, {"include": ["model"]}, {"candidates": ["typo"]}, {"groups": []}],
)
def test_unknown_or_empty_selection_fails(options):
    """Menolak pemilihan fitur kosong, salah eja, atau kolom label."""
    with pytest.raises(ValueError):
        select_style_columns(options)
