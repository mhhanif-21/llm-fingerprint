"""Verifikasi pemisahan prompt dan persistensi fold yang deterministik."""

import pandas as pd
import pytest

from llmname.data.splits import make_folds


def test_groups_are_disjoint_and_folds_stable(real_frame):
    """Menetapkan setiap respons tepat sekali tanpa membagi prompt antar-fold."""
    folds = make_folds(real_frame, 3)
    assert folds.groupby("prompt_id")["fold"].nunique().eq(1).all()
    assert folds["id"].is_unique
    pd.testing.assert_frame_equal(folds, make_folds(real_frame, 3))


def test_rejects_insufficient_groups(real_frame):
    """Menolak jumlah fold yang melebihi jumlah prompt tersedia."""
    with pytest.raises(ValueError):
        make_folds(real_frame, real_frame["prompt_id"].nunique() + 1)
