"""Pengujian fitur pada respons nyata dan kasus batasnya."""

import numpy as np

from llmname.eda.features import FEATURE_COLUMNS, add_features
from llmname.features.candidates import candidate_frame
from llmname.features.transformers import StyleTransformer


def test_single_text_features_need_no_label(real_frame):
    """Menghasilkan fitur yang sama ketika label dan prompt tidak diberikan."""
    columns = tuple(FEATURE_COLUMNS)
    transformer = StyleTransformer(columns)
    first = transformer.fit_transform(real_frame[["text"]]).toarray()
    second = transformer.transform(real_frame).toarray()
    np.testing.assert_allclose(first, second)
    assert transformer.get_feature_names_out().tolist() == list(columns)


def test_empty_text_is_finite_and_unicode_is_preserved(real_frame):
    """Menangani salinan teks kosong dan tetap menghitung Unicode pada teks asli."""
    frame = real_frame[["text"]].head(4).copy()
    original = frame["text"].copy()
    frame.loc[frame.index[0], "text"] = ""
    result = add_features(frame)
    assert np.isfinite(result[FEATURE_COLUMNS].to_numpy()).all()
    assert result.loc[0, "n_chars"] == 0
    assert frame["text"].iloc[1:].tolist() == original.iloc[1:].tolist()
    assert np.isfinite(candidate_frame(frame["text"]).to_numpy()).all()
