"""Pengujian cache nyata dan invalidasi berdasarkan masukan."""

import numpy as np

from llmname.modeling.pipeline import build_pipeline


def test_cache_preserves_results_and_invalidates_features(config, real_frame, tmp_path):
    """Memakai ulang fitur hanya ketika input dan parameter transformernya cocok."""
    config["features"]["branches"] = ["style"]
    first = build_pipeline(config, tmp_path)
    first.fit(real_frame[["text"]], real_frame["model"])
    entries = list(tmp_path.rglob("output.pkl"))
    assert entries
    second = build_pipeline(config, tmp_path)
    second.fit(real_frame[["text"]], real_frame["model"])
    assert len(list(tmp_path.rglob("output.pkl"))) == len(entries)
    np.testing.assert_array_equal(first.predict(real_frame), second.predict(real_frame))
    config["features"]["style"]["exclude"] = ["em_dash_rate"]
    changed = build_pipeline(config, tmp_path)
    changed.fit(real_frame[["text"]], real_frame["model"])
    assert len(list(tmp_path.rglob("output.pkl"))) > len(entries)


def test_cache_changes_when_training_text_changes(config, real_frame, tmp_path):
    """Membedakan subset input training sehingga cache tidak menyeberang fold."""
    config["features"]["branches"] = ["style"]
    first = build_pipeline(config, tmp_path)
    first.fit(real_frame[["text"]], real_frame["model"])
    count = len(list(tmp_path.rglob("output.pkl")))
    second = build_pipeline(config, tmp_path)
    second.fit(real_frame.iloc[1:][["text"]], real_frame.iloc[1:]["model"])
    assert len(list(tmp_path.rglob("output.pkl"))) > count
