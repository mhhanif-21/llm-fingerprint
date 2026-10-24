"""Verifikasi pipeline nyata dan isolasi statistik preprocessing."""

import pytest

from llmname.modeling.pipeline import build_pipeline


def test_vectorizer_and_scaler_fit_training_only(config, real_frame):
    """Vocabulary hanya berisi token training dan scaler tidak berubah saat transformasi."""
    config["features"]["word"].update({"ngram_range": [1, 1], "min_df": 1, "max_df": 1.0})
    training = real_frame.iloc[: len(real_frame) // 2]
    validation = real_frame.iloc[len(real_frame) // 2 :]
    pipeline = build_pipeline(config)
    pipeline.fit(training[["text"]], training["model"])
    branches = dict(pipeline.named_steps["features"].transformer_list)
    vectorizer = branches["word"].named_steps["tfidf"]
    training_tokens = set(vectorizer.build_analyzer()(" ".join(training["text"])))
    assert set(vectorizer.vocabulary_).issubset(training_tokens)
    scaler = branches["style"].named_steps["scale"]
    before = scaler.max_abs_.copy()
    pipeline.predict(validation[["text"]])
    assert (before == scaler.max_abs_).all()
    heldout_tokens = (
        set(vectorizer.build_analyzer()(" ".join(validation["text"]))) - training_tokens
    )
    assert heldout_tokens
    assert not heldout_tokens.intersection(vectorizer.vocabulary_)


def test_style_only_pipeline_and_invalid_branch(config, real_frame):
    """Menjalankan model tanpa TF-IDF serta menolak cabang yang tidak dikenal."""
    config["features"]["branches"] = ["style"]
    model = build_pipeline(config)
    model.fit(real_frame[["text"]], real_frame["model"])
    assert len(model.predict(real_frame[["text"]])) == len(real_frame)
    config["model"] = {"name": "logistic_regression", "params": {"max_iter": 2000}}
    alternative = build_pipeline(config)
    alternative.fit(real_frame[["text"]], real_frame["model"])
    assert len(alternative.predict(real_frame[["text"]])) == len(real_frame)
    config["features"]["branches"] = ["typo"]
    with pytest.raises(ValueError):
        build_pipeline(config)
