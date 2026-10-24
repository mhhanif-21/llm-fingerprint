"""Komposisi transformer dan estimator untuk eksperimen lokal maupun Kaggle."""

from pathlib import Path

from joblib import Memory
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import MaxAbsScaler

from llmname.config import validate_config
from llmname.features.transformers import StyleTransformer, TextSelector, select_style_columns

from .factory import build_model


def make_vectorizer(options: dict) -> TfidfVectorizer:
    """Membuat vectorizer tanpa mempelajari vocabulary pada tahap konstruksi."""
    params = dict(options)
    if set(params) - set(TfidfVectorizer().get_params()):
        raise ValueError("Parameter TF-IDF tidak dikenal")
    if "ngram_range" in params:
        bounds = params["ngram_range"]
        if len(bounds) != 2 or any(type(value) is not int for value in bounds):
            raise ValueError("ngram_range harus dua integer")
        if not 1 <= bounds[0] <= bounds[1]:
            raise ValueError("Rentang n-gram tidak valid")
        params["ngram_range"] = tuple(bounds)
    return TfidfVectorizer(**params)


def build_pipeline(config: dict, cache_dir: Path | None = None) -> Pipeline:
    """Membangun cabang fitur yang aktif dan estimator sesuai konfigurasi."""
    config = validate_config(config)
    settings = config["features"]
    transformers = []
    for branch in settings["branches"]:
        if branch == "style":
            columns = tuple(select_style_columns(settings["style"]))
            transformer = Pipeline(
                [
                    ("extract", StyleTransformer(columns)),
                    ("scale", MaxAbsScaler()),
                ]
            )
        else:
            transformer = Pipeline(
                [
                    ("text", TextSelector()),
                    ("tfidf", make_vectorizer(settings[branch])),
                ]
            )
        transformers.append((branch, transformer))
    # Memory mengunci cache pada estimator, input training, label, dan parameter tiap fold.
    memory = Memory(cache_dir, verbose=0) if cache_dir and settings.get("cache") else None
    return Pipeline(
        [
            ("features", FeatureUnion(transformers)),
            ("model", build_model(config["model"], config["experiment"]["seed"])),
        ],
        memory=memory,
    )
