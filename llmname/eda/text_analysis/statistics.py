"""Analisis teks: statistik, karakter, kata, frasa, dan near-duplicate."""

from __future__ import annotations

import logging
import unicodedata
from collections import Counter
from typing import Any

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer

from ..features import FEATURE_DESCRIPTIONS

LOGGER = logging.getLogger("llm_eda")

TEXT_STATISTIC_FEATURES = [
    "n_chars",
    "n_words",
    "n_sentences",
    "avg_word_length",
    "type_token_ratio",
    "avg_sentence_words",
    "em_dash_rate",
    "curly_apostrophe_rate",
    "colon_rate",
    "semicolon_rate",
    "comma_rate",
    "question_rate",
]


def build_text_statistics(
    feature_table: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Menghitung statistik teks keseluruhan dan per model."""
    overall = (
        feature_table[TEXT_STATISTIC_FEATURES]
        .describe(percentiles=[0.05, 0.25, 0.5, 0.75, 0.95])
        .T.reset_index()
        .rename(columns={"index": "feature"})
    )
    overall.insert(
        1,
        "description",
        overall["feature"].map(FEATURE_DESCRIPTIONS).fillna(""),
    )

    frame = feature_table.copy()
    frame["has_em_dash"] = (frame["em_dash_rate"] > 0).astype(int)
    frame["has_curly_apostrophe"] = (frame["curly_apostrophe_rate"] > 0).astype(int)
    frame["has_curly_quote"] = (frame["curly_quote_rate"] > 0).astype(int)

    by_model = (
        frame.groupby("model")
        .agg(
            family=("family", "first"),
            n=("text", "size"),
            chars_mean=("n_chars", "mean"),
            chars_median=("n_chars", "median"),
            words_mean=("n_words", "mean"),
            words_median=("n_words", "median"),
            sentences_mean=("n_sentences", "mean"),
            avg_word_length_mean=("avg_word_length", "mean"),
            type_token_ratio_mean=("type_token_ratio", "mean"),
            em_dash_pct=("has_em_dash", lambda values: values.mean() * 100.0),
            curly_apostrophe_pct=(
                "has_curly_apostrophe",
                lambda values: values.mean() * 100.0,
            ),
            curly_quote_pct=("has_curly_quote", lambda values: values.mean() * 100.0),
            colon_rate_mean=("colon_rate", "mean"),
            semicolon_rate_mean=("semicolon_rate", "mean"),
            comma_rate_mean=("comma_rate", "mean"),
            question_rate_mean=("question_rate", "mean"),
            refusal_pct=("refusal_flag", lambda values: values.mean() * 100.0),
        )
        .reset_index()
        .sort_values("em_dash_pct", ascending=False)
    )

    numeric_columns = by_model.select_dtypes(include="number").columns
    by_model[numeric_columns] = by_model[numeric_columns].round(2)
    overall_numeric = overall.select_dtypes(include="number").columns
    overall[overall_numeric] = overall[overall_numeric].round(3)
    return overall, by_model


def build_non_ascii_table(frame: pd.DataFrame, top_n: int = 15) -> pd.DataFrame:
    """Mendata karakter non-ASCII yang paling sering muncul."""
    texts = frame["text"].fillna("").astype(str)
    counter: Counter[str] = Counter()
    for value in texts:
        counter.update(char for char in value if ord(char) > 127)

    total_chars = int(texts.str.len().sum())
    rows: list[dict[str, Any]] = []
    for char, count in counter.most_common(top_n):
        mask = texts.str.contains(char, regex=False)
        texts_using = int(mask.sum())
        models_using = int(frame.loc[mask, "model"].nunique())
        first_index = frame.index[mask][0] if mask.any() else None
        rows.append(
            {
                "character": char,
                "codepoint": f"U+{ord(char):04X}",
                "unicode_name": unicodedata.name(char, "TIDAK DIKETAHUI"),
                "count": int(count),
                "pct_of_chars": count / total_chars * 100.0 if total_chars else 0.0,
                "texts_using": texts_using,
                "pct_of_texts": texts_using / len(frame) * 100.0 if len(frame) else 0.0,
                "models_using": models_using,
                "example_id": frame.loc[first_index, "id"] if first_index is not None else "",
                "example_model": (
                    frame.loc[first_index, "model"] if first_index is not None else ""
                ),
            }
        )

    return pd.DataFrame(rows)


def build_top_words(frame: pd.DataFrame, top_n: int = 15) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Menghitung kata paling sering secara keseluruhan dan per model."""
    texts = frame["text"].fillna("").astype(str)
    vectorizer = CountVectorizer(
        stop_words="english",
        ngram_range=(1, 1),
        min_df=2,
        lowercase=True,
    )
    matrix = vectorizer.fit_transform(texts)
    terms = vectorizer.get_feature_names_out()
    total_count = float(matrix.sum())
    counts = np.asarray(matrix.sum(axis=0)).ravel()

    if total_count <= 0:
        return (
            pd.DataFrame(columns=["term", "count", "percentage_of_tokens"]),
            pd.DataFrame(columns=["model", "rank", "term", "count", "rate_per_1000_tokens"]),
        )

    top_indices = np.argsort(counts)[::-1][:top_n]
    overall = pd.DataFrame(
        {
            "term": terms[top_indices],
            "count": counts[top_indices].astype(int),
            "percentage_of_tokens": counts[top_indices] / total_count * 100.0,
        }
    )

    rows: list[dict[str, Any]] = []
    for model, group in frame.groupby("model"):
        group_matrix = vectorizer.transform(group["text"].fillna("").astype(str))
        group_counts = np.asarray(group_matrix.sum(axis=0)).ravel()
        group_total = float(group_counts.sum())
        group_top = np.argsort(group_counts)[::-1][:top_n]
        for rank, index in enumerate(group_top, start=1):
            if group_counts[index] <= 0:
                continue
            rows.append(
                {
                    "model": model,
                    "rank": rank,
                    "term": terms[index],
                    "count": int(group_counts[index]),
                    "rate_per_1000_tokens": (
                        group_counts[index] / group_total * 1000.0 if group_total else 0.0
                    ),
                }
            )

    by_model = pd.DataFrame(rows)
    return overall, by_model


def build_opening_phrases(frame: pd.DataFrame, top_n: int = 3) -> pd.DataFrame:
    """Menghitung frasa pembuka yang paling sering per model."""
    if "opening_phrase" not in frame.columns:
        return pd.DataFrame(columns=["model", "rank", "opening_phrase", "count", "percentage"])

    valid = frame[frame["opening_phrase"].fillna("").ne("")]
    grouped = valid.groupby(["model", "opening_phrase"]).size().reset_index(name="count")
    totals = valid.groupby("model").size()
    grouped["percentage"] = grouped["count"] / grouped["model"].map(totals) * 100.0
    grouped = grouped.sort_values(
        ["model", "count", "opening_phrase"],
        ascending=[True, False, True],
    )
    grouped["rank"] = grouped.groupby("model").cumcount() + 1
    return grouped[grouped["rank"] <= top_n].reset_index(drop=True)


def build_prompt_text_statistics(feature_table: pd.DataFrame) -> pd.DataFrame:
    """Menghitung variasi panjang dan gaya teks dalam setiap prompt."""
    grouped = (
        feature_table.groupby("prompt_id")
        .agg(
            n=("text", "size"),
            mean_chars=("n_chars", "mean"),
            std_chars=("n_chars", "std"),
            min_chars=("n_chars", "min"),
            max_chars=("n_chars", "max"),
            mean_words=("n_words", "mean"),
            std_words=("n_words", "std"),
            mean_em_dash=("em_dash_rate", "mean"),
            std_em_dash=("em_dash_rate", "std"),
            mean_curly_apostrophe=("curly_apostrophe_rate", "mean"),
            std_curly_apostrophe=("curly_apostrophe_rate", "std"),
        )
        .reset_index()
    )
    grouped[["std_chars", "std_words", "std_em_dash", "std_curly_apostrophe"]] = grouped[
        ["std_chars", "std_words", "std_em_dash", "std_curly_apostrophe"]
    ].fillna(0.0)
    grouped["char_range"] = grouped["max_chars"] - grouped["min_chars"]
    grouped["cv_chars"] = np.where(
        grouped["mean_chars"] > 0,
        grouped["std_chars"] / grouped["mean_chars"],
        0.0,
    )
    return grouped.sort_values("cv_chars", ascending=False).reset_index(drop=True)
