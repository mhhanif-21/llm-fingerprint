"""Komputasi fitur teks menggunakan definisi bersama."""

from __future__ import annotations

import re

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

from .definitions import (
    FEATURE_COLUMNS,
    FIRST_PERSON_WORDS,
    FORMAT_PATTERNS,
    LEXICAL_PATTERNS,
    PARAGRAPH_SPLIT_PATTERN,
    PUNCTUATION_PATTERNS,
    REFUSAL_PATTERN,
    SECOND_PERSON_WORDS,
    SENTENCE_SPLIT_PATTERN,
    START_PATTERNS,
    WORD_PATTERN,
)


def infer_model_family(model_name: str) -> str:
    """Menentukan keluarga model dari nama model."""
    normalized = str(model_name).strip().lower()
    for family in ("claude", "gemini", "glm", "gpt", "kimi", "qwen"):
        if normalized.startswith(family):
            return family
    return "other"


def extract_opening_phrase(text: str, n_words: int = 4) -> str:
    """Mengambil frasa pembuka dalam bentuk lowercase."""
    tokens = WORD_PATTERN.findall(str(text))[:n_words]
    return " ".join(tokens).lower()


def _token_statistics(text: str) -> dict[str, float]:
    """Menghitung statistik kata, kalimat, dan paragraf dari satu teks."""
    words = WORD_PATTERN.findall(text)
    word_lengths = (
        np.asarray([len(word) for word in words], dtype=float)
        if words
        else np.asarray([], dtype=float)
    )

    sentences = [
        part.strip() for part in SENTENCE_SPLIT_PATTERN.split(text.strip()) if part.strip()
    ]
    sentence_word_counts = np.asarray(
        [len(WORD_PATTERN.findall(sentence)) for sentence in sentences],
        dtype=float,
    )

    paragraphs = [
        part.strip() for part in PARAGRAPH_SPLIT_PATTERN.split(text.strip()) if part.strip()
    ]
    paragraph_lengths = np.asarray([len(paragraph) for paragraph in paragraphs], dtype=float)

    n_words = len(words)
    unique_words = len({word.lower() for word in words})
    stopword_count = sum(word.lower() in ENGLISH_STOP_WORDS for word in words)
    contraction_count = sum("'" in word or "’" in word for word in words)
    first_person_count = sum(word.lower() in FIRST_PERSON_WORDS for word in words)
    second_person_count = sum(word.lower() in SECOND_PERSON_WORDS for word in words)
    all_caps_count = sum(word.isupper() and len(word) > 1 for word in words)
    title_case_count = sum(word[:1].isupper() for word in words)

    return {
        "n_chars": float(len(text)),
        "n_words": float(n_words),
        "n_unique_words": float(unique_words),
        "type_token_ratio": float(unique_words / n_words) if n_words else 0.0,
        "avg_word_length": float(word_lengths.mean()) if word_lengths.size else 0.0,
        "std_word_length": float(word_lengths.std(ddof=0)) if word_lengths.size else 0.0,
        "n_sentences": float(len(sentences)),
        "avg_sentence_words": float(sentence_word_counts.mean())
        if sentence_word_counts.size
        else 0.0,
        "std_sentence_words": float(sentence_word_counts.std(ddof=0))
        if sentence_word_counts.size
        else 0.0,
        "max_sentence_words": float(sentence_word_counts.max())
        if sentence_word_counts.size
        else 0.0,
        "n_paragraphs": float(len(paragraphs)),
        "avg_paragraph_chars": float(paragraph_lengths.mean()) if paragraph_lengths.size else 0.0,
        "max_paragraph_chars": float(paragraph_lengths.max()) if paragraph_lengths.size else 0.0,
        "stopword_ratio": float(stopword_count / n_words) if n_words else 0.0,
        "contraction_rate": float(contraction_count / n_words) if n_words else 0.0,
        "first_person_rate": float(first_person_count / n_words) if n_words else 0.0,
        "second_person_rate": float(second_person_count / n_words) if n_words else 0.0,
        "all_caps_word_ratio": float(all_caps_count / n_words) if n_words else 0.0,
        "title_case_word_ratio": float(title_case_count / n_words) if n_words else 0.0,
    }


def _count_matches(texts: pd.Series, pattern: re.Pattern[str]) -> pd.Series:
    """Menghitung jumlah kecocokan pola regex pada setiap teks."""
    return texts.apply(lambda value: len(pattern.findall(value))).astype(float)


def _compute_features(texts: pd.Series) -> pd.DataFrame:
    """Menghitung fitur dasar dan menormalkan frekuensi tanpa nilai tak hingga."""
    records = [_token_statistics(value) for value in texts]
    features = pd.DataFrame.from_records(records)
    features.index = texts.index

    n_chars = features["n_chars"].replace(0, np.nan)
    n_words = features["n_words"].replace(0, np.nan)

    features["uppercase_ratio"] = texts.str.count(r"[A-Z]") / n_chars
    features["digit_ratio"] = texts.str.count(r"\d") / n_chars
    features["whitespace_ratio"] = texts.str.count(r"\s") / n_chars
    features["punctuation_ratio"] = texts.str.count(r"[^\w\s]") / n_chars
    features["non_ascii_ratio"] = (
        texts.apply(lambda value: sum(ord(char) > 127 for char in value)) / n_chars
    )

    for name, pattern in PUNCTUATION_PATTERNS.items():
        features[name] = _count_matches(texts, pattern) / n_chars * 1000.0

    for name, pattern in FORMAT_PATTERNS.items():
        features[name] = _count_matches(texts, pattern) / n_chars * 1000.0

    stripped = texts.str.lstrip()
    for name, pattern in START_PATTERNS.items():
        features[name] = stripped.str.match(pattern, case=False).astype(int)

    features["refusal_flag"] = texts.apply(
        lambda value: int(bool(REFUSAL_PATTERN.search(value)))
    ).astype(int)

    for name, pattern in LEXICAL_PATTERNS.items():
        features[name] = _count_matches(texts, pattern) / n_words * 1000.0

    features = features.fillna(0.0)
    missing = [column for column in FEATURE_COLUMNS if column not in features.columns]
    if missing:
        raise RuntimeError(f"Fitur berikut gagal dibuat: {', '.join(missing)}")
    return features


def add_features(frame: pd.DataFrame, text_column: str = "text") -> pd.DataFrame:
    """Menambahkan seluruh fitur teks dan kolom turunan ke DataFrame."""
    texts = frame[text_column].fillna("").astype(str).reset_index(drop=True)
    features = _compute_features(texts).reset_index(drop=True)
    result = pd.concat([frame.reset_index(drop=True), features], axis=1)

    result["family"] = (
        result["model"].map(infer_model_family) if "model" in result.columns else "other"
    )
    result["opening_phrase"] = texts.map(extract_opening_phrase)
    return result
