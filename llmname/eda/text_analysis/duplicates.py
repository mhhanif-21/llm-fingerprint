"""Analisis teks: statistik, karakter, kata, frasa, dan near-duplicate."""

from __future__ import annotations

import heapq
import logging
from typing import Any

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from ..config import EDAConfig
from ..io import compact_text

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


def _pair_rows(
    frame: pd.DataFrame,
    pairs: list[tuple[float, int, int]],
    threshold: float,
) -> pd.DataFrame:
    """Menyusun rincian pasangan teks berdasarkan indeks kemiripan."""
    rows: list[dict[str, Any]] = []
    for score, left, right in pairs:
        left_row = frame.iloc[left]
        right_row = frame.iloc[right]
        rows.append(
            {
                "similarity": round(float(score), 4),
                "above_threshold": bool(score >= threshold),
                "same_prompt": bool(left_row["prompt_id"] == right_row["prompt_id"]),
                "same_model": bool(left_row["model"] == right_row["model"]),
                "id_a": left_row["id"],
                "prompt_id_a": left_row["prompt_id"],
                "model_a": left_row["model"],
                "id_b": right_row["id"],
                "prompt_id_b": right_row["prompt_id"],
                "model_b": right_row["model"],
                "text_a_preview": compact_text(left_row["text"], max_chars=160),
                "text_b_preview": compact_text(right_row["text"], max_chars=160),
            }
        )
    return pd.DataFrame(rows)


def build_near_duplicate_analysis(
    frame: pd.DataFrame, config: EDAConfig
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Mendeteksi pasangan teks yang sangat mirip menggunakan cosine similarity."""
    columns = [
        "similarity",
        "above_threshold",
        "same_prompt",
        "same_model",
        "id_a",
        "prompt_id_a",
        "model_a",
        "id_b",
        "prompt_id_b",
        "model_b",
        "text_a_preview",
        "text_b_preview",
    ]
    if config.skip_near_duplicates:
        LOGGER.info("Analisis near-duplicate dilewati atas permintaan pengguna.")
        return pd.DataFrame(columns=columns), {
            "skipped": True,
            "threshold": config.near_duplicate_threshold,
        }

    texts = frame["text"].fillna("").astype(str)
    if len(texts) < 2:
        return pd.DataFrame(columns=columns), {
            "skipped": False,
            "threshold": config.near_duplicate_threshold,
            "max_similarity": float("nan"),
            "pairs_above_threshold": 0,
        }

    vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=1, sublinear_tf=True)
    matrix = vectorizer.fit_transform(texts)
    similarity = cosine_similarity(matrix)
    np.fill_diagonal(similarity, 0.0)

    row_max = similarity.max(axis=1)
    threshold = float(config.near_duplicate_threshold)
    top_heap: list[tuple[float, int, int]] = []
    threshold_heap: list[tuple[float, int, int]] = []
    threshold_count = 0
    heap_size = 20

    for left in range(len(texts) - 1):
        row = similarity[left, left + 1 :]

        candidate_count = min(5, row.size)
        candidate_indices = np.argpartition(row, -candidate_count)[-candidate_count:]
        for relative_index in candidate_indices:
            score = float(row[relative_index])
            right = left + 1 + int(relative_index)
            if len(top_heap) < heap_size:
                heapq.heappush(top_heap, (score, left, right))
            elif score > top_heap[0][0]:
                heapq.heapreplace(top_heap, (score, left, right))

        above_indices = np.flatnonzero(row >= threshold)
        threshold_count += int(above_indices.size)
        for relative_index in above_indices:
            score = float(row[relative_index])
            right = left + 1 + int(relative_index)
            if len(threshold_heap) < heap_size:
                heapq.heappush(threshold_heap, (score, left, right))
            elif score > threshold_heap[0][0]:
                heapq.heapreplace(threshold_heap, (score, left, right))

    top_pairs = sorted(top_heap, reverse=True)
    threshold_pairs = sorted(threshold_heap, reverse=True)
    selected_pairs = threshold_pairs if threshold_count > 0 else top_pairs
    pairs_frame = _pair_rows(frame, selected_pairs, threshold)

    summary = {
        "skipped": False,
        "threshold": threshold,
        "max_similarity": float(row_max.max()),
        "mean_max_similarity": float(row_max.mean()),
        "median_max_similarity": float(np.median(row_max)),
        "p95_max_similarity": float(np.percentile(row_max, 95)),
        "top_pair_similarity": float(top_pairs[0][0]) if top_pairs else float("nan"),
        "pairs_above_threshold": int(threshold_count),
    }

    LOGGER.info(
        "Near-duplicate: similarity maksimum %.4f; pasangan di atas ambang %.2f: %d.",
        summary["max_similarity"],
        threshold,
        threshold_count,
    )
    return pairs_frame, summary
