"""Fitur struktur tambahan yang belum diaktifkan pada baseline."""

import re

import numpy as np
import pandas as pd

from llmname.eda.features.definitions import SENTENCE_SPLIT_PATTERN, WORD_PATTERN

CANDIDATE_DESCRIPTIONS = {
    "numbered_list_rate": "Jumlah baris daftar bernomor per 1.000 karakter",
    "markdown_table_rate": "Jumlah baris mirip tabel Markdown per 1.000 karakter",
    "sentence_length_cv": "Koefisien variasi jumlah kata per kalimat",
    "repeated_bigram_ratio": "Proporsi kemunculan bigram yang mengulang bigram sebelumnya",
    "blank_line_ratio": "Proporsi baris kosong pada teks",
    "parenthetical_span_rate": "Jumlah rentang dalam tanda kurung per 1.000 karakter",
}


def extract_candidates(text: str) -> dict[str, float]:
    """Menghitung pola struktur dari teks tanpa informasi label maupun prompt."""
    denominator = max(len(text), 1)
    words = WORD_PATTERN.findall(text.lower())
    bigrams = list(zip(words, words[1:]))
    lengths = np.asarray(
        [
            len(WORD_PATTERN.findall(sentence))
            for sentence in SENTENCE_SPLIT_PATTERN.split(text)
            if sentence.strip()
        ],
        dtype=float,
    )
    lines = text.splitlines()
    return {
        "numbered_list_rate": len(re.findall(r"(?m)^\s*\d+[.)]\s+", text)) / denominator * 1000,
        "markdown_table_rate": len(re.findall(r"(?m)^\s*\|.+\|\s*$", text)) / denominator * 1000,
        "sentence_length_cv": float(lengths.std() / lengths.mean())
        if lengths.size and lengths.mean() > 0
        else 0.0,
        "repeated_bigram_ratio": (len(bigrams) - len(set(bigrams))) / max(len(bigrams), 1),
        "blank_line_ratio": sum(not line.strip() for line in lines) / max(len(lines), 1),
        "parenthetical_span_rate": len(re.findall(r"\([^()]*\)", text)) / denominator * 1000,
    }


def candidate_frame(texts: pd.Series) -> pd.DataFrame:
    """Menyusun tabel numerik dengan urutan kolom yang stabil."""
    return pd.DataFrame(
        [extract_candidates(text) for text in texts],
        columns=list(CANDIDATE_DESCRIPTIONS),
        index=texts.index,
    )
