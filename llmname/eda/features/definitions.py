"""Ekstraksi fitur teks, gaya, tanda baca, dan format."""

from __future__ import annotations

import re

WORD_PATTERN = re.compile(r"[A-Za-zÀ-ÖØ-öø-ÿ]+(?:['’\-][A-Za-zÀ-ÖØ-öø-ÿ]+)*")
SENTENCE_SPLIT_PATTERN = re.compile(r"(?<=[.!?])\s+")
PARAGRAPH_SPLIT_PATTERN = re.compile(r"\n\s*\n")

FIRST_PERSON_WORDS = {"i", "me", "my", "mine", "we", "us", "our", "ours"}
SECOND_PERSON_WORDS = {"you", "your", "yours"}
REFUSAL_PATTERN = re.compile(
    r"\b(as an ai|i cannot|i can't|i'm sorry|i am unable|i apologize)\b",
    flags=re.IGNORECASE,
)

PUNCTUATION_PATTERNS: dict[str, re.Pattern[str]] = {
    "em_dash_rate": re.compile("—"),
    "en_dash_rate": re.compile("–"),
    "hyphen_rate": re.compile("-"),
    "curly_apostrophe_rate": re.compile("’"),
    "straight_apostrophe_rate": re.compile("'"),
    "curly_quote_rate": re.compile(r"[“”]"),
    "straight_quote_rate": re.compile('"'),
    "colon_rate": re.compile(":"),
    "semicolon_rate": re.compile(";"),
    "comma_rate": re.compile(","),
    "period_rate": re.compile(r"\."),
    "question_rate": re.compile(r"\?"),
    "exclamation_rate": re.compile("!"),
    "parenthesis_rate": re.compile(r"[()]"),
    "bracket_rate": re.compile(r"[\[\]]"),
    "ellipsis_rate": re.compile(r"…|\.\.\."),
    "slash_rate": re.compile("/"),
    "ampersand_rate": re.compile("&"),
    "percent_rate": re.compile("%"),
    "bullet_rate": re.compile("•"),
    "hash_rate": re.compile("#"),
    "asterisk_rate": re.compile(r"\*"),
    "underscore_rate": re.compile("_"),
}

FORMAT_PATTERNS: dict[str, re.Pattern[str]] = {
    "markdown_bold_rate": re.compile(r"\*\*[^*]+\*\*"),
    "markdown_heading_rate": re.compile(r"(?m)^#{1,6}\s"),
    "markdown_bullet_rate": re.compile(r"(?m)^\s*[-*•]\s"),
    "code_fence_rate": re.compile(r"```"),
    "url_rate": re.compile(r"https?://"),
    "newline_rate": re.compile(r"\n"),
}

START_PATTERNS: dict[str, str] = {
    "starts_with_sure": r"sure\b",
    "starts_with_here": r"here\b",
    "starts_with_i": r"i\b",
    "starts_with_the": r"the\b",
}

LEXICAL_PATTERNS: dict[str, re.Pattern[str]] = {
    "hedge_rate": re.compile(
        r"\b(may|might|could|perhaps|possibly|likely|probably|apparently|seemingly)\b",
        flags=re.IGNORECASE,
    ),
    "booster_rate": re.compile(
        r"\b(very|really|extremely|absolutely|definitely|certainly|clearly|obviously|"
        r"significantly|highly|incredibly|truly)\b",
        flags=re.IGNORECASE,
    ),
    "discourse_marker_rate": re.compile(
        r"\b(however|moreover|therefore|thus|furthermore|nevertheless|consequently|"
        r"additionally|meanwhile|overall|ultimately)\b",
        flags=re.IGNORECASE,
    ),
    "negation_rate": re.compile(
        r"\b(not|no|never|none|neither|nor|cannot|can't|don't|doesn't|didn't|"
        r"isn't|aren't|wasn't|weren't|won't|wouldn't|shouldn't|couldn't)\b",
        flags=re.IGNORECASE,
    ),
    "modal_rate": re.compile(
        r"\b(can|could|may|might|must|shall|should|will|would)\b",
        flags=re.IGNORECASE,
    ),
    "ai_cliche_rate": re.compile(
        r"\b(delve|tapestry|landscape|realm|navigate|navigating|multifaceted|pivotal|"
        r"crucial|underscore|underscores|testament|seamless|seamlessly|leverage|"
        r"robust|holistic|intricate|profound|comprehensive)\b",
        flags=re.IGNORECASE,
    ),
}

FEATURE_GROUPS: dict[str, list[str]] = {
    "length": [
        "n_chars",
        "n_words",
        "n_unique_words",
        "type_token_ratio",
        "avg_word_length",
        "std_word_length",
        "n_sentences",
        "avg_sentence_words",
        "std_sentence_words",
        "max_sentence_words",
        "n_paragraphs",
        "avg_paragraph_chars",
        "max_paragraph_chars",
    ],
    "case": [
        "uppercase_ratio",
        "all_caps_word_ratio",
        "title_case_word_ratio",
    ],
    "lexical": [
        "stopword_ratio",
        "contraction_rate",
        "first_person_rate",
        "second_person_rate",
        "hedge_rate",
        "booster_rate",
        "discourse_marker_rate",
        "negation_rate",
        "modal_rate",
        "ai_cliche_rate",
        "refusal_flag",
    ],
    "punctuation": [
        "em_dash_rate",
        "en_dash_rate",
        "hyphen_rate",
        "curly_apostrophe_rate",
        "straight_apostrophe_rate",
        "curly_quote_rate",
        "straight_quote_rate",
        "colon_rate",
        "semicolon_rate",
        "comma_rate",
        "period_rate",
        "question_rate",
        "exclamation_rate",
        "parenthesis_rate",
        "bracket_rate",
        "ellipsis_rate",
        "slash_rate",
        "ampersand_rate",
        "percent_rate",
        "bullet_rate",
        "hash_rate",
        "asterisk_rate",
        "underscore_rate",
    ],
    "format": [
        "markdown_bold_rate",
        "markdown_heading_rate",
        "markdown_bullet_rate",
        "code_fence_rate",
        "url_rate",
        "newline_rate",
        "starts_with_sure",
        "starts_with_here",
        "starts_with_i",
        "starts_with_the",
    ],
    "unicode": [
        "non_ascii_ratio",
        "punctuation_ratio",
        "digit_ratio",
        "whitespace_ratio",
    ],
}

FEATURE_COLUMNS: list[str] = [column for columns in FEATURE_GROUPS.values() for column in columns]

FEATURE_DESCRIPTIONS: dict[str, str] = {
    "n_chars": "Jumlah karakter",
    "n_words": "Jumlah kata",
    "n_unique_words": "Jumlah kata unik",
    "type_token_ratio": "Rasio kata unik terhadap total kata",
    "avg_word_length": "Rata-rata panjang kata (karakter)",
    "std_word_length": "Simpangan baku panjang kata",
    "n_sentences": "Jumlah kalimat",
    "avg_sentence_words": "Rata-rata jumlah kata per kalimat",
    "std_sentence_words": "Simpangan baku jumlah kata per kalimat",
    "max_sentence_words": "Jumlah kata pada kalimat terpanjang",
    "n_paragraphs": "Jumlah paragraf",
    "avg_paragraph_chars": "Rata-rata panjang paragraf (karakter)",
    "max_paragraph_chars": "Panjang paragraf terpanjang (karakter)",
    "uppercase_ratio": "Rasio huruf kapital terhadap seluruh karakter",
    "all_caps_word_ratio": "Rasio kata yang seluruhnya kapital",
    "title_case_word_ratio": "Rasio kata yang diawali huruf kapital",
    "stopword_ratio": "Rasio stopword bahasa Inggris terhadap seluruh kata",
    "contraction_rate": "Rasio kata yang mengandung apostrof (kontraksi)",
    "first_person_rate": "Rasio kata orang pertama (I, we, my, our, dan sejenisnya)",
    "second_person_rate": "Rasio kata orang kedua (you, your, yours)",
    "hedge_rate": "Frekuensi kata hedging per 1.000 kata",
    "booster_rate": "Frekuensi kata penguat per 1.000 kata",
    "discourse_marker_rate": "Frekuensi penanda wacana per 1.000 kata",
    "negation_rate": "Frekuensi negasi per 1.000 kata",
    "modal_rate": "Frekuensi modal verb per 1.000 kata",
    "ai_cliche_rate": "Frekuensi frasa klise LLM per 1.000 kata",
    "refusal_flag": "Indikasi teks berisi penolakan atau permintaan maaf",
    "em_dash_rate": "Frekuensi em dash (—) per 1.000 karakter",
    "en_dash_rate": "Frekuensi en dash (–) per 1.000 karakter",
    "hyphen_rate": "Frekuensi tanda hubung (-) per 1.000 karakter",
    "curly_apostrophe_rate": "Frekuensi apostrof melengkung (’) per 1.000 karakter",
    "straight_apostrophe_rate": "Frekuensi apostrof lurus (') per 1.000 karakter",
    "curly_quote_rate": "Frekuensi kutip melengkung (“ dan ”) per 1.000 karakter",
    "straight_quote_rate": 'Frekuensi kutip lurus (") per 1.000 karakter',
    "colon_rate": "Frekuensi titik dua per 1.000 karakter",
    "semicolon_rate": "Frekuensi titik koma per 1.000 karakter",
    "comma_rate": "Frekuensi koma per 1.000 karakter",
    "period_rate": "Frekuensi titik per 1.000 karakter",
    "question_rate": "Frekuensi tanda tanya per 1.000 karakter",
    "exclamation_rate": "Frekuensi tanda seru per 1.000 karakter",
    "parenthesis_rate": "Frekuensi tanda kurung per 1.000 karakter",
    "bracket_rate": "Frekuensi tanda kurung siku per 1.000 karakter",
    "ellipsis_rate": "Frekuensi elipsis (… atau ...) per 1.000 karakter",
    "slash_rate": "Frekuensi garis miring per 1.000 karakter",
    "ampersand_rate": "Frekuensi ampersand (&) per 1.000 karakter",
    "percent_rate": "Frekuensi tanda persen per 1.000 karakter",
    "bullet_rate": "Frekuensi bullet (•) per 1.000 karakter",
    "hash_rate": "Frekuensi tanda pagar (#) per 1.000 karakter",
    "asterisk_rate": "Frekuensi asterisk (*) per 1.000 karakter",
    "underscore_rate": "Frekuensi garis bawah (_) per 1.000 karakter",
    "markdown_bold_rate": "Frekuensi penanda bold Markdown per 1.000 karakter",
    "markdown_heading_rate": "Frekuensi heading Markdown per 1.000 karakter",
    "markdown_bullet_rate": "Frekuensi baris bullet Markdown per 1.000 karakter",
    "code_fence_rate": "Frekuensi code fence Markdown per 1.000 karakter",
    "url_rate": "Frekuensi URL per 1.000 karakter",
    "newline_rate": "Frekuensi newline per 1.000 karakter",
    "starts_with_sure": "Teks diawali kata “Sure”",
    "starts_with_here": "Teks diawali kata “Here”",
    "starts_with_i": "Teks diawali kata “I”",
    "starts_with_the": "Teks diawali kata “The”",
    "non_ascii_ratio": "Rasio karakter non-ASCII terhadap seluruh karakter",
    "punctuation_ratio": "Rasio karakter non-kata dan non-spasi",
    "digit_ratio": "Rasio digit terhadap seluruh karakter",
    "whitespace_ratio": "Rasio whitespace terhadap seluruh karakter",
}
