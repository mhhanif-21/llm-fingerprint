"""Facade kompatibilitas untuk modul analisis yang telah dipisahkan."""

from .duplicates import _pair_rows as _pair_rows
from .duplicates import build_near_duplicate_analysis as build_near_duplicate_analysis
from .statistics import build_non_ascii_table as build_non_ascii_table
from .statistics import build_opening_phrases as build_opening_phrases
from .statistics import build_prompt_text_statistics as build_prompt_text_statistics
from .statistics import build_text_statistics as build_text_statistics
from .statistics import build_top_words as build_top_words
