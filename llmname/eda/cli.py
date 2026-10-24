"""Antarmuka command line untuk pipeline EDA Name That LLM."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from .config import DEFAULT_OUTPUT_DIR, EDAConfig, resolve_data_dir
from .pipeline import run_eda


def build_parser() -> argparse.ArgumentParser:
    """Membangun parser argumen command line."""
    parser = argparse.ArgumentParser(
        prog="llm-eda",
        description="EDA lengkap untuk dataset Name That LLM.",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        help=("Folder data yang memuat train.csv, test.csv, dan sample_submission.parquet."),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Folder output untuk laporan, tabel, dan grafik.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed untuk evaluasi (default: 42).",
    )
    parser.add_argument(
        "--n-splits",
        type=int,
        default=5,
        help="Jumlah fold cross-validation (default: 5).",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=15,
        help="Jumlah item teratas pada tabel peringkat (default: 15).",
    )
    parser.add_argument(
        "--near-duplicate-threshold",
        type=float,
        default=0.90,
        help="Ambang cosine similarity untuk near-duplicate (default: 0.90).",
    )
    parser.add_argument(
        "--skip-modeling",
        action="store_true",
        help="Lewati evaluasi baseline dan analisis kesalahan.",
    )
    parser.add_argument(
        "--skip-near-duplicates",
        action="store_true",
        help="Lewati analisis near-duplicate.",
    )
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Level logging (default: INFO).",
    )
    parser.add_argument(
        "--version",
        action="version",
        version="llm-eda 1.0.0",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Menjalankan pipeline dari argumen command line."""
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.n_splits < 2:
        parser.error("--n-splits harus minimal 2.")
    if args.top_n < 1:
        parser.error("--top-n harus minimal 1.")
    if not 0.0 <= args.near_duplicate_threshold <= 1.0:
        parser.error("--near-duplicate-threshold harus berada pada rentang 0–1.")

    try:
        data_dir = resolve_data_dir(args.data_dir)
        config = EDAConfig(
            data_dir=data_dir,
            output_dir=args.output_dir.expanduser().resolve(),
            seed=args.seed,
            n_splits=args.n_splits,
            top_n=args.top_n,
            near_duplicate_threshold=args.near_duplicate_threshold,
            skip_modeling=args.skip_modeling,
            skip_near_duplicates=args.skip_near_duplicates,
            log_level=args.log_level,
        )
        run_eda(config)
    except Exception:
        logging.getLogger("llm_eda").exception("EDA gagal dijalankan.")
        return 1
    return 0
