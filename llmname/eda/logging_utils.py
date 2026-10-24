"""Utilitas logging untuk pipeline EDA."""

from __future__ import annotations

import logging
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

LOGGER_NAME = "llm_eda"
LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def configure_logging(level: str = "INFO", log_file: Path | None = None) -> logging.Logger:
    """Mengatur logger utama untuk konsol dan berkas."""
    logger = logging.getLogger(LOGGER_NAME)
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    logger.handlers.clear()
    logger.propagate = False

    formatter = logging.Formatter(LOG_FORMAT, datefmt=DATE_FORMAT)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def get_logger() -> logging.Logger:
    """Mengambil logger utama tanpa mengubah konfigurasi."""
    return logging.getLogger(LOGGER_NAME)


@contextmanager
def log_duration(logger: logging.Logger, message: str) -> Iterator[None]:
    """Mencatat awal, akhir, dan durasi sebuah tahapan."""
    start = time.perf_counter()
    logger.info("%s ...", message)
    try:
        yield
    except Exception:
        elapsed = time.perf_counter() - start
        logger.exception("%s gagal setelah %.2f detik.", message, elapsed)
        raise
    else:
        elapsed = time.perf_counter() - start
        logger.info("%s selesai (%.2f detik).", message, elapsed)
