"""Eksekusi eksperimen dengan manifest, batas smoke, dan artefak terverifikasi."""

import logging
import resource
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

import joblib
from threadpoolctl import threadpool_limits

from llmname.artifacts import code_hash, collect_versions, file_hash, json_hash, write_json
from llmname.config import validate_config
from llmname.data.splits import make_folds
from llmname.data.validation import load_frame, smoke_sample

from .evaluation import evaluate_folds
from .pipeline import build_pipeline
from .prediction import predict_run


def train_run(
    config: dict,
    data_dir: Path,
    root: Path,
    smoke: bool = False,
    run_id: str | None = None,
    output_dir: Path | None = None,
) -> Path:
    """Menjalankan CV dan final fit dengan metadata lengkap serta status kegagalan."""
    config = validate_config(config)
    run_id = (
        run_id or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:8]
    )
    if not run_id or any(
        char not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_"
        for char in run_id
    ):
        raise ValueError("run_id hanya boleh memuat huruf, angka, tanda hubung, dan garis bawah")
    run_dir = (output_dir or root / config["experiment"]["output_dir"]) / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    start = time.perf_counter()
    manifest = {
        "run_id": run_id,
        "status": "running",
        "mode": "smoke" if smoke else "full",
        "code_hash": code_hash(root),
        "config_hash": json_hash(config),
        "config": config,
        "versions": collect_versions(),
        "started_at": datetime.now(timezone.utc).isoformat(),
    }
    write_json(run_dir / "manifest.json", manifest)
    handler = logging.FileHandler(run_dir / "run.log", encoding="utf-8")
    logger = logging.getLogger("llmname.training")
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    try:
        frame = load_frame(data_dir)
        if smoke:
            frame = smoke_sample(frame)
        n_splits = (
            min(config["experiment"]["n_splits"], 2) if smoke else config["experiment"]["n_splits"]
        )
        folds = make_folds(frame, n_splits)
        folds.to_csv(run_dir / "folds.csv", index=False)
        manifest.update(
            {
                "train_file_hash": file_hash(data_dir / "train.csv"),
                "sample_hash": json_hash({"ids": frame["id"].tolist()}),
                "fold_hash": file_hash(run_dir / "folds.csv"),
                "classes": sorted(frame["model"].unique()),
                "train_rows": len(frame),
                "n_splits": n_splits,
            }
        )
        cache_dir = root / ".cache" / "features" / manifest["code_hash"]
        logger.info("Memulai run %s pada %d respons nyata", run_id, len(frame))
        with threadpool_limits(limits=config["experiment"]["threads"]):
            metrics, fold_metrics = evaluate_folds(frame, folds, config, run_dir, cache_dir)
            model = build_pipeline(config, cache_dir)
            model.fit(frame[["text"]], frame["model"])
        # Jalur cache bukan dependensi inferensi dan tidak disertakan dalam model tersimpan.
        model.memory = None
        joblib.dump(model, run_dir / "model.joblib")
        manifest.update(
            {
                "status": "complete",
                "metrics": metrics,
                "fold_metrics": fold_metrics,
                "seconds": time.perf_counter() - start,
                "peak_process_rss_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
                "finished_at": datetime.now(timezone.utc).isoformat(),
                "files": {
                    p.name: file_hash(p)
                    for p in run_dir.iterdir()
                    if p.is_file() and p.name not in {"manifest.json", "run.log"}
                },
            }
        )
        write_json(run_dir / "manifest.json", manifest)
        predict_run(run_dir, data_dir)
        logger.info("Run selesai: %s", run_id)
    except Exception as error:
        logger.exception("Run gagal")
        manifest.update({"status": "failed", "error_type": type(error).__name__})
        write_json(run_dir / "manifest.json", manifest)
        raise
    finally:
        logger.removeHandler(handler)
        handler.close()
    return run_dir
