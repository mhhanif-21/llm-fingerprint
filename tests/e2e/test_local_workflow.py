"""Smoke E2E lokal pada prompt nyata tanpa eksperimen baseline penuh."""

from pathlib import Path

import pandas as pd
import pytest

from llmname.artifacts import read_json, verify_files
from llmname.modeling.prediction import predict_run
from llmname.modeling.training import train_run

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.e2e
def test_local_train_save_load_predict(config, data_dir, tmp_path):
    """Menguji dua fold, final fit kecil, persistensi, inferensi, dan integritas hasil."""
    run = train_run(config, data_dir, ROOT, smoke=True, run_id="local-e2e", output_dir=tmp_path)
    manifest = read_json(run / "manifest.json")
    assert manifest["mode"] == "smoke"
    assert manifest["status"] == "complete"
    assert manifest["n_splits"] == 2
    assert manifest["train_rows"] < len(pd.read_csv(data_dir / "train.csv"))
    assert len(manifest["fold_metrics"]) == 2
    verify_files(run, manifest["files"])
    first = pd.read_csv(run / "submission.csv", dtype={"id": str})
    second = predict_run(run, data_dir)
    pd.testing.assert_frame_equal(first, second, check_dtype=False)
    folds = pd.read_csv(run / "folds.csv")
    assert folds.groupby("prompt_id")["fold"].nunique().eq(1).all()
