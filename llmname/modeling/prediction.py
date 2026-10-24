"""Inferensi model tersimpan dan validasi submission tanpa penggantian label."""

from pathlib import Path

import joblib
import pandas as pd

from llmname.artifacts import collect_versions, file_hash, read_json, verify_files, write_json
from llmname.data.validation import load_frame


def validate_submission(submission: pd.DataFrame, test: pd.DataFrame, classes) -> None:
    """Memastikan skema, urutan ID, dan seluruh label prediksi sesuai kontrak."""
    if list(submission) != ["id", "model"] or len(submission) != len(test):
        raise ValueError("Skema atau jumlah baris submission tidak sesuai")
    if submission["id"].tolist() != test["id"].tolist() or submission["id"].duplicated().any():
        raise ValueError("ID submission tidak sesuai urutan test")
    if submission["model"].isna().any() or not set(submission["model"]).issubset(set(classes)):
        raise ValueError("Label submission tidak valid")


def predict_run(run_dir: Path, data_dir: Path, check_versions: bool = True) -> pd.DataFrame:
    """Memuat model tepercaya buatan proyek setelah memeriksa hash dan runtime."""
    manifest_path = run_dir / "manifest.json"
    manifest = read_json(manifest_path)
    if manifest.get("status") != "complete":
        raise ValueError("Run belum selesai")
    verify_files(run_dir, manifest["files"])
    if (
        check_versions
        and manifest["versions"]["scikit-learn"] != collect_versions()["scikit-learn"]
    ):
        raise ValueError("Versi scikit-learn berbeda; gunakan environment yang sama dengan model")
    test = load_frame(data_dir, training=False)
    # Joblib hanya dimuat dari run sendiri; checksum bukan mekanisme kepercayaan pihak ketiga.
    model = joblib.load(run_dir / "model.joblib")
    prediction = pd.DataFrame({"id": test["id"], "model": model.predict(test[["text"]])})
    validate_submission(prediction, test, manifest["classes"])
    prediction.to_csv(run_dir / "submission.csv", index=False)
    prediction.to_parquet(run_dir / "submission.parquet", index=False)
    manifest["prediction_data_hash"] = file_hash(data_dir / "test.csv")
    for name in ["submission.csv", "submission.parquet"]:
        manifest["files"][name] = file_hash(run_dir / name)
    write_json(manifest_path, manifest)
    return prediction
