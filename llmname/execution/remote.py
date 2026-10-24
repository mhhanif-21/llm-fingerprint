"""Pemanggilan pipeline yang sama pada runtime Kaggle."""

from pathlib import Path

from llmname.artifacts import code_hash, file_hash, json_hash
from llmname.modeling.training import train_run


def run_remote(request: dict, source_dir: Path) -> Path:
    """Memverifikasi kode dan konfigurasi sebelum smoke atau training remote."""
    if code_hash(source_dir) != request["code_hash"]:
        raise ValueError("Fingerprint sumber remote tidak cocok")
    if json_hash(request["config"]) != request["config_hash"]:
        raise ValueError("Fingerprint konfigurasi remote tidak cocok")
    candidates = [
        path.parent
        for path in Path("/kaggle/input").rglob("train.csv")
        if (path.parent / "test.csv").is_file() and "name-that-llm" in str(path.parent)
    ]
    if len(candidates) != 1:
        raise FileNotFoundError("Dataset name-that-llm tunggal tidak ditemukan")
    if (
        request.get("expected_train_hash")
        and file_hash(candidates[0] / "train.csv") != request["expected_train_hash"]
    ):
        raise ValueError("Fingerprint data kompetisi berbeda dari data lokal")
    return train_run(
        request["config"],
        candidates[0],
        source_dir,
        smoke=request["mode"] == "smoke",
        run_id=request["run_id"],
        output_dir=Path("/kaggle/working"),
    )
