"""Validasi ketat dataset untuk training maupun inferensi."""

from pathlib import Path

import pandas as pd


def validate_frame(frame: pd.DataFrame, training: bool = True) -> None:
    """Menolak data kosong, ID ganda, label kosong, dan teks yang tidak valid."""
    required = ["id", "text", *(["prompt_id", "model"] if training else [])]
    missing = set(required) - set(frame)
    if missing or frame.empty:
        raise ValueError(f"Dataset kosong atau kolom hilang: {sorted(missing)}")
    for name in required:
        if frame[name].isna().any() or frame[name].astype(str).str.strip().eq("").any():
            raise ValueError(f"Nilai kosong pada kolom {name}")
    if not frame["text"].map(lambda value: isinstance(value, str)).all():
        raise ValueError("text harus berupa string")
    if frame["id"].duplicated().any():
        raise ValueError("ID duplikat")
    if training:
        if frame["model"].nunique() < 2:
            raise ValueError("Training memerlukan minimal dua kelas")
        if frame.duplicated(["prompt_id", "model"]).any():
            raise ValueError("Pasangan prompt-model duplikat")


def load_frame(data_dir: str | Path, training: bool = True) -> pd.DataFrame:
    """Membaca CSV dengan ID string dan mempertahankan teks asli."""
    path = Path(data_dir) / ("train.csv" if training else "test.csv")
    frame = pd.read_csv(path, dtype={"id": str, "prompt_id": str})
    validate_frame(frame, training)
    return frame


def smoke_sample(frame: pd.DataFrame, n_prompts: int = 6) -> pd.DataFrame:
    """Memilih prompt nyata secara deterministik tanpa memotong isi teks."""
    prompts = sorted(frame["prompt_id"].unique())[:n_prompts]
    sample = frame[frame["prompt_id"].isin(prompts)].reset_index(drop=True)
    validate_frame(sample)
    return sample
