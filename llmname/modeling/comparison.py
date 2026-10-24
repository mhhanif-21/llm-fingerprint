"""Perbandingan run dengan pemeriksaan kesetaraan data dan skema evaluasi."""

from pathlib import Path

import pandas as pd

from llmname.artifacts import read_json


def compare_runs(directory: Path, output_path: Path) -> pd.DataFrame:
    """Mengelompokkan eksperimen yang sebanding tanpa mencampur smoke dan training penuh."""
    rows = []
    for path in sorted(directory.glob("*/manifest.json")):
        manifest = read_json(path)
        if manifest.get("status") != "complete":
            continue
        group = ":".join(
            str(manifest[name]) for name in ["train_file_hash", "sample_hash", "fold_hash", "mode"]
        )
        rows.append(
            {
                "run_id": manifest["run_id"],
                "comparison_group": group,
                "model": manifest["config"]["model"]["name"],
                "mode": manifest["mode"],
                "config_hash": manifest["config_hash"],
                "seconds": manifest["seconds"],
                **manifest["metrics"],
            }
        )
    if not rows:
        raise ValueError("Tidak ada run selesai untuk dibandingkan")
    result = pd.DataFrame(rows).sort_values(
        ["comparison_group", "balanced_accuracy"], ascending=[True, False]
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_path, index=False)
    return result
