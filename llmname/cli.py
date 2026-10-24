"""Antarmuka perintah eksperimen, diagnostik, dan eksekusi Kaggle."""

import argparse
from pathlib import Path

from llmname.artifacts import read_json
from llmname.config import load_config
from llmname.execution.workflow import fetch, load_target, status, submit, wait_and_fetch
from llmname.modeling.comparison import compare_runs
from llmname.modeling.prediction import predict_run
from llmname.modeling.training import train_run


def diagnose(run_dir: Path, root: Path) -> Path:
    """Menggabungkan metadata dan log run yang tersensor tanpa membaca kredensial."""
    from datetime import datetime, timezone

    from llmname.execution.client import redact

    if not run_dir.is_dir() or run_dir.resolve() == (root / "artifacts/runs").resolve():
        raise ValueError("RUN_DIR harus menunjuk pada satu run yang tersedia")
    chunks = []
    for name in ["manifest.json", "remote.json", "run.log", "kaggle.log"]:
        path = run_dir / name
        if path.is_file():
            chunks.append(f"File: {name}\n{redact(path.read_text()[-20000:])}")
    if not chunks:
        raise FileNotFoundError("Tidak ada metadata atau log pada run tersebut")
    output = (
        root
        / "diagnostics"
        / ("diagnose-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + ".log")
    )
    output.parent.mkdir(exist_ok=True)
    output.write_text("\n\n".join(chunks))
    return output


def build_parser() -> argparse.ArgumentParser:
    """Menentukan perintah dan opsi yang didukung pipeline."""
    parser = argparse.ArgumentParser(description="Eksperimen atribusi model pembuat teks")
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ["train", "kaggle-run"]:
        command = commands.add_parser(name)
        command.add_argument("--config", type=Path, required=True)
        command.add_argument("--smoke", action="store_true")
        if name == "train":
            command.add_argument("--data-dir", type=Path, required=True)
        else:
            command.add_argument("--target", type=Path, default=Path("configs/kaggle.local.toml"))
    for name in ["predict", "kaggle-status", "kaggle-fetch", "diagnose"]:
        command = commands.add_parser(name)
        command.add_argument("--run-dir", type=Path, required=True)
        if name == "predict":
            command.add_argument("--data-dir", type=Path, required=True)
    comparison = commands.add_parser("compare")
    comparison.add_argument("--runs-dir", type=Path, default=Path("artifacts/runs"))
    return parser


def main() -> int:
    """Menjalankan satu perintah dengan lokasi root yang mengikuti paket sumber."""
    args = build_parser().parse_args()
    root = Path(__file__).resolve().parents[1]
    if args.command == "train":
        print(train_run(load_config(args.config), args.data_dir, root, args.smoke))
    elif args.command == "predict":
        print(predict_run(args.run_dir, args.data_dir).to_string(index=False))
    elif args.command == "kaggle-run":
        directory = submit(root, load_config(args.config), load_target(args.target), args.smoke)
        print(f"Remote run: {directory}", flush=True)
        print(wait_and_fetch(directory)["run_id"])
    elif args.command == "kaggle-status":
        print(status(args.run_dir))
        print(read_json(args.run_dir / "remote.json")["status"])
    elif args.command == "kaggle-fetch":
        print(fetch(args.run_dir)["run_id"])
    elif args.command == "compare":
        comparison = compare_runs(args.runs_dir, args.runs_dir.parent / "comparison.csv")
        columns = ["run_id", "model", "mode", "seconds", "balanced_accuracy", "macro_f1"]
        print(comparison[columns].to_string(index=False))
    else:
        print(diagnose(args.run_dir, root))
    return 0
