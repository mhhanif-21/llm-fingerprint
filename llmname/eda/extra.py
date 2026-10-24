"""Analisis tambahan untuk memisahkan pola model, panjang, dan pengaruh prompt."""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from llmname.artifacts import write_json
from llmname.data.validation import load_frame
from llmname.features.candidates import CANDIDATE_DESCRIPTIONS, candidate_frame

from .features import FEATURE_COLUMNS, add_features
from .statistics import eta_squared


def safe_spearman(first: pd.Series, second: pd.Series) -> float:
    """Menyatakan korelasi tidak terdefinisi ketika salah satu variabel konstan."""
    if first.nunique() < 2 or second.nunique() < 2:
        return float("nan")
    return first.corr(second, method="spearman")


def analyze_extra(frame: pd.DataFrame, output_dir: Path) -> dict:
    """Menghasilkan diagnostik fitur yang bersifat eksploratif, bukan seleksi CV."""
    output_dir.mkdir(parents=True, exist_ok=True)
    features = add_features(frame)
    candidates = candidate_frame(frame["text"])
    features[list(candidates)] = candidates.to_numpy()
    columns = [*FEATURE_COLUMNS, *CANDIDATE_DESCRIPTIONS]
    rows = []
    prompts = sorted(frame["prompt_id"].unique())
    halves = [features[features["prompt_id"].isin(prompts[offset::2])] for offset in [0, 1]]
    for name in columns:
        residual = features[name] - features.groupby("prompt_id")[name].transform("mean")
        profiles = [part.groupby("model")[name].mean() for part in halves]
        stability = safe_spearman(profiles[0], profiles[1])
        rows.append(
            {
                "feature": name,
                "unique_values": int(features[name].nunique()),
                "eta_squared": eta_squared(features, name, "model"),
                "prompt_centered_eta_squared": eta_squared(
                    pd.DataFrame({"value": residual, "model": frame["model"]}), "value", "model"
                ),
                "word_length_spearman": safe_spearman(features[name], features["n_words"]),
                "prompt_half_profile_spearman": stability,
                "candidate": name in CANDIDATE_DESCRIPTIONS,
            }
        )
    diagnostics = pd.DataFrame(rows).sort_values("prompt_centered_eta_squared", ascending=False)
    diagnostics.to_csv(output_dir / "feature_diagnostics.csv", index=False)
    correlation = features[columns].corr(method="spearman")
    redundant = [
        {"first": first, "second": second, "spearman": correlation.loc[first, second]}
        for i, first in enumerate(columns)
        for second in columns[i + 1 :]
        if abs(correlation.loc[first, second]) >= 0.90
    ]
    pd.DataFrame(redundant, columns=["first", "second", "spearman"]).to_csv(
        output_dir / "redundant_features.csv", index=False
    )
    # Statistik per prompt memakai respons sejawat sehingga tidak masuk transformer inferensi.
    words = features["n_words"]
    median = features.groupby("prompt_id")["n_words"].transform("median")
    features["relative_length"] = words.div(median.where(median.gt(0)))
    length_summary = (
        features.groupby("model")
        .agg(
            responses=("text", "size"),
            median_words=("n_words", "median"),
            median_relative_length=("relative_length", "median"),
            relative_q25=("relative_length", lambda values: values.quantile(0.25)),
            relative_q75=("relative_length", lambda values: values.quantile(0.75)),
        )
        .sort_values("median_relative_length")
    )
    length_summary.to_csv(output_dir / "prompt_normalized_length.csv")
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.barh(length_summary.index, length_summary["median_relative_length"])
    ax.axvline(1, color="orange", linestyle="--")
    ax.set(title="Panjang respons relatif dalam prompt", xlabel="Median rasio jumlah kata")
    fig.tight_layout()
    fig.savefig(output_dir / "prompt_normalized_length.png", dpi=130)
    plt.close(fig)
    labels = sorted(frame["model"].unique(), key=len, reverse=True)
    mentions = []
    for label in labels:
        mask = frame["text"].str.contains(label, case=False, regex=False)
        for index in frame.index[mask]:
            mentions.append(
                {
                    "id": frame.loc[index, "id"],
                    "mentioned_label": label,
                    "actual_label": frame.loc[index, "model"],
                }
            )
    pd.DataFrame(mentions, columns=["id", "mentioned_label", "actual_label"]).to_csv(
        output_dir / "label_mentions.csv", index=False
    )
    normalized = frame["text"].str.casefold().str.replace(r"\s+", " ", regex=True).str.strip()
    duplicate_rows = frame.loc[normalized.duplicated(keep=False), ["id", "prompt_id", "model"]]
    duplicate_rows.to_csv(output_dir / "normalized_duplicate_rows.csv", index=False)
    summary = {
        "train_rows": len(frame),
        "feature_count": len(columns),
        "constant_features": diagnostics.loc[
            diagnostics["unique_values"].le(1), "feature"
        ].tolist(),
        "redundant_pairs": len(redundant),
        "label_mentions": len(mentions),
        "normalized_duplicate_rows": len(duplicate_rows),
        "top_prompt_centered_features": diagnostics["feature"].head(10).tolist(),
        "scope": "descriptive_train_only",
        "training_executed": False,
    }
    write_json(output_dir / "summary.json", summary)
    report = [
        "# EDA tambahan",
        "Analisis deskriptif pada seluruh train; belum membuktikan peningkatan OOF.",
        "Pemusatan per prompt bukan fitur inferensi atau model kausal.",
        "Stabilitas dua belahan prompt adalah diagnostik kasar, bukan confidence interval.",
        "![Panjang relatif](prompt_normalized_length.png)",
        diagnostics.head(15).to_string(index=False),
        "Kandidat baru tetap nonaktif sampai fase ablation yang disetujui.",
    ]
    (output_dir / "report.md").write_text("\n\n".join(report) + "\n")
    return summary


def main() -> None:
    """Menjalankan analisis tambahan dari lokasi data yang diberikan."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    print(analyze_extra(load_frame(args.data_dir), args.output_dir))


if __name__ == "__main__":
    main()
