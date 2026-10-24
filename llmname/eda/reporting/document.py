"""Penyusunan laporan berdasarkan hasil komputasi tanpa klaim statis."""

from datetime import datetime, timezone

import pandas as pd

from ..artifacts import AnalysisArtifacts, DatasetBundle
from ..config import EDAConfig
from .tables import markdown_table


def build_report(
    artifacts: AnalysisArtifacts,
    config: EDAConfig,
    bundle: DatasetBundle,
    runtime_seconds: float,
    versions: dict[str, str],
) -> str:
    """Merangkai tabel dan grafik yang benar-benar tersedia pada run EDA."""
    sections = [
        "# EDA Name That LLM",
        f"Dibuat: {datetime.now(timezone.utc).isoformat()}; durasi {runtime_seconds:.2f} detik.",
        "## Ringkasan dataset",
        markdown_table(pd.DataFrame([artifacts.audit.summary])),
        "## Integritas",
        markdown_table(pd.DataFrame([artifacts.audit.integrity]).astype(str)),
        "## Distribusi kelas",
        markdown_table(artifacts.audit.class_distribution),
        "## Panjang dan statistik teks",
        markdown_table(artifacts.text_statistics_overall),
        "## Fitur gaya",
        "Peringkat ini bersifat eksploratif pada seluruh train dan bukan estimasi manfaat OOF.",
        markdown_table(artifacts.feature_analysis.discriminability.head(config.top_n)),
        "## Pengaruh model dan prompt",
        "R kuadrat dihitung in-sample dan hanya bersifat deskriptif.",
        markdown_table(artifacts.feature_analysis.variance_explained),
        "## Near-duplicate",
        markdown_table(pd.DataFrame([artifacts.near_duplicate_summary])),
        "## Evaluasi model",
    ]
    if artifacts.modeling is None:
        sections.append("Modeling tidak dijalankan; tidak ada skor baru pada run ini.")
    else:
        sections.extend(
            [
                markdown_table(artifacts.modeling.comparison),
                "Perbandingan random/group adalah observasi, bukan bukti otomatis arah kebocoran.",
                "Konfigurasi dipilih pada CV yang sama; skor terbaik dapat optimistis akibat seleksi.",
                markdown_table(artifacts.modeling.class_metrics),
            ]
        )
    sections.append("## Grafik")
    for figure in sorted(config.figures_dir.glob("*.png")):
        sections.append(f"![{figure.stem}](figures/{figure.name})")
    sections.extend(
        [
            "## Keterbatasan",
            f"Test publik berisi {len(bundle.test)} baris; distribusi hidden test tidak diketahui.",
            "Analisis statistik global tidak boleh digunakan untuk seleksi fitur di luar fold training.",
            "Fitur regex bergantung pada bahasa dan heuristik; panjang relatif per prompt hanya EDA.",
            "## Reproduksibilitas",
            markdown_table(pd.DataFrame(sorted(versions.items()), columns=["package", "version"])),
            f"Data: `{config.data_dir}`; seed: `{config.seed}`.",
        ]
    )
    return "\n\n".join(sections) + "\n"
