"""Orkestrasi pipeline EDA dari pemuatan data sampai laporan."""

from __future__ import annotations

import logging
import time

from .artifacts import AnalysisArtifacts
from .audit import build_dataset_audit
from .config import EDAConfig
from .features import add_features
from .io import load_datasets, write_dataframe, write_json
from .logging_utils import configure_logging, log_duration
from .modeling import run_modeling
from .plotting import (
    configure_plots,
    plot_class_distribution,
    plot_class_recall,
    plot_confusion_matrix,
    plot_family_confusion,
    plot_feature_correlation,
    plot_feature_effect_sizes,
    plot_non_ascii_characters,
    plot_prompt_variability,
    plot_punctuation_heatmap,
    plot_stylometry_heatmap,
    plot_text_length_by_model,
    plot_variance_explained,
)
from .reporting import build_report, build_summary, collect_versions, write_reports
from .stylometry import build_stylometry_analysis
from .text_analysis import (
    build_near_duplicate_analysis,
    build_non_ascii_table,
    build_opening_phrases,
    build_prompt_text_statistics,
    build_text_statistics,
    build_top_words,
)
from .validation import validate_datasets

LOGGER = logging.getLogger("llm_eda")


def _save_tables(artifacts: AnalysisArtifacts, config: EDAConfig) -> None:
    """Menyimpan seluruh tabel analisis ke folder output."""
    tables = config.tables_dir
    audit = artifacts.audit
    feature_analysis = artifacts.feature_analysis

    write_dataframe(audit.schema, tables / "schema.csv")
    write_dataframe(audit.class_distribution, tables / "class_distribution.csv")
    write_dataframe(audit.prompt_coverage, tables / "prompt_coverage.csv", index=True)
    write_dataframe(
        audit.missing_cells,
        tables / "missing_prompt_model_cells.csv",
    )
    write_dataframe(audit.rows_per_prompt, tables / "rows_per_prompt.csv")
    write_dataframe(audit.models_per_prompt, tables / "models_per_prompt.csv")

    write_dataframe(artifacts.feature_table, tables / "feature_table.csv")
    write_dataframe(feature_analysis.dictionary, tables / "feature_dictionary.csv")
    write_dataframe(
        feature_analysis.discriminability,
        tables / "feature_discriminability.csv",
    )
    write_dataframe(
        feature_analysis.means_by_model,
        tables / "feature_means_by_model.csv",
    )
    write_dataframe(
        feature_analysis.zscores_by_model,
        tables / "feature_zscores_by_model.csv",
        index=True,
    )
    write_dataframe(
        feature_analysis.correlation,
        tables / "feature_correlation.csv",
        index=True,
    )
    write_dataframe(
        feature_analysis.punctuation_means_by_model,
        tables / "punctuation_means_by_model.csv",
        index=True,
    )
    write_dataframe(
        feature_analysis.variance_explained,
        tables / "variance_explained.csv",
    )

    write_dataframe(
        artifacts.text_statistics_overall,
        tables / "text_statistics_overall.csv",
    )
    write_dataframe(
        artifacts.text_statistics_by_model,
        tables / "text_statistics_by_model.csv",
    )
    write_dataframe(
        artifacts.prompt_text_statistics,
        tables / "prompt_text_statistics.csv",
    )
    write_dataframe(artifacts.non_ascii_characters, tables / "non_ascii_characters.csv")
    write_dataframe(artifacts.top_words_overall, tables / "top_words_overall.csv")
    write_dataframe(artifacts.top_words_by_model, tables / "top_words_by_model.csv")
    write_dataframe(artifacts.opening_phrases, tables / "opening_phrases.csv")
    write_dataframe(
        artifacts.near_duplicate_pairs,
        tables / "near_duplicate_pairs.csv",
    )

    if artifacts.modeling is not None:
        modeling = artifacts.modeling
        write_dataframe(modeling.comparison, tables / "model_comparison.csv")
        write_dataframe(modeling.class_metrics, tables / "class_metrics.csv")
        write_dataframe(
            modeling.confusion_matrix,
            tables / "confusion_matrix.csv",
            index=True,
        )
        write_dataframe(modeling.confusion_pairs, tables / "confusion_pairs.csv")
        write_dataframe(
            modeling.family_confusion,
            tables / "family_confusion.csv",
            index=True,
        )
        write_dataframe(
            modeling.distinctive_terms,
            tables / "distinctive_terms.csv",
        )
        write_dataframe(
            modeling.oof_predictions,
            config.predictions_dir / "oof_predictions.csv",
        )

    write_json(
        {
            "summary": audit.summary,
            "integrity": audit.integrity,
            "imbalance": audit.imbalance,
            "near_duplicate": artifacts.near_duplicate_summary,
            "warnings": artifacts.warnings,
        },
        tables / "dataset_summary.json",
    )


def _save_figures(artifacts: AnalysisArtifacts, config: EDAConfig) -> None:
    """Membuat seluruh grafik EDA."""
    configure_plots()
    figures = config.figures_dir
    audit = artifacts.audit
    feature_analysis = artifacts.feature_analysis

    plot_class_distribution(
        audit.class_distribution,
        figures / "01_class_distribution.png",
    )
    plot_text_length_by_model(
        artifacts.feature_table,
        figures / "02_text_length_by_model.png",
    )
    plot_feature_effect_sizes(
        feature_analysis.discriminability,
        figures / "03_feature_effect_sizes.png",
        top_n=config.top_n,
    )
    plot_punctuation_heatmap(
        feature_analysis.punctuation_means_by_model,
        figures / "04_punctuation_profile.png",
    )
    plot_stylometry_heatmap(
        feature_analysis.zscores_by_model,
        figures / "05_stylometry_heatmap.png",
    )
    plot_feature_correlation(
        feature_analysis.correlation,
        figures / "06_feature_correlation.png",
    )
    plot_prompt_variability(
        artifacts.prompt_text_statistics,
        figures / "07_prompt_variability.png",
    )
    plot_variance_explained(
        feature_analysis.variance_explained,
        figures / "08_variance_explained.png",
    )
    plot_non_ascii_characters(
        artifacts.non_ascii_characters,
        figures / "09_non_ascii_characters.png",
    )

    if artifacts.modeling is not None:
        plot_class_recall(
            artifacts.modeling.class_metrics,
            figures / "10_class_recall.png",
        )
        plot_confusion_matrix(
            artifacts.modeling.confusion_matrix,
            figures / "11_confusion_matrix.png",
        )
        plot_family_confusion(
            artifacts.modeling.family_confusion,
            figures / "12_family_confusion.png",
        )


def run_eda(config: EDAConfig) -> AnalysisArtifacts:
    """Menjalankan pipeline EDA lengkap."""
    start = time.perf_counter()
    config.prepare_directories()
    logger = configure_logging(config.log_level, config.log_file)

    logger.info("Memulai EDA Name That LLM.")
    logger.info("Folder data: %s", config.data_dir)
    logger.info("Folder output: %s", config.output_dir)

    versions = collect_versions()
    bundle = load_datasets(config)
    validate_datasets(bundle)

    warnings = list(bundle.load_warnings)

    with log_duration(logger, "Ekstraksi fitur teks dan gaya"):
        feature_table = add_features(bundle.train)

    with log_duration(logger, "Audit struktur dan integritas dataset"):
        audit = build_dataset_audit(bundle)

    with log_duration(logger, "Analisis statistik teks"):
        text_overall, text_by_model = build_text_statistics(feature_table)
        prompt_text_statistics = build_prompt_text_statistics(feature_table)
        non_ascii_characters = build_non_ascii_table(feature_table, config.top_n)
        top_words_overall, top_words_by_model = build_top_words(feature_table, config.top_n)
        opening_phrases = build_opening_phrases(feature_table, top_n=3)

    with log_duration(logger, "Analisis near-duplicate"):
        near_duplicate_pairs, near_duplicate_summary = build_near_duplicate_analysis(
            feature_table, config
        )

    with log_duration(logger, "Analisis stylometri dan diskriminabilitas"):
        feature_analysis = build_stylometry_analysis(feature_table, config)

    modeling = None
    if config.skip_modeling:
        logger.info("Evaluasi baseline dilewati atas permintaan pengguna.")
    else:
        with log_duration(logger, "Evaluasi baseline dan analisis kesalahan"):
            modeling = run_modeling(feature_table, config)

    artifacts = AnalysisArtifacts(
        audit=audit,
        feature_table=feature_table,
        feature_analysis=feature_analysis,
        text_statistics_overall=text_overall,
        text_statistics_by_model=text_by_model,
        prompt_text_statistics=prompt_text_statistics,
        non_ascii_characters=non_ascii_characters,
        top_words_overall=top_words_overall,
        top_words_by_model=top_words_by_model,
        opening_phrases=opening_phrases,
        near_duplicate_pairs=near_duplicate_pairs,
        near_duplicate_summary=near_duplicate_summary,
        modeling=modeling,
        warnings=warnings,
    )

    with log_duration(logger, "Penyimpanan tabel"):
        _save_tables(artifacts, config)

    with log_duration(logger, "Pembuatan grafik"):
        _save_figures(artifacts, config)

    runtime_seconds = time.perf_counter() - start
    with log_duration(logger, "Penyusunan laporan"):
        report_markdown = build_report(
            artifacts,
            config,
            bundle,
            runtime_seconds,
            versions,
        )
        write_reports(report_markdown, config)
        summary = build_summary(artifacts, config, runtime_seconds, versions)
        write_json(summary, config.output_dir / "summary.json")

    logger.info("EDA selesai dalam %.2f detik.", runtime_seconds)
    logger.info("Laporan HTML: %s", config.output_dir / "report.html")
    logger.info("Ringkasan JSON: %s", config.output_dir / "summary.json")
    return artifacts
