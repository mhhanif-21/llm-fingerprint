"""Fixture berbasis data kompetisi nyata yang tidak disertakan dalam Git."""

from pathlib import Path

import pytest

from llmname.config import load_config
from llmname.data.validation import load_frame, smoke_sample

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="session")
def data_dir():
    """Menunjuk data lokal yang diperlukan untuk verifikasi nyata."""
    path = ROOT / "name-that-llm"
    if not (path / "train.csv").is_file():
        pytest.fail("Data nyata diperlukan; siapkan train.csv dan test.csv pada name-that-llm/")
    return path


@pytest.fixture(scope="session")
def real_frame(data_dir):
    """Memilih enam prompt lengkap secara deterministik dari data nyata."""
    return smoke_sample(load_frame(data_dir))


@pytest.fixture
def config():
    """Membaca konfigurasi baseline tanpa melakukan training penuh."""
    return load_config(ROOT / "configs/baseline_svc.toml")


@pytest.fixture(scope="session")
def trained_run(data_dir, tmp_path_factory):
    """Menyediakan satu hasil smoke nyata untuk pengujian artefak dan perbandingan."""
    from llmname.modeling.training import train_run

    config = load_config(ROOT / "configs/baseline_svc.toml")
    return train_run(
        config,
        data_dir,
        ROOT,
        smoke=True,
        run_id="artifact-tests",
        output_dir=tmp_path_factory.mktemp("training"),
    )
