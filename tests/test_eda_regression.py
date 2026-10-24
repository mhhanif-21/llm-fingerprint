"""Regresi numerik terhadap fitur sebelum refactor pada data nyata."""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from llmname.artifacts import file_hash
from llmname.eda.features import add_features


def test_legacy_feature_values_are_preserved(data_dir):
    """Membandingkan seluruh fitur delapan respons dengan hasil kode lama."""
    reference = json.loads((Path(__file__).parent / "fixtures/legacy_features.json").read_text())
    assert file_hash(data_dir / "train.csv") == reference["source_hash"]
    frame = pd.read_csv(data_dir / "train.csv").head(8)
    result = add_features(frame)
    for name, expected in reference["values"].items():
        np.testing.assert_allclose(result[name], expected, atol=1e-10, rtol=1e-10)
