"""Validasi ID dan kelas submission menggunakan test publik nyata."""

import pandas as pd
import pytest

from llmname.modeling.prediction import validate_submission


def test_real_sample_submission_contract(data_dir):
    """Menerima sample submission asli dan menolak urutan ID atau label yang rusak."""
    test = pd.read_csv(data_dir / "test.csv", dtype={"id": str})
    sample = pd.read_parquet(data_dir / "sample_submission.parquet")
    sample["id"] = sample["id"].astype(str)
    classes = pd.read_csv(data_dir / "train.csv", usecols=["model"])["model"].unique()
    sample = test[["id"]].merge(sample, on="id", how="left")
    validate_submission(sample, test, classes)
    with pytest.raises(ValueError, match="ID"):
        validate_submission(sample.iloc[::-1], test, classes)
    sample.loc[0, "model"] = ""
    with pytest.raises(ValueError, match="Label"):
        validate_submission(sample, test, classes)
