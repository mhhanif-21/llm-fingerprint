"""Pemeriksaan penggantian estimator dan parameter melalui factory."""

import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC

from llmname.modeling.factory import build_model


def test_model_switch_requires_no_runner_change():
    """Membuat dua estimator nyata melalui kontrak yang sama."""
    assert isinstance(build_model({"name": "linear_svc", "params": {"C": 2}}, 42), LinearSVC)
    assert isinstance(
        build_model({"name": "logistic_regression", "params": {}}, 42), LogisticRegression
    )


@pytest.mark.parametrize(
    "options",
    [
        {"name": "unknown", "params": {}},
        {"name": "linear_svc", "params": {"C": -1}},
        {"name": "linear_svc", "params": {"typo": 1}},
    ],
)
def test_invalid_model_parameters_fail(options):
    """Menolak nama estimator maupun parameter yang tidak didukung."""
    with pytest.raises(ValueError):
        build_model(options, 42)
