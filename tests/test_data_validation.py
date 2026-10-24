"""Validasi data nyata dan salinan input yang sengaja dibuat invalid."""

import pytest

from llmname.data.validation import validate_frame


def test_real_data_is_accepted(real_frame):
    """Menerima sampel prompt nyata yang skema dan labelnya lengkap."""
    validate_frame(real_frame)


@pytest.mark.parametrize(
    "problem", ["missing_column", "duplicate_id", "empty_text", "missing_label", "duplicate_pair"]
)
def test_rejects_invalid_data(real_frame, problem):
    """Menolak salinan data nyata dengan satu pelanggaran kontrak."""
    frame = real_frame.copy()
    if problem == "missing_column":
        frame = frame.drop(columns="text")
    elif problem == "duplicate_id":
        frame.loc[1, "id"] = frame.loc[0, "id"]
    elif problem == "empty_text":
        frame.loc[0, "text"] = " "
    elif problem == "missing_label":
        frame.loc[0, "model"] = None
    else:
        frame.loc[1, ["prompt_id", "model"]] = frame.loc[0, ["prompt_id", "model"]].values
    with pytest.raises(ValueError):
        validate_frame(frame)
