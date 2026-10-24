"""Pembagian grup yang dapat ditelusuri dan diperiksa ulang."""

import pandas as pd
from sklearn.model_selection import GroupKFold


def make_folds(frame: pd.DataFrame, n_splits: int) -> pd.DataFrame:
    """Menyimpan satu fold validasi untuk setiap ID dengan grup yang terpisah."""
    if n_splits < 2 or frame["prompt_id"].nunique() < n_splits:
        raise ValueError("Jumlah prompt tidak cukup untuk fold yang diminta")
    folds = frame[["id", "prompt_id", "model"]].copy()
    folds["fold"] = -1
    splitter = GroupKFold(n_splits=n_splits)
    for fold, (train_indices, valid_indices) in enumerate(
        splitter.split(frame, frame["model"], frame["prompt_id"])
    ):
        training = frame.iloc[train_indices]
        validation = frame.iloc[valid_indices]
        if set(training["prompt_id"]) & set(validation["prompt_id"]):
            raise ValueError("Kebocoran grup pada pembagian fold")
        if set(training["model"]) != set(frame["model"]):
            raise ValueError("Fold training kehilangan kelas")
        folds.iloc[valid_indices, folds.columns.get_loc("fold")] = fold
    if folds["fold"].lt(0).any():
        raise ValueError("Ada baris yang tidak mendapatkan fold")
    return folds
