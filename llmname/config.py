"""Pembacaan konfigurasi eksperimen dengan skema yang eksplisit."""

import tomllib
from copy import deepcopy
from pathlib import Path

SCHEMA = {
    "experiment": {"seed", "n_splits", "output_dir", "threads"},
    "model": {"name", "params"},
    "features": {"branches", "cache", "word", "char", "style"},
}


def load_config(path: str | Path) -> dict:
    """Membaca TOML dan menolak bagian maupun opsi yang tidak dikenal."""
    with Path(path).open("rb") as stream:
        config = tomllib.load(stream)
    return validate_config(config)


def validate_config(config: dict) -> dict:
    """Memastikan konfigurasi lengkap dan memiliki nilai operasional yang valid."""
    if set(config) != set(SCHEMA):
        raise ValueError(f"Bagian konfigurasi wajib: {sorted(SCHEMA)}")
    result = deepcopy(config)
    for section, allowed in SCHEMA.items():
        if not isinstance(result[section], dict) or set(result[section]) - allowed:
            raise ValueError(f"Opsi tidak dikenal pada {section}")
    experiment = result["experiment"]
    if set(experiment) != SCHEMA["experiment"]:
        raise ValueError("Parameter experiment tidak lengkap")
    for field in ["seed", "n_splits", "threads"]:
        if type(experiment[field]) is not int:
            raise ValueError(f"{field} harus integer")
    if experiment["n_splits"] < 2 or experiment["threads"] < 1 or experiment["seed"] < 0:
        raise ValueError("Fold minimal dua, threads positif, dan seed nonnegatif")
    if not isinstance(experiment["output_dir"], str) or not experiment["output_dir"]:
        raise ValueError("output_dir wajib terisi")
    if set(result["model"]) != SCHEMA["model"] or not isinstance(result["model"]["params"], dict):
        raise ValueError("Konfigurasi model wajib memuat name dan params")
    features = result["features"]
    branches = features.get("branches", [])
    if not isinstance(branches, list) or not branches or len(set(branches)) != len(branches):
        raise ValueError("branches wajib berupa daftar unik yang tidak kosong")
    if set(branches) - {"word", "char", "style"}:
        raise ValueError("Kelompok fitur tidak dikenal")
    if type(features.get("cache", False)) is not bool:
        raise ValueError("cache harus boolean")
    for name in branches:
        if not isinstance(features.get(name), dict):
            raise ValueError(f"Konfigurasi fitur {name} wajib diisi")
    return result
