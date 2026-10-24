"""Factory estimator yang kompatibel dengan matriks fitur sparse."""

from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC

MODEL_FACTORIES = {
    "linear_svc": LinearSVC,
    "logistic_regression": LogisticRegression,
}


def build_model(options: dict, seed: int):
    """Membuat estimator terdaftar dan menolak parameter yang tidak dikenal."""
    name = options["name"]
    if name not in MODEL_FACTORIES:
        raise ValueError(f"Model tidak dikenal: {name}")
    estimator = MODEL_FACTORIES[name](random_state=seed)
    params = options.get("params", {})
    if set(params) - set(estimator.get_params()):
        raise ValueError(f"Parameter {name} tidak dikenal")
    if "C" in params and (type(params["C"]) not in (int, float) or params["C"] <= 0):
        raise ValueError("C harus positif")
    return estimator.set_params(**params)
