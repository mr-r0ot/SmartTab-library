"""Heuristic single-model selection between CatBoost and LightGBM."""

from __future__ import annotations

from smarttab.analysis.dataset_analyzer import DatasetProfile
from smarttab.exceptions import UnsupportedModelError

SUPPORTED_MODELS = ("catboost", "lightgbm")
VERY_LARGE_ROWS = 250_000
LARGE_MATRIX_CELLS = 30_000_000
WIDE_DATASET_FEATURES = 300
CATBOOST_ROW_CAP = 200_000
HIGH_CARDINALITY_THRESHOLD = 50


def select_model(profile: DatasetProfile, model: str = "auto") -> tuple[str, list[str]]:
    if model != "auto":
        if model not in SUPPORTED_MODELS:
            raise UnsupportedModelError(
                f"model must be 'auto' or one of {SUPPORTED_MODELS}; XGBoost is conditional ensemble-only"
            )
        return model, [f"model={model!r} explicitly requested"]

    n_rows = profile.n_samples
    n_features = max(profile.n_features, 1)
    cells = n_rows * n_features
    n_categorical = len(profile.categorical_columns)
    categorical_ratio = n_categorical / n_features
    high_cardinality = [
        column for column, cardinality in profile.cardinality.items()
        if cardinality >= HIGH_CARDINALITY_THRESHOLD
    ]

    if n_rows >= VERY_LARGE_ROWS:
        return "lightgbm", [
            f"model=auto -> lightgbm ({n_rows:,} rows exceeds the large-data safety threshold)"
        ]
    if cells >= LARGE_MATRIX_CELLS or n_features >= WIDE_DATASET_FEATURES:
        return "lightgbm", [
            f"model=auto -> lightgbm (matrix size={cells:,} cells, features={n_features})"
        ]
    if profile.task_type.is_ranking and n_rows > 75_000:
        return "lightgbm", ["model=auto -> lightgbm (large ranking dataset)"]

    categorical_signal = categorical_ratio >= 0.08 or bool(high_cardinality)
    if categorical_signal and n_rows <= CATBOOST_ROW_CAP:
        reason = (
            f"{n_categorical} categorical feature(s), "
            f"{len(high_cardinality)} high-cardinality"
        )
        return "catboost", [f"model=auto -> catboost ({reason})"]

    return "lightgbm", [
        "model=auto -> lightgbm (numeric/low-categorical data or CatBoost would add unnecessary cost)"
    ]
