"""Stage 4 — Model Selection.

CatBoost and LightGBM are the two standalone selectable models. XGBoost is
never selectable on its own — it only appears as a third base learner when
``ensemble`` is ``"voting"``, ``"stacking"``, or ``"auto"``, in which case
``api.fit()`` routes to ``training/ensemble.py`` instead of calling
``select_model`` at all.
"""

from __future__ import annotations

from smarttab.analysis.dataset_analyzer import DatasetProfile
from smarttab.exceptions import UnsupportedModelError

SUPPORTED_MODELS = ("catboost", "lightgbm")

LARGE_DATASET_SAMPLE_THRESHOLD = 100_000
HIGH_CARDINALITY_THRESHOLD = 50


def select_model(profile: DatasetProfile, model: str = "auto") -> tuple[str, list[str]]:
    """Return (selected_model_name, human-readable decision notes)."""
    if model != "auto":
        if model not in SUPPORTED_MODELS:
            raise UnsupportedModelError(
                f"model must be 'auto' or one of {SUPPORTED_MODELS}, got {model!r}. "
                "'xgboost' is only available via ensemble='voting'/'stacking'/'auto'."
            )
        return model, [f"model={model!r} was explicitly requested"]

    notes: list[str] = []
    high_cardinality_columns = [c for c, card in profile.cardinality.items() if card >= HIGH_CARDINALITY_THRESHOLD]

    if profile.n_samples <= LARGE_DATASET_SAMPLE_THRESHOLD:
        notes.append(
            f"model=auto -> catboost ({profile.n_samples} samples <= {LARGE_DATASET_SAMPLE_THRESHOLD} threshold)"
        )
        return "catboost", notes

    if high_cardinality_columns:
        notes.append(
            f"model=auto -> catboost (dataset is large, but {len(high_cardinality_columns)} high-cardinality "
            f"categorical column(s) benefit from CatBoost's native categorical handling)"
        )
        return "catboost", notes

    notes.append(f"model=auto -> lightgbm ({profile.n_samples} samples > {LARGE_DATASET_SAMPLE_THRESHOLD} threshold)")
    return "lightgbm", notes
