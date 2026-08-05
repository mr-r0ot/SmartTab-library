"""Native and SHAP feature importance helpers."""

from __future__ import annotations

import numpy as np
import pandas as pd

from smarttab.exceptions import UnsupportedModelError
from smarttab.logging_utils import get_logger

logger = get_logger()
SHAP_SAMPLE_SIZE = 200


def _normalized(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    total = float(np.abs(result["importance"]).sum())
    if total > 0:
        result["importance"] = result["importance"] / total
    return result


def get_feature_importance(estimator, model_name: str, feature_names: list[str]) -> pd.DataFrame:
    if hasattr(estimator, "base_models"):
        parts: list[pd.DataFrame] = []
        for entry in estimator.base_models:
            if len(entry) == 4:
                _alias, base_name, base_estimator, member_features = entry
            elif len(entry) == 3:
                _alias, base_name, base_estimator = entry
                member_features = None
            else:  # compatibility with very early in-memory objects
                base_name, base_estimator = entry
                member_features = None
            names = list(member_features) if member_features is not None else feature_names
            parts.append(_normalized(get_feature_importance(base_estimator, base_name, names)))
        combined = pd.concat(parts, ignore_index=True).groupby("feature", as_index=False)["importance"].mean()
        return combined.sort_values("importance", ascending=False, ignore_index=True)

    if hasattr(estimator, "estimators_") and model_name in {"catboost", "lightgbm", "xgboost"}:
        parts = [
            _normalized(get_feature_importance(sub_estimator, model_name, feature_names))
            for sub_estimator in estimator.estimators_
        ]
        combined = pd.concat(parts, ignore_index=True).groupby("feature", as_index=False)["importance"].mean()
        return combined.sort_values("importance", ascending=False, ignore_index=True)

    if model_name == "catboost":
        importances = estimator.get_feature_importance(type="PredictionValuesChange")
    elif model_name in {"lightgbm", "xgboost"}:
        importances = estimator.feature_importances_
    else:
        raise UnsupportedModelError(f"Unknown model_name {model_name!r}")

    values = np.asarray(importances, dtype=float).reshape(-1)
    if len(values) != len(feature_names):
        raise ValueError(
            f"feature importance length mismatch: estimator returned {len(values)}, "
            f"but the fitted schema has {len(feature_names)} features"
        )
    return pd.DataFrame({"feature": feature_names, "importance": values}).sort_values(
        "importance", ascending=False, ignore_index=True
    )


def get_shap_importance(estimator, model_name: str, X_sample: pd.DataFrame) -> pd.DataFrame | None:
    """Return mean absolute SHAP values, aggregated across classes when required."""
    if hasattr(estimator, "base_models") or hasattr(estimator, "estimators_"):
        return None
    if model_name not in {"catboost", "lightgbm", "xgboost"}:
        return None
    try:
        import shap

        sample = (
            X_sample.sample(n=SHAP_SAMPLE_SIZE, random_state=42)
            if len(X_sample) > SHAP_SAMPLE_SIZE
            else X_sample
        )
        values = shap.TreeExplainer(estimator).shap_values(sample)
        if isinstance(values, list):
            arrays = [np.asarray(value, dtype=float) for value in values]
            mean_abs = np.mean([np.abs(value).mean(axis=0) for value in arrays], axis=0)
        else:
            array = np.asarray(values, dtype=float)
            if array.ndim == 2:
                mean_abs = np.abs(array).mean(axis=0)
            elif array.ndim == 3:
                # SHAP versions differ between (samples, features, classes) and
                # (samples, classes, features). Detect the feature axis by size.
                if array.shape[1] == sample.shape[1]:
                    mean_abs = np.abs(array).mean(axis=(0, 2))
                elif array.shape[2] == sample.shape[1]:
                    mean_abs = np.abs(array).mean(axis=(0, 1))
                else:
                    raise ValueError(f"unrecognized SHAP tensor shape {array.shape}")
            else:
                raise ValueError(f"unrecognized SHAP value shape {array.shape}")
        return pd.DataFrame(
            {"feature": list(sample.columns), "mean_abs_shap": np.asarray(mean_abs).reshape(-1)}
        ).sort_values("mean_abs_shap", ascending=False, ignore_index=True)
    except Exception as exc:
        logger.warning("SHAP importance unavailable for %s: %s", model_name, exc)
        return None
