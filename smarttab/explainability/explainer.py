"""Stage 8 — Explainability: native feature importance + SHAP.

Both CatBoost and LightGBM (and XGBoost, used inside ensembles) expose
feature importances aligned with the column order used at fit time, so
``get_feature_importance`` just normalizes access to that into a single
tidy DataFrame — including for voting/stacking ensembles (importance-weighted
average across base learners) and multilabel/multi-output regression (plain
average across the per-label/output ``MultiOutputClassifier``/``Regressor``
sub-estimators). ``get_shap_importance`` adds a model-agnostic (via SHAP's
TreeExplainer) mean-|SHAP| ranking as a second, complementary view; it's
skipped for ensembles, multi-output wrappers, and on any failure rather than
breaking a report.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.multioutput import MultiOutputClassifier, MultiOutputRegressor

from smarttab.exceptions import UnsupportedModelError
from smarttab.logging_utils import get_logger

logger = get_logger()

SHAP_SAMPLE_SIZE = 200


def get_feature_importance(estimator, model_name: str, feature_names: list[str]) -> pd.DataFrame:
    if hasattr(estimator, "base_models"):
        parts = [get_feature_importance(est, name, feature_names) for name, est in estimator.base_models]
        combined = pd.concat(parts, ignore_index=True).groupby("feature", as_index=False)["importance"].mean()
        return combined.sort_values("importance", ascending=False, ignore_index=True)

    if isinstance(estimator, (MultiOutputClassifier, MultiOutputRegressor)):
        parts = [get_feature_importance(sub, model_name, feature_names) for sub in estimator.estimators_]
        combined = pd.concat(parts, ignore_index=True).groupby("feature", as_index=False)["importance"].mean()
        return combined.sort_values("importance", ascending=False, ignore_index=True)

    if model_name == "catboost":
        # explicit type= avoids CatBoostRanker's default (LossFunctionChange), which would
        # otherwise require passing the training pool back in just to compute this.
        importances = estimator.get_feature_importance(type="PredictionValuesChange")
    elif model_name in ("lightgbm", "xgboost"):
        importances = estimator.feature_importances_
    else:
        raise UnsupportedModelError(f"Unknown model_name {model_name!r}")

    df = pd.DataFrame({"feature": feature_names, "importance": importances})
    return df.sort_values("importance", ascending=False, ignore_index=True)


def get_shap_importance(estimator, model_name: str, X_sample: pd.DataFrame) -> pd.DataFrame | None:
    """Mean |SHAP value| per feature, computed on up to SHAP_SAMPLE_SIZE rows of X_sample.

    Returns None (rather than raising) if the estimator is an ensemble or SHAP computation
    fails for any reason — this is a supplementary chart, not something a report should fail
    without.
    """
    if (
        hasattr(estimator, "base_models")
        or isinstance(estimator, (MultiOutputClassifier, MultiOutputRegressor))
        or model_name not in ("catboost", "lightgbm", "xgboost")
    ):
        return None

    try:
        import shap

        sample = X_sample.sample(n=min(SHAP_SAMPLE_SIZE, len(X_sample)), random_state=42) if len(X_sample) > SHAP_SAMPLE_SIZE else X_sample
        explainer = shap.TreeExplainer(estimator)
        shap_values = explainer.shap_values(sample)

        if isinstance(shap_values, list):
            shap_values = shap_values[-1]
        shap_values = np.asarray(shap_values)
        if shap_values.ndim == 3:
            shap_values = shap_values[:, :, -1]

        mean_abs = np.abs(shap_values).mean(axis=0)
        df = pd.DataFrame({"feature": sample.columns, "mean_abs_shap": mean_abs})
        return df.sort_values("mean_abs_shap", ascending=False, ignore_index=True)
    except Exception as exc:
        logger.debug("SHAP importance unavailable for %s: %s", model_name, exc)
        return None
