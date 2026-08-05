"""Decision-threshold optimization for binary and multilabel classification."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

from smarttab.analysis.dataset_analyzer import TaskType
from smarttab.deadline import FitDeadline
from smarttab.hardware.resource_planner import ResourcePlan

DEFAULT_THRESHOLD = 0.5
DEFAULT_REJECT_THRESHOLD = 0.0
DEFAULT_OBJECTIVE = "mcc"
_THRESHOLD_GRID = np.linspace(0.01, 0.99, 99)
_PROBE_HOLDOUT_FRACTION = 0.2

SWEEPABLE_OBJECTIVES = (
    "f1",
    "precision",
    "recall",
    "accuracy",
    "balanced_accuracy",
    "mcc",
)
VALID_OBJECTIVES = SWEEPABLE_OBJECTIVES + ("roc_auc",)


def apply_threshold(y_proba_positive: np.ndarray, threshold: float) -> np.ndarray:
    return (np.asarray(y_proba_positive) >= float(threshold)).astype("int8")


def optimize_threshold(
    y_true: np.ndarray,
    y_proba_positive: np.ndarray,
    objective: str = DEFAULT_OBJECTIVE,
) -> tuple[float, float]:
    y_true = np.asarray(y_true)
    probabilities = np.asarray(y_proba_positive, dtype=float)
    if objective not in VALID_OBJECTIVES:
        raise ValueError(f"objective must be one of {VALID_OBJECTIVES}, got {objective!r}")
    if len(np.unique(y_true)) < 2:
        return DEFAULT_THRESHOLD, 0.0
    if objective == "roc_auc":
        return DEFAULT_THRESHOLD, float(roc_auc_score(y_true, probabilities))

    best_threshold = DEFAULT_THRESHOLD
    best_score = _score_at_threshold(objective, y_true, probabilities, best_threshold)
    for threshold in _THRESHOLD_GRID:
        score = _score_at_threshold(objective, y_true, probabilities, float(threshold))
        if score > best_score:
            best_score = float(score)
            best_threshold = float(threshold)
    return best_threshold, best_score


def optimize_per_label_thresholds(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    objective: str = DEFAULT_OBJECTIVE,
) -> list[float]:
    y_true = np.asarray(y_true)
    y_proba = np.asarray(y_proba)
    if y_true.shape != y_proba.shape:
        raise ValueError("multilabel y_true and y_proba must have the same shape")
    return [
        optimize_threshold(y_true[:, index], y_proba[:, index], objective)[0]
        for index in range(y_true.shape[1])
    ]


def optimize_multiclass_reject_threshold(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    objective: str = DEFAULT_OBJECTIVE,
) -> tuple[float, float]:
    """Compatibility no-op.

    Automatic multiclass rejection was removed because it changed the production
    contract without producing calibrated uncertainty. Plain argmax is retained.
    """
    predictions = np.asarray(y_proba).argmax(axis=1)
    score = f1_score(y_true, predictions, average="macro", zero_division=0)
    return DEFAULT_REJECT_THRESHOLD, float(score)


def probe_and_optimize_threshold(
    model_name: str,
    params: dict,
    n_estimators: int,
    task_type: TaskType,
    resource_plan: ResourcePlan,
    cat_features: list[str],
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    random_state: int = 42,
    objective: str = DEFAULT_OBJECTIVE,
    deadline: FitDeadline | None = None,
) -> float:
    if task_type is not TaskType.BINARY:
        return DEFAULT_THRESHOLD
    from smarttab.training.trainer import build_estimator, fit_estimator, predict_proba

    try:
        X_fit, X_valid, y_fit, y_valid = train_test_split(
            X_train,
            y_train,
            test_size=_PROBE_HOLDOUT_FRACTION,
            random_state=random_state,
            stratify=y_train,
        )
    except ValueError:
        return DEFAULT_THRESHOLD
    estimator = build_estimator(
        model_name,
        params,
        task_type,
        n_estimators,
        resource_plan.cpu_threads,
        resource_plan.use_gpu,
        random_state,
        resource_plan=resource_plan,
    )
    fit_estimator(
        estimator,
        model_name,
        X_fit,
        y_fit,
        X_valid,
        y_valid,
        cat_features=cat_features,
        early_stopping_rounds=30,
        deadline=deadline,
        resource_plan=resource_plan,
    )
    probabilities = predict_proba(estimator, model_name, X_valid)[:, 1]
    return optimize_threshold(y_valid, probabilities, objective)[0]


def probe_and_optimize_multilabel_thresholds(
    model_name: str,
    params: dict,
    n_estimators: int,
    task_type: TaskType,
    resource_plan: ResourcePlan,
    cat_features: list[str],
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    random_state: int = 42,
    objective: str = DEFAULT_OBJECTIVE,
    deadline: FitDeadline | None = None,
) -> list[float]:
    from smarttab.training.trainer import build_estimator, fit_estimator, predict_proba

    n_labels = y_train.shape[1]
    try:
        X_fit, X_valid, y_fit, y_valid = train_test_split(
            X_train,
            y_train,
            test_size=_PROBE_HOLDOUT_FRACTION,
            random_state=random_state,
        )
    except ValueError:
        return [DEFAULT_THRESHOLD] * n_labels
    estimator = build_estimator(
        model_name,
        params,
        task_type,
        n_estimators,
        resource_plan.cpu_threads,
        resource_plan.use_gpu,
        random_state,
        resource_plan=resource_plan,
    )
    fit_estimator(
        estimator,
        model_name,
        X_fit,
        y_fit,
        X_valid,
        y_valid,
        cat_features=cat_features,
        early_stopping_rounds=30,
        deadline=deadline,
        resource_plan=resource_plan,
    )
    probabilities = predict_proba(estimator, model_name, X_valid)
    return optimize_per_label_thresholds(y_valid, probabilities, objective)


def _score_at_threshold(
    objective: str,
    y_true: np.ndarray,
    probabilities: np.ndarray,
    threshold: float,
) -> float:
    predictions = apply_threshold(probabilities, threshold)
    if objective == "f1":
        return float(f1_score(y_true, predictions, zero_division=0))
    if objective == "precision":
        return float(precision_score(y_true, predictions, zero_division=0))
    if objective == "recall":
        return float(recall_score(y_true, predictions, zero_division=0))
    if objective == "accuracy":
        return float(accuracy_score(y_true, predictions))
    if objective == "balanced_accuracy":
        return float(balanced_accuracy_score(y_true, predictions))
    if objective == "mcc":
        return float(matthews_corrcoef(y_true, predictions))
    raise ValueError(f"unsupported threshold objective {objective!r}")
