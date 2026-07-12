"""Stage 7 — Evaluation.

Computes the full classification/regression metric sets from the spec, and
exposes :func:`compute_metric` as the single source of truth for scoring a
single named metric — used both here and by the Optuna objective in
``optimization/optimizer.py`` so search and reporting never disagree about
what a metric means.
"""

from __future__ import annotations

import numpy as np
from sklearn import metrics as skm

from smarttab.logging_utils import get_logger

logger = get_logger()


def evaluate_classification(
    y_true: np.ndarray, y_pred: np.ndarray, y_proba: np.ndarray | None = None
) -> dict[str, float]:
    n_classes = len(np.unique(y_true))
    average = "binary" if n_classes <= 2 else "macro"

    results: dict[str, float] = {
        "accuracy": skm.accuracy_score(y_true, y_pred),
        "precision": skm.precision_score(y_true, y_pred, average=average, zero_division=0),
        "recall": skm.recall_score(y_true, y_pred, average=average, zero_division=0),
        "f1": skm.f1_score(y_true, y_pred, average=average, zero_division=0),
        "f1_macro": skm.f1_score(y_true, y_pred, average="macro", zero_division=0),
        "f1_weighted": skm.f1_score(y_true, y_pred, average="weighted", zero_division=0),
        "mcc": skm.matthews_corrcoef(y_true, y_pred),
        "cohen_kappa": skm.cohen_kappa_score(y_true, y_pred),
        "balanced_accuracy": skm.balanced_accuracy_score(y_true, y_pred),
    }

    if y_proba is not None:
        try:
            if n_classes <= 2:
                results["roc_auc"] = skm.roc_auc_score(y_true, y_proba[:, 1])
            else:
                results["roc_auc"] = skm.roc_auc_score(y_true, y_proba, multi_class="ovr", average="macro")
        except ValueError as exc:
            logger.debug("roc_auc_score unavailable: %s", exc)
        try:
            results["log_loss"] = skm.log_loss(y_true, y_proba)
        except ValueError as exc:
            logger.debug("log_loss unavailable: %s", exc)

    return {k: float(v) for k, v in results.items()}


def evaluate_multilabel(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """``y_true``/``y_pred`` are (n_samples, n_labels) 0/1 indicator matrices."""
    results = {
        "subset_accuracy": skm.accuracy_score(y_true, y_pred),  # exact match across all labels
        "hamming_loss": skm.hamming_loss(y_true, y_pred),
        "f1_macro": skm.f1_score(y_true, y_pred, average="macro", zero_division=0),
        "f1_micro": skm.f1_score(y_true, y_pred, average="micro", zero_division=0),
        "f1_samples": skm.f1_score(y_true, y_pred, average="samples", zero_division=0),
        "precision_macro": skm.precision_score(y_true, y_pred, average="macro", zero_division=0),
        "recall_macro": skm.recall_score(y_true, y_pred, average="macro", zero_division=0),
    }
    return {k: float(v) for k, v in results.items()}


def evaluate_multioutput_regression(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """``y_true``/``y_pred`` are (n_samples, n_outputs); metrics are averaged across outputs
    (sklearn's default ``multioutput='uniform_average'``), plus a per-output RMSE breakdown."""
    mse = skm.mean_squared_error(y_true, y_pred)
    results = {
        "mae": skm.mean_absolute_error(y_true, y_pred),
        "mse": mse,
        "rmse": float(np.sqrt(mse)),
        "r2": skm.r2_score(y_true, y_pred),
    }
    per_output_mse = skm.mean_squared_error(y_true, y_pred, multioutput="raw_values")
    results["rmse_per_output"] = [float(np.sqrt(v)) for v in per_output_mse]
    return {k: (float(v) if not isinstance(v, list) else v) for k, v in results.items()}


def compute_ndcg(y_true: np.ndarray, y_pred: np.ndarray, group_ids: np.ndarray, k: int = 10) -> float:
    """Mean NDCG@k across query groups. Groups with fewer than 2 items are skipped (NDCG is
    undefined/trivial for a single-item ranking)."""
    from sklearn.metrics import ndcg_score

    scores = []
    for group in np.unique(group_ids):
        mask = group_ids == group
        if mask.sum() < 2:
            continue
        scores.append(ndcg_score([y_true[mask]], [y_pred[mask]], k=k))
    return float(np.mean(scores)) if scores else 0.0


def evaluate_ranking(y_true: np.ndarray, y_pred: np.ndarray, group_ids: np.ndarray) -> dict[str, float]:
    return {
        "ndcg@5": compute_ndcg(y_true, y_pred, group_ids, k=5),
        "ndcg@10": compute_ndcg(y_true, y_pred, group_ids, k=10),
        "n_groups": float(len(np.unique(group_ids))),
    }


def evaluate_regression(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    mse = skm.mean_squared_error(y_true, y_pred)
    results = {
        "mae": skm.mean_absolute_error(y_true, y_pred),
        "mse": mse,
        "rmse": float(np.sqrt(mse)),
        "r2": skm.r2_score(y_true, y_pred),
        "median_ae": skm.median_absolute_error(y_true, y_pred),
    }
    try:
        results["mape"] = skm.mean_absolute_percentage_error(y_true, y_pred)
    except Exception as exc:
        logger.debug("MAPE unavailable (likely zero values in y_true): %s", exc)

    return {k: float(v) for k, v in results.items()}


def compute_metric(name: str, y_true: np.ndarray, y_pred: np.ndarray, y_proba: np.ndarray | None = None) -> float:
    """Score a single named metric — used by the optimizer's objective function."""
    if name == "rmse":
        return float(np.sqrt(skm.mean_squared_error(y_true, y_pred)))
    if name == "mae":
        return float(skm.mean_absolute_error(y_true, y_pred))
    if name == "r2":
        return float(skm.r2_score(y_true, y_pred))
    if name == "roc_auc":
        if y_proba is None:
            raise ValueError("roc_auc requires y_proba")
        n_classes = y_proba.shape[1]
        if n_classes <= 2:
            return float(skm.roc_auc_score(y_true, y_proba[:, 1]))
        return float(skm.roc_auc_score(y_true, y_proba, multi_class="ovr", average="macro"))
    if name == "f1_macro":
        return float(skm.f1_score(y_true, y_pred, average="macro", zero_division=0))
    if name == "f1":
        return float(skm.f1_score(y_true, y_pred, average="binary", zero_division=0))

    raise ValueError(f"Unknown metric name: {name!r}")
