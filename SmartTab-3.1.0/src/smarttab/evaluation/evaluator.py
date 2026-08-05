"""Evaluation metrics shared by optimization, holdout evaluation, and reports."""

from __future__ import annotations

import numpy as np
from sklearn import metrics as skm

from smarttab.logging_utils import get_logger

logger = get_logger()


def _expected_calibration_error(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    *,
    n_bins: int = 10,
) -> float:
    """Return top-label ECE for binary or multiclass probabilities."""
    y_true = np.asarray(y_true)
    proba = np.asarray(y_proba, dtype=float)
    if proba.ndim != 2 or proba.shape[0] != len(y_true):
        raise ValueError("y_proba must have shape (n_samples, n_classes)")
    confidence = proba.max(axis=1)
    prediction = proba.argmax(axis=1)
    correctness = (prediction == y_true).astype(float)
    boundaries = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for index in range(n_bins):
        lower, upper = boundaries[index], boundaries[index + 1]
        if index == n_bins - 1:
            mask = (confidence >= lower) & (confidence <= upper)
        else:
            mask = (confidence >= lower) & (confidence < upper)
        if mask.any():
            ece += float(mask.mean()) * abs(float(correctness[mask].mean()) - float(confidence[mask].mean()))
    return float(ece)


def evaluate_classification(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray | None = None,
) -> dict[str, float]:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
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
        proba = np.asarray(y_proba, dtype=float)
        try:
            results["log_loss"] = skm.log_loss(y_true, proba, labels=np.arange(proba.shape[1]))
        except ValueError as exc:
            logger.debug("log_loss unavailable: %s", exc)
        try:
            results["expected_calibration_error"] = _expected_calibration_error(y_true, proba)
        except ValueError as exc:
            logger.debug("calibration metric unavailable: %s", exc)

        if proba.shape[1] == 2:
            positive = proba[:, 1]
            try:
                results["roc_auc"] = skm.roc_auc_score(y_true, positive)
            except ValueError as exc:
                logger.debug("roc_auc unavailable: %s", exc)
            try:
                results["pr_auc"] = skm.average_precision_score(y_true, positive)
            except ValueError as exc:
                logger.debug("pr_auc unavailable: %s", exc)
            try:
                results["brier_score"] = skm.brier_score_loss(y_true, positive)
            except ValueError as exc:
                logger.debug("brier_score unavailable: %s", exc)
        else:
            try:
                results["roc_auc"] = skm.roc_auc_score(
                    y_true,
                    proba,
                    multi_class="ovr",
                    average="macro",
                    labels=np.arange(proba.shape[1]),
                )
            except ValueError as exc:
                logger.debug("multiclass roc_auc unavailable: %s", exc)

    return {key: float(value) for key, value in results.items()}


def evaluate_multilabel(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray | None = None,
) -> dict[str, float]:
    """Evaluate ``(n_samples, n_labels)`` indicator matrices."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    results: dict[str, float] = {
        "subset_accuracy": skm.accuracy_score(y_true, y_pred),
        "hamming_loss": skm.hamming_loss(y_true, y_pred),
        "f1_macro": skm.f1_score(y_true, y_pred, average="macro", zero_division=0),
        "f1_micro": skm.f1_score(y_true, y_pred, average="micro", zero_division=0),
        "f1_samples": skm.f1_score(y_true, y_pred, average="samples", zero_division=0),
        "precision_macro": skm.precision_score(y_true, y_pred, average="macro", zero_division=0),
        "recall_macro": skm.recall_score(y_true, y_pred, average="macro", zero_division=0),
    }
    if y_proba is not None:
        proba = np.asarray(y_proba, dtype=float)
        if proba.shape == y_true.shape:
            try:
                results["average_precision_macro"] = skm.average_precision_score(
                    y_true, proba, average="macro"
                )
            except ValueError as exc:
                logger.debug("multilabel average precision unavailable: %s", exc)
    return {key: float(value) for key, value in results.items()}


def evaluate_multioutput_regression(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float | list[float]]:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mse = skm.mean_squared_error(y_true, y_pred)
    per_output_mse = skm.mean_squared_error(y_true, y_pred, multioutput="raw_values")
    return {
        "mae": float(skm.mean_absolute_error(y_true, y_pred)),
        "mse": float(mse),
        "rmse": float(np.sqrt(mse)),
        "r2": float(skm.r2_score(y_true, y_pred)),
        "rmse_per_output": [float(np.sqrt(value)) for value in per_output_mse],
    }


def compute_ndcg(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    group_ids: np.ndarray,
    k: int = 10,
) -> float:
    scores: list[float] = []
    for group in np.unique(group_ids):
        mask = group_ids == group
        if mask.sum() < 2:
            continue
        scores.append(float(skm.ndcg_score([y_true[mask]], [y_pred[mask]], k=k)))
    return float(np.mean(scores)) if scores else 0.0


def evaluate_ranking(y_true: np.ndarray, y_pred: np.ndarray, group_ids: np.ndarray) -> dict[str, float]:
    return {
        "ndcg@5": compute_ndcg(y_true, y_pred, group_ids, k=5),
        "ndcg@10": compute_ndcg(y_true, y_pred, group_ids, k=10),
        "n_groups": float(len(np.unique(group_ids))),
    }


def evaluate_regression(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
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
    except ValueError as exc:
        logger.debug("MAPE unavailable: %s", exc)
    return {key: float(value) for key, value in results.items()}


def compute_metric(
    name: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray | None = None,
) -> float:
    """Score a single metric for optimization and ensemble selection."""
    if name == "rmse":
        return float(np.sqrt(skm.mean_squared_error(y_true, y_pred)))
    if name == "mae":
        return float(skm.mean_absolute_error(y_true, y_pred))
    if name == "r2":
        return float(skm.r2_score(y_true, y_pred))
    if name == "roc_auc":
        if y_proba is None:
            raise ValueError("roc_auc requires probabilities")
        proba = np.asarray(y_proba)
        if proba.shape[1] == 2:
            return float(skm.roc_auc_score(y_true, proba[:, 1]))
        return float(skm.roc_auc_score(y_true, proba, multi_class="ovr", average="macro"))
    if name == "f1_macro":
        return float(skm.f1_score(y_true, y_pred, average="macro", zero_division=0))
    if name == "f1":
        return float(skm.f1_score(y_true, y_pred, average="binary", zero_division=0))
    if name == "accuracy":
        return float(skm.accuracy_score(y_true, y_pred))
    raise ValueError(f"Unknown metric name: {name!r}")
