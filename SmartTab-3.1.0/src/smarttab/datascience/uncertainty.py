"""Probability calibration, conformal uncertainty, and robust OOD scoring."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

_EPS = 1e-7


@dataclass
class _BinaryCalibrator:
    method: str
    model: Any

    def transform(self, probability: np.ndarray) -> np.ndarray:
        p = np.clip(np.asarray(probability, dtype=float), _EPS, 1.0 - _EPS)
        if self.method == "isotonic":
            return np.clip(self.model.predict(p), 0.0, 1.0)
        logit = np.log(p / (1.0 - p)).reshape(-1, 1)
        return self.model.predict_proba(logit)[:, 1]


@dataclass
class ProbabilityCalibrator:
    method: str = "none"
    task: str = "binary"
    calibrators_: list[_BinaryCalibrator] = field(default_factory=list)
    fitted_: bool = False
    diagnostics_: dict[str, Any] = field(default_factory=dict)

    def fit(self, y_true: np.ndarray, probabilities: np.ndarray, method: str = "auto") -> "ProbabilityCalibrator":
        y = np.asarray(y_true)
        proba = np.asarray(probabilities, dtype=float)
        if method == "auto":
            method = "isotonic" if len(y) >= 1200 else "sigmoid"
        if method == "none":
            self.method = "none"; self.fitted_ = True; return self
        if method not in {"sigmoid", "isotonic"}:
            raise ValueError("calibration method must be auto, none, sigmoid, or isotonic")
        self.method = method
        self.calibrators_ = []
        if proba.ndim == 1:
            proba = np.column_stack([1.0 - proba, proba])
        if self.task == "binary":
            self.calibrators_.append(_fit_binary(y.astype(int), proba[:, 1], method))
        elif self.task == "multiclass":
            for class_index in range(proba.shape[1]):
                self.calibrators_.append(_fit_binary((y == class_index).astype(int), proba[:, class_index], method))
        elif self.task == "multilabel":
            if y.shape != proba.shape:
                raise ValueError("multilabel y/probability shape mismatch")
            for label_index in range(proba.shape[1]):
                self.calibrators_.append(_fit_binary(y[:, label_index].astype(int), proba[:, label_index], method))
        else:
            raise ValueError(f"unsupported calibration task {self.task!r}")
        self.fitted_ = True
        self.diagnostics_ = {"method": method, "n_samples": int(len(y)), "n_outputs": len(self.calibrators_)}
        return self

    def transform(self, probabilities: np.ndarray) -> np.ndarray:
        proba = np.asarray(probabilities, dtype=float)
        if not self.fitted_ or self.method == "none" or not self.calibrators_:
            return proba
        if self.task == "binary":
            positive = self.calibrators_[0].transform(proba[:, 1])
            return np.column_stack([1.0 - positive, positive])
        transformed = np.column_stack([
            calibrator.transform(proba[:, index]) for index, calibrator in enumerate(self.calibrators_)
        ])
        if self.task == "multiclass":
            sums = transformed.sum(axis=1, keepdims=True)
            transformed = transformed / np.where(sums > 0, sums, 1.0)
        return np.clip(transformed, 0.0, 1.0)


@dataclass
class ConformalPredictor:
    task: str
    alpha: float = 0.10
    quantiles_: Any = None
    fitted_: bool = False

    def fit(self, y_true: np.ndarray, predictions: np.ndarray) -> "ConformalPredictor":
        y = np.asarray(y_true)
        values = np.asarray(predictions, dtype=float)
        if self.task == "regression":
            residuals = np.abs(y.astype(float) - values.astype(float))
            self.quantiles_ = _finite_sample_quantile(residuals, self.alpha, axis=0)
        elif self.task == "multioutput_regression":
            residuals = np.abs(y.astype(float) - values.astype(float))
            self.quantiles_ = _finite_sample_quantile(residuals, self.alpha, axis=0)
        elif self.task == "binary":
            scores = 1.0 - values[np.arange(len(y)), y.astype(int)]
            self.quantiles_ = float(_finite_sample_quantile(scores, self.alpha))
        elif self.task == "multiclass":
            scores = 1.0 - values[np.arange(len(y)), y.astype(int)]
            self.quantiles_ = float(_finite_sample_quantile(scores, self.alpha))
        elif self.task == "multilabel":
            chosen = np.where(y.astype(int) == 1, values, 1.0 - values)
            self.quantiles_ = _finite_sample_quantile(1.0 - chosen, self.alpha, axis=0)
        else:
            raise ValueError(f"unsupported conformal task {self.task!r}")
        self.fitted_ = True
        return self

    def interval(self, predictions: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if not self.fitted_ or self.task not in {"regression", "multioutput_regression"}:
            raise RuntimeError("regression conformal predictor is not fitted")
        values = np.asarray(predictions, dtype=float)
        q = np.asarray(self.quantiles_, dtype=float)
        return values - q, values + q

    def prediction_set(self, probabilities: np.ndarray) -> np.ndarray:
        if not self.fitted_ or self.task not in {"binary", "multiclass"}:
            raise RuntimeError("classification conformal predictor is not fitted")
        threshold = 1.0 - float(self.quantiles_)
        probabilities = np.asarray(probabilities, dtype=float)
        mask = probabilities >= threshold
        empty = ~mask.any(axis=1)
        if empty.any():
            mask[empty, np.argmax(probabilities[empty], axis=1)] = True
        return mask

    def multilabel_sets(self, probabilities: np.ndarray) -> dict[str, np.ndarray]:
        if not self.fitted_ or self.task != "multilabel":
            raise RuntimeError("multilabel conformal predictor is not fitted")
        p = np.asarray(probabilities, dtype=float)
        q = np.asarray(self.quantiles_, dtype=float)
        positive = p >= (1.0 - q)
        negative = (1.0 - p) >= (1.0 - q)
        return {"negative_possible": negative, "positive_possible": positive}


@dataclass
class OODDetector:
    medians_: np.ndarray | None = None
    iqrs_: np.ndarray | None = None
    lower_: np.ndarray | None = None
    upper_: np.ndarray | None = None
    columns_: list[str] = field(default_factory=list)
    score_threshold_: float = 0.65

    def fit(self, frame: pd.DataFrame) -> "OODDetector":
        matrix = _numeric_matrix(frame)
        self.columns_ = list(frame.columns)
        self.medians_ = np.nanmedian(matrix, axis=0)
        q1 = np.nanquantile(matrix, 0.25, axis=0)
        q3 = np.nanquantile(matrix, 0.75, axis=0)
        self.iqrs_ = np.where(q3 - q1 > _EPS, q3 - q1, 1.0)
        self.lower_ = np.nanquantile(matrix, 0.005, axis=0)
        self.upper_ = np.nanquantile(matrix, 0.995, axis=0)
        train_scores = self.score(frame)
        self.score_threshold_ = float(max(0.35, np.quantile(train_scores, 0.995)))
        return self

    def score(self, frame: pd.DataFrame) -> np.ndarray:
        if self.medians_ is None or self.iqrs_ is None:
            raise RuntimeError("OOD detector is not fitted")
        matrix = _numeric_matrix(frame.loc[:, self.columns_])
        filled = np.where(np.isfinite(matrix), matrix, self.medians_)
        robust = np.abs(filled - self.medians_) / self.iqrs_
        robust_score = np.mean(np.clip((robust - 2.0) / 8.0, 0.0, 1.0), axis=1)
        out_of_range = np.mean((filled < self.lower_) | (filled > self.upper_), axis=1)
        missing = np.mean(~np.isfinite(matrix), axis=1)
        return np.clip(0.55 * robust_score + 0.30 * out_of_range + 0.15 * missing, 0.0, 1.0)

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        return self.score(frame) >= self.score_threshold_


def _fit_binary(y: np.ndarray, probability: np.ndarray, method: str) -> _BinaryCalibrator:
    p = np.clip(np.asarray(probability, dtype=float), _EPS, 1.0 - _EPS)
    if len(np.unique(y)) < 2:
        constant = float(np.mean(y))
        model = IsotonicRegression(out_of_bounds="clip").fit([0.0, 1.0], [constant, constant])
        return _BinaryCalibrator("isotonic", model)
    if method == "isotonic":
        model = IsotonicRegression(out_of_bounds="clip").fit(p, y)
        return _BinaryCalibrator(method, model)
    logit = np.log(p / (1.0 - p)).reshape(-1, 1)
    model = LogisticRegression(C=1.0, solver="lbfgs", max_iter=300).fit(logit, y)
    return _BinaryCalibrator(method, model)


def _finite_sample_quantile(values: np.ndarray, alpha: float, axis: int | None = None) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    n = array.shape[0] if array.ndim else 1
    quantile = min(1.0, np.ceil((n + 1) * (1.0 - alpha)) / max(n, 1))
    return np.quantile(array, quantile, axis=axis, method="higher")


def _numeric_matrix(frame: pd.DataFrame) -> np.ndarray:
    columns = []
    for column in frame.columns:
        series = frame[column]
        if isinstance(series.dtype, pd.CategoricalDtype):
            values = series.cat.codes.replace(-1, np.nan).to_numpy(dtype=float)
        else:
            values = pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)
        columns.append(values)
    return np.column_stack(columns) if columns else np.empty((len(frame), 0), dtype=float)
