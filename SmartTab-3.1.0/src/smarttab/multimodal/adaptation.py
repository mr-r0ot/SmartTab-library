"""Bounded supervised adaptation of extracted modality features.

The adapter intentionally learns a small task-aware projection on top of frozen
classical/pretrained features. It improves domain alignment without requiring an
end-to-end neural training stack or allowing the feature space to grow without a
hard cap.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from sklearn.cross_decomposition import PLSRegression
from sklearn.preprocessing import StandardScaler

from smarttab.analysis.dataset_analyzer import TaskType


@dataclass
class SupervisedFeatureAdapter:
    """Learn a compact PLS projection from modality features to the task target."""

    task_type: TaskType
    n_components: int = 16
    max_fit_rows: int = 20_000
    random_state: int = 42
    scaler_: StandardScaler | None = None
    model_: PLSRegression | None = None
    medians_: np.ndarray | None = None
    feature_names_: list[str] = field(default_factory=list)
    diagnostics_: dict[str, Any] = field(default_factory=dict)

    def fit(self, frame: pd.DataFrame, y: pd.Series | pd.DataFrame | np.ndarray) -> "SupervisedFeatureAdapter":
        matrix = frame.to_numpy(dtype=np.float64, copy=True)
        if matrix.ndim != 2 or matrix.shape[0] < 20 or matrix.shape[1] < 2:
            return self
        finite = np.where(np.isfinite(matrix), matrix, np.nan)
        self.medians_ = np.zeros(matrix.shape[1], dtype=np.float64)
        valid_columns = np.isfinite(finite).any(axis=0)
        if valid_columns.any():
            self.medians_[valid_columns] = np.nanmedian(finite[:, valid_columns], axis=0)
        self.medians_ = np.nan_to_num(self.medians_, nan=0.0, posinf=0.0, neginf=0.0)
        matrix = np.where(np.isfinite(matrix), matrix, self.medians_)
        target = _target_matrix(y, self.task_type)
        if target.shape[0] != matrix.shape[0] or target.shape[1] == 0:
            return self
        valid_target = np.all(np.isfinite(target), axis=1)
        matrix = matrix[valid_target]
        target = target[valid_target]
        if len(matrix) < 20 or np.all(np.nanstd(target, axis=0) < 1e-12):
            return self
        if len(matrix) > self.max_fit_rows:
            rng = np.random.default_rng(self.random_state)
            indices = np.sort(rng.choice(len(matrix), self.max_fit_rows, replace=False))
            matrix = matrix[indices]
            target = target[indices]
        self.scaler_ = StandardScaler().fit(matrix)
        scaled = self.scaler_.transform(matrix)
        nonconstant = np.nanstd(scaled, axis=0) > 1e-10
        if int(nonconstant.sum()) < 2:
            self.scaler_ = None
            return self
        # Keep the original dimensionality contract. Constant inputs remain in the
        # scaler but PLS receives only informative columns through a stored mask.
        self._nonconstant_mask = nonconstant
        requested = min(
            int(self.n_components),
            int(nonconstant.sum()),
            max(1, len(scaled) - 1),
        )
        if requested < 1:
            self.scaler_ = None
            return self
        try:
            self.model_ = PLSRegression(
                n_components=requested,
                scale=False,
                max_iter=500,
                tol=1e-6,
            ).fit(scaled[:, nonconstant], target)
        except Exception:
            self.model_ = None
            self.scaler_ = None
            return self
        self.feature_names_ = [f"supervised_adapter_{index:03d}" for index in range(requested)]
        self.diagnostics_ = {
            "method": "pls",
            "n_components": requested,
            "n_fit_rows": int(len(scaled)),
            "n_input_features": int(frame.shape[1]),
            "target_dimensions": int(target.shape[1]),
        }
        return self

    @property
    def fitted(self) -> bool:
        return self.model_ is not None and self.scaler_ is not None and bool(self.feature_names_)

    def transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        if not self.fitted or self.medians_ is None:
            return pd.DataFrame(index=frame.index)
        matrix = frame.to_numpy(dtype=np.float64, copy=True)
        matrix = np.where(np.isfinite(matrix), matrix, self.medians_)
        scaled = self.scaler_.transform(matrix)
        scores = self.model_.transform(scaled[:, self._nonconstant_mask])
        if isinstance(scores, tuple):
            scores = scores[0]
        scores = np.asarray(scores, dtype=np.float32)
        if scores.ndim == 1:
            scores = scores.reshape(-1, 1)
        scores = scores[:, : len(self.feature_names_)]
        return pd.DataFrame(scores, columns=self.feature_names_[: scores.shape[1]], index=frame.index)


def _target_matrix(y: pd.Series | pd.DataFrame | np.ndarray, task_type: TaskType) -> np.ndarray:
    target = y.to_frame() if isinstance(y, pd.Series) else pd.DataFrame(y)
    if task_type in {TaskType.BINARY, TaskType.MULTICLASS}:
        encoded = pd.get_dummies(target.iloc[:, 0].astype("string"), dtype=float)
        return encoded.to_numpy(dtype=np.float64)
    if task_type is TaskType.MULTILABEL:
        columns: list[np.ndarray] = []
        for name in target.columns:
            series = target[name]
            numeric = pd.to_numeric(series, errors="coerce")
            if numeric.notna().all():
                columns.append(numeric.to_numpy(dtype=np.float64))
            else:
                columns.append(pd.factorize(series.astype("string"), sort=True)[0].astype(np.float64))
        return np.column_stack(columns)
    numeric = target.apply(pd.to_numeric, errors="coerce")
    return numeric.to_numpy(dtype=np.float64)
