"""Categorical and target encoders used by SmartTab."""

from __future__ import annotations

import numpy as np
import pandas as pd

from smarttab.exceptions import DataValidationError


class OrdinalCategoricalEncoder:
    """Stable string-category encoding with an explicit unknown category code."""

    def __init__(self) -> None:
        self.mappings_: dict[str, dict[str, int]] = {}
        self.unknown_codes_: dict[str, int] = {}

    def fit(self, df: pd.DataFrame, columns: list[str]) -> "OrdinalCategoricalEncoder":
        self.mappings_.clear()
        self.unknown_codes_.clear()
        for column in columns:
            values = df[column].astype("string").fillna("__missing__")
            categories = sorted(values.unique().tolist())
            self.mappings_[column] = {category: index for index, category in enumerate(categories)}
            self.unknown_codes_[column] = len(categories)
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        result = df.copy()
        for column, mapping in self.mappings_.items():
            if column not in result.columns:
                raise DataValidationError(f"internal cleaning error: categorical column {column!r} is missing")
            values = result[column].astype("string").fillna("__missing__")
            result[column] = values.map(mapping).fillna(self.unknown_codes_[column]).astype("int32")
        return result


class NativeCategoricalEncoder:
    """Preserve categorical semantics using fitted pandas category vocabularies."""

    def __init__(self) -> None:
        self.categories_: dict[str, list[str]] = {}
        self.unknown_token = "__unknown__"

    def fit(self, df: pd.DataFrame, columns: list[str]) -> "NativeCategoricalEncoder":
        self.categories_ = {}
        for column in columns:
            values = df[column].astype("string").fillna("__missing__")
            categories = sorted(values.unique().tolist())
            if self.unknown_token not in categories:
                categories.append(self.unknown_token)
            self.categories_[column] = categories
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        result = df.copy()
        for column, categories in self.categories_.items():
            if column not in result.columns:
                raise DataValidationError(
                    f"internal cleaning error: categorical column {column!r} is missing"
                )
            values = result[column].astype("string").fillna("__missing__")
            known = set(categories)
            values = values.where(values.isin(known), self.unknown_token)
            result[column] = pd.Categorical(values, categories=categories)
        return result


class TargetLabelEncoder:
    """Encode one classification target to contiguous integer class codes."""

    def __init__(self) -> None:
        self.classes_: np.ndarray | None = None
        self._mapping: dict[object, int] = {}

    def fit(self, y: pd.Series) -> "TargetLabelEncoder":
        classes = pd.unique(y.dropna())
        try:
            classes = np.asarray(sorted(classes.tolist()))
        except TypeError:
            classes = np.asarray(sorted(classes.tolist(), key=lambda value: str(value)))
        if len(classes) < 2:
            raise DataValidationError("classification target must contain at least two classes")
        self.classes_ = classes
        self._mapping = {value: index for index, value in enumerate(classes)}
        return self

    def transform(self, y: pd.Series) -> np.ndarray:
        if self.classes_ is None:
            raise RuntimeError("TargetLabelEncoder is not fitted")
        encoded = y.map(self._mapping)
        if encoded.isna().any():
            unknown = pd.unique(y[encoded.isna()]).tolist()
            raise DataValidationError(f"target contains classes not seen during training: {unknown}")
        return encoded.to_numpy(dtype="int32")

    def inverse_transform(self, codes: np.ndarray) -> np.ndarray:
        if self.classes_ is None:
            raise RuntimeError("TargetLabelEncoder is not fitted")
        indices = np.asarray(codes).astype(int)
        if np.any(indices < 0) or np.any(indices >= len(self.classes_)):
            raise DataValidationError("predicted target code is outside the fitted class range")
        return np.asarray(self.classes_)[indices]


class MultiLabelTargetEncoder:
    """One binary label encoder per multilabel target column."""

    def __init__(self) -> None:
        self.columns_: list[str] = []
        self.encoders_: dict[str, TargetLabelEncoder] = {}

    def fit(self, frame: pd.DataFrame) -> "MultiLabelTargetEncoder":
        self.columns_ = list(frame.columns)
        self.encoders_ = {
            column: TargetLabelEncoder().fit(frame[column]) for column in self.columns_
        }
        for column, encoder in self.encoders_.items():
            if len(encoder.classes_) != 2:
                raise DataValidationError(f"multilabel target {column!r} must contain exactly two classes")
        return self

    def transform(self, frame: pd.DataFrame) -> np.ndarray:
        missing = [column for column in self.columns_ if column not in frame.columns]
        if missing:
            raise DataValidationError(f"missing multilabel target columns: {missing}")
        return np.column_stack(
            [self.encoders_[column].transform(frame[column]) for column in self.columns_]
        ).astype("int8")

    def inverse_transform(self, codes: np.ndarray) -> np.ndarray:
        matrix = np.asarray(codes)
        if matrix.ndim != 2 or matrix.shape[1] != len(self.columns_):
            raise DataValidationError(
                f"multilabel codes must have shape (n_samples, {len(self.columns_)})"
            )
        columns = [
            self.encoders_[column].inverse_transform(matrix[:, index])
            for index, column in enumerate(self.columns_)
        ]
        return np.column_stack(columns)

    @property
    def classes_(self) -> dict[str, list]:
        return {
            column: encoder.classes_.tolist() for column, encoder in self.encoders_.items()
        }
