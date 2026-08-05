"""Replayable robust preprocessing primitives learned on training data only."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer
from sklearn.preprocessing import PowerTransformer

from smarttab.exceptions import DataValidationError

RARE_TOKEN = "__rare__"


@dataclass
class MissingIndicatorAdder:
    columns_: list[str] = field(default_factory=list)
    feature_names_: list[str] = field(default_factory=list)

    def fit(self, frame: pd.DataFrame, columns: list[str]) -> "MissingIndicatorAdder":
        self.columns_ = [column for column in columns if column in frame and frame[column].isna().any()]
        self.feature_names_ = [f"{column}__missing" for column in self.columns_]
        return self

    def transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        result = frame.copy()
        for column, feature in zip(self.columns_, self.feature_names_, strict=True):
            if column not in result:
                raise DataValidationError(f"missing-indicator source column {column!r} is unavailable")
            result[feature] = result[column].isna().astype("int8")
        return result


@dataclass
class RareCategoryGrouper:
    min_frequency: float = 0.01
    max_categories: int = 128
    keep_: dict[str, set[str]] = field(default_factory=dict)
    grouped_counts_: dict[str, int] = field(default_factory=dict)

    def fit(self, frame: pd.DataFrame, columns: list[str]) -> "RareCategoryGrouper":
        self.keep_ = {}
        self.grouped_counts_ = {}
        n_rows = max(len(frame), 1)
        min_count = max(2, int(np.ceil(n_rows * self.min_frequency))) if self.min_frequency > 0 else 1
        for column in columns:
            values = frame[column].astype("string").fillna("__missing__")
            counts = values.value_counts()
            keep = counts[counts >= min_count].head(self.max_categories - 1).index.astype(str).tolist()
            self.keep_[column] = set(keep)
            self.grouped_counts_[column] = int((~values.isin(keep)).sum())
        return self

    def transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        result = frame.copy()
        for column, keep in self.keep_.items():
            if column not in result:
                raise DataValidationError(f"rare-category source column {column!r} is unavailable")
            values = result[column].astype("string").fillna("__missing__")
            result[column] = values.where(values.isin(keep), RARE_TOKEN)
        return result


@dataclass
class NumericWinsorizer:
    fraction: float = 0.01
    bounds_: dict[str, tuple[float, float]] = field(default_factory=dict)
    clipped_counts_: dict[str, int] = field(default_factory=dict)

    def fit(self, frame: pd.DataFrame, columns: list[str]) -> "NumericWinsorizer":
        self.bounds_ = {}
        self.clipped_counts_ = {}
        for column in columns:
            values = pd.to_numeric(frame[column], errors="coerce").replace([np.inf, -np.inf], np.nan)
            finite = values.dropna()
            if finite.empty:
                continue
            lower = float(finite.quantile(self.fraction))
            upper = float(finite.quantile(1.0 - self.fraction))
            if lower >= upper:
                continue
            self.bounds_[column] = (lower, upper)
            self.clipped_counts_[column] = int(((finite < lower) | (finite > upper)).sum())
        return self

    def transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        result = frame.copy()
        for column, (lower, upper) in self.bounds_.items():
            values = pd.to_numeric(result[column], errors="coerce").replace([np.inf, -np.inf], np.nan)
            result[column] = values.clip(lower, upper)
        return result


class ConfigurableNumericImputer:
    def __init__(self, strategy: str = "median", fill_value: float = 0.0) -> None:
        self.strategy = strategy
        self.fill_value = float(fill_value)
        self.columns_: list[str] = []
        self.values_: dict[str, float] = {}
        self.medians_: dict[str, float] = {}
        self.model_: object | None = None

    def fit(self, frame: pd.DataFrame, columns: list[str]) -> "ConfigurableNumericImputer":
        self.columns_ = list(columns)
        matrix = frame[self.columns_].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
        if self.strategy in {"median", "mean", "constant"}:
            self.values_ = {}
            for column in self.columns_:
                values = matrix[column]
                if self.strategy == "median":
                    fill = float(values.median()) if values.notna().any() else self.fill_value
                elif self.strategy == "mean":
                    fill = float(values.mean()) if values.notna().any() else self.fill_value
                else:
                    fill = self.fill_value
                self.values_[column] = fill
            self.medians_ = dict(self.values_) if self.strategy == "median" else {}
            return self
        if self.strategy == "knn":
            neighbors = max(2, min(8, int(np.sqrt(max(len(frame), 1)))))
            self.model_ = KNNImputer(n_neighbors=neighbors, weights="distance").fit(matrix)
            return self
        if self.strategy == "iterative":
            from sklearn.experimental import enable_iterative_imputer  # noqa: F401
            from sklearn.impute import IterativeImputer

            self.model_ = IterativeImputer(
                max_iter=10,
                random_state=42,
                initial_strategy="median",
                skip_complete=True,
            ).fit(matrix)
            return self
        raise ValueError(f"unknown numeric imputation strategy {self.strategy!r}")

    def transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        result = frame.copy()
        if not self.columns_:
            return result
        matrix = result[self.columns_].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
        if self.model_ is not None:
            values = np.asarray(self.model_.transform(matrix), dtype=float)
            result.loc[:, self.columns_] = values
        else:
            for column, fill in self.values_.items():
                result[column] = matrix[column].fillna(fill)
        return result


class NumericDistributionTransformer:
    def __init__(self, strategy: str = "auto", skew_threshold: float = 1.5) -> None:
        self.strategy = strategy
        self.skew_threshold = float(skew_threshold)
        self.log_columns_: list[str] = []
        self.power_columns_: list[str] = []
        self.shifts_: dict[str, float] = {}
        self.power_: PowerTransformer | None = None

    def fit(self, frame: pd.DataFrame, columns: list[str]) -> "NumericDistributionTransformer":
        self.log_columns_ = []
        self.power_columns_ = []
        self.shifts_ = {}
        for column in columns:
            values = pd.to_numeric(frame[column], errors="coerce").dropna()
            if len(values) < 20 or values.nunique() < 5:
                continue
            skew = float(values.skew())
            if self.strategy == "none" or abs(skew) < self.skew_threshold:
                continue
            if self.strategy == "log1p" or (self.strategy == "auto" and skew > self.skew_threshold):
                shift = max(0.0, -float(values.min()))
                self.log_columns_.append(column)
                self.shifts_[column] = shift
            else:
                self.power_columns_.append(column)
        if self.power_columns_:
            self.power_ = PowerTransformer(method="yeo-johnson", standardize=False)
            self.power_.fit(frame[self.power_columns_].astype(float))
        return self

    def transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        result = frame.copy()
        for column in self.log_columns_:
            values = pd.to_numeric(result[column], errors="coerce").astype(float)
            shifted = np.maximum(values + self.shifts_[column], 0.0)
            result[column] = np.log1p(shifted)
        if self.power_columns_ and self.power_ is not None:
            transformed = self.power_.transform(result[self.power_columns_].astype(float))
            for index, column in enumerate(self.power_columns_):
                result[column] = transformed[:, index].astype(np.float64, copy=False)
        return result
