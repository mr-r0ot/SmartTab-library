"""Base class for fitted multimodal feature extractors."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import pandas as pd


class BaseFeatureExtractor(ABC):
    modality: str

    def __init__(self, *, max_features: int, error_policy: str = "warn", **_: Any) -> None:
        self.max_features = int(max_features)
        self.error_policy = error_policy
        self.feature_names_: list[str] = []
        self.errors_: list[str] = []
        self._fitted = False

    @abstractmethod
    def fit_transform(self, values: pd.Series, y: Any = None) -> pd.DataFrame:
        raise NotImplementedError

    @abstractmethod
    def transform(self, values: pd.Series) -> pd.DataFrame:
        raise NotImplementedError

    def _record_error(self, message: str) -> None:
        if self.error_policy == "error":
            raise ValueError(message)
        if self.error_policy == "warn" and len(self.errors_) < 20:
            self.errors_.append(message)
