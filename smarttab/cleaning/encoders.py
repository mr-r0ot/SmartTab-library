"""Categorical encoding used by the cleaning pipeline.

Categorical columns are ordinal-encoded to small integers rather than
one-hot encoded: both CatBoost and LightGBM accept integer-coded categorical
columns natively (via ``cat_features`` / ``categorical_feature``), which
avoids exploding dimensionality on high-cardinality columns and lets each
library handle categories with its own (better) internal statistics.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


class OrdinalCategoricalEncoder:
    """Per-column category -> integer code map, with a reserved code for unseen categories."""

    def __init__(self) -> None:
        self.mappings_: dict[str, dict[str, int]] = {}
        self.unknown_codes_: dict[str, int] = {}

    def fit(self, df: pd.DataFrame, columns: list[str]) -> "OrdinalCategoricalEncoder":
        for c in columns:
            categories = sorted(df[c].astype(str).unique().tolist())
            mapping = {cat: i for i, cat in enumerate(categories)}
            self.mappings_[c] = mapping
            self.unknown_codes_[c] = len(mapping)  # reserved code for unseen-at-transform-time values
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        for c, mapping in self.mappings_.items():
            if c not in df.columns:
                continue
            unknown_code = self.unknown_codes_[c]
            df[c] = df[c].astype(str).map(mapping).fillna(unknown_code).astype("int32")
        return df


class TargetLabelEncoder:
    """Encodes a classification target to contiguous integer codes and back.

    Only used for classification targets so training/evaluation can work with
    plain integer class indices; ``inverse_transform`` restores the caller's
    original label values (e.g. strings) for ``predict()`` output.
    """

    def __init__(self) -> None:
        self.classes_: np.ndarray | None = None

    def fit(self, y: pd.Series) -> "TargetLabelEncoder":
        self.classes_ = np.sort(y.dropna().unique())
        return self

    def transform(self, y: pd.Series) -> np.ndarray:
        mapping = {v: i for i, v in enumerate(self.classes_)}
        return y.map(mapping).to_numpy()

    def inverse_transform(self, codes: np.ndarray) -> np.ndarray:
        return np.asarray(self.classes_)[np.asarray(codes).astype(int)]
