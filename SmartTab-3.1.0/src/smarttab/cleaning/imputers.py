"""Simple, dependency-free imputers used by the cleaning pipeline."""

from __future__ import annotations

import pandas as pd

MISSING_CATEGORY_TOKEN = "__missing__"


class NumericMedianImputer:
    """Fills missing numeric values with the per-column median learned at fit time."""

    def __init__(self) -> None:
        self.medians_: dict[str, float] = {}

    def fit(self, df: pd.DataFrame, columns: list[str]) -> "NumericMedianImputer":
        self.medians_ = {c: float(df[c].median()) if df[c].notna().any() else 0.0 for c in columns}
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        for c, median in self.medians_.items():
            if c in df.columns:
                df[c] = df[c].fillna(median)
        return df


class CategoricalConstantImputer:
    """Fills missing categorical values with a fixed placeholder token."""

    def __init__(self, token: str = MISSING_CATEGORY_TOKEN) -> None:
        self.token = token
        self.columns_: list[str] = []

    def fit(self, df: pd.DataFrame, columns: list[str]) -> "CategoricalConstantImputer":
        self.columns_ = list(columns)
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        for c in self.columns_:
            if c in df.columns:
                df[c] = df[c].astype("object").where(df[c].notna(), self.token)
        return df
