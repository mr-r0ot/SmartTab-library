"""Conservative post-cleaning feature selection.

Tree ensembles (CatBoost/LightGBM) handle correlated/redundant features
gracefully, so ``feature_selection="auto"`` deliberately only removes
*near-exact duplicate* numeric columns (correlation >= 0.999) rather than
aggressively pruning merely-correlated features, which could remove real
signal without improving tree-based models.
"""

from __future__ import annotations

import pandas as pd

NEAR_DUPLICATE_THRESHOLD = 0.999


def select_columns_to_drop(df: pd.DataFrame, numeric_columns: list[str]) -> list[str]:
    candidates = [c for c in numeric_columns if c in df.columns]
    if len(candidates) < 2:
        return []
    corr = df[candidates].corr(numeric_only=True).abs()
    to_drop: set[str] = set()
    for i, col_a in enumerate(candidates):
        if col_a in to_drop:
            continue
        for col_b in candidates[i + 1 :]:
            if col_b in to_drop:
                continue
            value = corr.loc[col_a, col_b]
            if pd.notna(value) and value >= NEAR_DUPLICATE_THRESHOLD:
                to_drop.add(col_b)
    return sorted(to_drop)
