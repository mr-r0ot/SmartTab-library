"""Outlier handling.

Row removal only ever happens once, on the training split, before the
cleaning pipeline is fit — never inside ``transform()``, since silently
dropping rows a caller is trying to predict on would be surprising and
destructive. ``outlier="auto"`` therefore only *detects* (see
``DatasetProfile.outlier_columns``); this module is only invoked when the
user explicitly opts in with ``outlier="remove"``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def compute_outlier_keep_mask(df: pd.DataFrame, numeric_columns: list[str]) -> np.ndarray:
    """Return a boolean mask (True = keep) for rows that are not IQR outliers in any numeric column."""
    keep = np.ones(len(df), dtype=bool)
    for c in numeric_columns:
        s = df[c]
        if s.notna().sum() < 4:
            continue
        q1, q3 = s.quantile(0.25), s.quantile(0.75)
        iqr = q3 - q1
        if iqr == 0:
            continue
        lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        is_outlier = ((s < lower) | (s > upper)).fillna(False).to_numpy()
        keep &= ~is_outlier
    return keep
