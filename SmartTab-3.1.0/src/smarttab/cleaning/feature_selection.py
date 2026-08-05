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


def select_top_features(
    df: pd.DataFrame,
    y,
    *,
    task_is_classification: bool,
    categorical_columns: list[str],
    max_features: int,
    random_state: int = 42,
) -> tuple[list[str], dict[str, float]]:
    """Rank a large final feature matrix with train-only mutual information.

    Missing-indicator and source-missing features are retained first because
    their low marginal variance can hide strong robustness value.
    """
    import numpy as np
    from sklearn.feature_selection import mutual_info_classif, mutual_info_regression

    if df.shape[1] <= max_features:
        return list(df.columns), {}
    sample_size = min(len(df), 20_000)
    sampled = df.sample(sample_size, random_state=random_state) if len(df) > sample_size else df
    matrix_columns: list[np.ndarray] = []
    discrete: list[bool] = []
    categorical_set = set(categorical_columns)
    for column in df.columns:
        series = sampled[column]
        is_categorical = column in categorical_set or pd.api.types.is_categorical_dtype(series.dtype)
        if is_categorical:
            if pd.api.types.is_categorical_dtype(series.dtype):
                values = series.cat.codes.to_numpy(dtype=float)
            else:
                values = pd.factorize(series.astype("string"), sort=True)[0].astype(float)
        else:
            values = pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)
            finite = values[np.isfinite(values)]
            fill = float(np.median(finite)) if finite.size else 0.0
            values = np.where(np.isfinite(values), values, fill)
        matrix_columns.append(values)
        discrete.append(is_categorical)
    matrix = np.column_stack(matrix_columns)
    target = y.iloc[sampled.index] if hasattr(y, "iloc") else np.asarray(y)[sampled.index.to_numpy()]
    target_frame = target.to_frame() if isinstance(target, pd.Series) else pd.DataFrame(target)
    scores = np.zeros(df.shape[1], dtype=float)
    for target_column in target_frame.columns:
        values = target_frame[target_column]
        try:
            if task_is_classification:
                target_values = pd.factorize(values.astype("string"), sort=True)[0]
                current = mutual_info_classif(
                    matrix,
                    target_values,
                    discrete_features=np.asarray(discrete, dtype=bool),
                    random_state=random_state,
                )
            else:
                target_values = pd.to_numeric(values, errors="coerce").fillna(0.0).to_numpy(dtype=float)
                current = mutual_info_regression(
                    matrix,
                    target_values,
                    discrete_features=np.asarray(discrete, dtype=bool),
                    random_state=random_state,
                )
            scores = np.maximum(scores, np.nan_to_num(current, nan=0.0))
        except Exception:
            variances = np.nanvar(matrix, axis=0)
            scores = np.maximum(scores, np.log1p(np.maximum(variances, 0.0)))
    mandatory = [
        index for index, column in enumerate(df.columns)
        if column.endswith("__missing") or column.endswith("source_missing")
    ][:max_features]
    mandatory_set = set(mandatory)
    remaining = max_features - len(mandatory)
    ranked = [
        int(index) for index in np.argsort(-scores, kind="stable")
        if int(index) not in mandatory_set
    ][:remaining]
    selected_indices = sorted(mandatory + ranked)
    selected = [str(df.columns[index]) for index in selected_indices]
    score_map = {str(df.columns[index]): float(scores[index]) for index in selected_indices}
    return selected, score_map
