"""Stage 1 — Dataset Analyzer.

Produces a :class:`DatasetProfile`: a single read-only snapshot of everything
downstream stages (cleaning, hardware planning, model selection, optimization,
reporting) need to know about the raw dataset. No stage other than this one
should re-derive these facts from the raw DataFrame.

Task type is detected automatically from the shape/dtype of ``target``:

- single column, low-cardinality discrete -> binary / multiclass
- single column, continuous -> regression
- multiple columns, all binary-like -> multilabel classification
- multiple columns, all continuous -> multi-output regression
- ``group_id`` given -> ranking (target is a single relevance/label column)
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from smarttab.exceptions import DataValidationError

# --- thresholds (kept local to this module; nothing else should hardcode these) ---
MAX_CLASSIFICATION_UNIQUE = 20
HIGH_CORR_THRESHOLD = 0.95
LEAKAGE_CORR_THRESHOLD = 0.98
IMBALANCE_RATIO_THRESHOLD = 0.1
TEXT_AVG_LEN_THRESHOLD = 30
DATETIME_PARSE_SUCCESS_THRESHOLD = 0.9


class TaskType(str, enum.Enum):
    BINARY = "binary"
    MULTICLASS = "multiclass"
    REGRESSION = "regression"
    MULTILABEL = "multilabel"
    MULTIOUTPUT_REGRESSION = "multioutput_regression"
    RANKING = "ranking"

    @property
    def is_classification(self) -> bool:
        return self in (TaskType.BINARY, TaskType.MULTICLASS, TaskType.MULTILABEL)

    @property
    def is_multi_target(self) -> bool:
        """True when the target is several columns fit/predicted together (not ranking)."""
        return self in (TaskType.MULTILABEL, TaskType.MULTIOUTPUT_REGRESSION)

    @property
    def is_ranking(self) -> bool:
        return self is TaskType.RANKING

    @property
    def is_single_target(self) -> bool:
        return self in (TaskType.BINARY, TaskType.MULTICLASS, TaskType.REGRESSION, TaskType.RANKING)


@dataclass
class DatasetProfile:
    task_type: TaskType
    target_column: str  # primary target column (first of target_columns) — kept for single-target call sites
    n_samples: int
    n_features: int
    feature_columns: list[str]

    target_columns: list[str] = field(default_factory=list)
    group_id_column: str | None = None

    dtypes: dict[str, str] = field(default_factory=dict)
    missing_report: dict[str, float] = field(default_factory=dict)
    duplicate_row_count: int = 0
    duplicate_column_groups: list[list[str]] = field(default_factory=list)
    constant_columns: list[str] = field(default_factory=list)
    high_correlation_pairs: list[tuple[str, str, float]] = field(default_factory=list)
    cardinality: dict[str, int] = field(default_factory=dict)

    numeric_columns: list[str] = field(default_factory=list)
    categorical_columns: list[str] = field(default_factory=list)
    datetime_columns: list[str] = field(default_factory=list)
    text_columns: list[str] = field(default_factory=list)
    id_like_columns: list[str] = field(default_factory=list)

    outlier_columns: dict[str, int] = field(default_factory=dict)
    potential_leakage_columns: list[str] = field(default_factory=list)

    class_distribution: dict[Any, int] | None = None
    class_imbalance_ratio: float | None = None
    is_imbalanced: bool = False

    # ranking only
    n_groups: int | None = None
    group_size_stats: dict[str, float] | None = None

    @property
    def n_classes(self) -> int | None:
        return len(self.class_distribution) if self.class_distribution else None


def analyze_dataset(df: pd.DataFrame, target: str | list[str], group_id: str | None = None) -> DatasetProfile:
    """Run full profiling of ``df`` with respect to ``target`` (and optionally ``group_id``
    for ranking) and return a DatasetProfile."""
    if not isinstance(df, pd.DataFrame):
        raise DataValidationError(f"data must be a pandas DataFrame, got {type(df).__name__}")
    if df.empty:
        raise DataValidationError("data is empty")

    task_type, target_columns = _resolve_task_and_targets(df, target, group_id)

    excluded = set(target_columns) | ({group_id} if group_id else set())
    feature_columns = [c for c in df.columns if c not in excluded]
    n_samples = len(df)
    columns = _classify_columns(df, feature_columns)

    profile = DatasetProfile(
        task_type=task_type,
        target_column=target_columns[0],
        target_columns=target_columns,
        group_id_column=group_id,
        n_samples=n_samples,
        n_features=len(feature_columns),
        feature_columns=feature_columns,
        dtypes={c: str(df[c].dtype) for c in df.columns},
        missing_report=_missing_report(df),
        duplicate_row_count=int(df.duplicated().sum()),
        duplicate_column_groups=_duplicate_column_groups(df, feature_columns),
        constant_columns=_constant_columns(df, feature_columns),
        high_correlation_pairs=_high_correlation_pairs(df, columns["numeric"]),
        cardinality={c: int(df[c].nunique(dropna=True)) for c in columns["categorical"]},
        numeric_columns=columns["numeric"],
        categorical_columns=columns["categorical"],
        datetime_columns=columns["datetime"],
        text_columns=columns["text"],
        id_like_columns=columns["id_like"],
        outlier_columns=_outlier_columns(df, columns["numeric"]),
        potential_leakage_columns=_potential_leakage_columns(df, target_columns, columns["numeric"], task_type),
    )

    if task_type in (TaskType.BINARY, TaskType.MULTICLASS):
        y = df[target_columns[0]]
        counts = y.value_counts(dropna=True)
        profile.class_distribution = {str(k): int(v) for k, v in counts.items()}
        ratio = float(counts.min() / counts.max()) if len(counts) > 0 else 1.0
        profile.class_imbalance_ratio = ratio
        profile.is_imbalanced = ratio < IMBALANCE_RATIO_THRESHOLD
    elif task_type is TaskType.MULTILABEL:
        positive_counts = {c: int(df[c].sum()) for c in target_columns}
        profile.class_distribution = positive_counts
        rates = [v / n_samples for v in positive_counts.values() if n_samples > 0]
        if rates:
            ratio = min(rates) / max(rates) if max(rates) > 0 else 1.0
            profile.class_imbalance_ratio = ratio
            profile.is_imbalanced = ratio < IMBALANCE_RATIO_THRESHOLD
    elif task_type is TaskType.RANKING:
        group_sizes = df.groupby(group_id, sort=False).size()
        profile.n_groups = int(len(group_sizes))
        profile.group_size_stats = {
            "min": float(group_sizes.min()), "max": float(group_sizes.max()),
            "mean": float(group_sizes.mean()),
        }

    return profile


def _resolve_task_and_targets(df: pd.DataFrame, target: str | list[str], group_id: str | None) -> tuple[TaskType, list[str]]:
    if isinstance(target, str):
        target_columns = [target]
    elif isinstance(target, (list, tuple)) and all(isinstance(t, str) for t in target):
        target_columns = list(target)
        if not target_columns:
            raise DataValidationError("target must not be an empty list")
    else:
        raise DataValidationError(f"target must be a column name or a list of column names, got {target!r}")

    missing = [c for c in target_columns if c not in df.columns]
    if missing:
        raise DataValidationError(f"target column(s) not found in data: {missing}")
    for c in target_columns:
        if df[c].isna().all():
            raise DataValidationError(f"target column '{c}' is entirely missing")

    if group_id is not None:
        if group_id not in df.columns:
            raise DataValidationError(f"group_id column '{group_id}' not found in data")
        if len(target_columns) != 1:
            raise DataValidationError("ranking (group_id given) requires exactly one target column")
        if df[group_id].isna().any():
            raise DataValidationError(f"group_id column '{group_id}' must not contain missing values")
        return TaskType.RANKING, target_columns

    if len(target_columns) > 1:
        return _infer_multi_target_task_type(df, target_columns), target_columns

    return _infer_task_type(df[target_columns[0]]), target_columns


def _infer_multi_target_task_type(df: pd.DataFrame, target_columns: list[str]) -> TaskType:
    all_binary_like = all(_looks_binary(df[c]) for c in target_columns)
    if all_binary_like:
        return TaskType.MULTILABEL

    all_numeric = all(pd.api.types.is_numeric_dtype(df[c]) for c in target_columns)
    if all_numeric:
        return TaskType.MULTIOUTPUT_REGRESSION

    raise DataValidationError(
        "Multiple target columns must be either all binary-like (0/1 -> multilabel classification) "
        "or all numeric continuous (-> multi-output regression); got a mix of types."
    )


def _looks_binary(y: pd.Series) -> bool:
    non_null = y.dropna()
    if non_null.empty:
        return False
    if pd.api.types.is_bool_dtype(y):
        return True
    return non_null.nunique() <= 2


def _infer_task_type(y: pd.Series) -> TaskType:
    y_non_null = y.dropna()
    if y_non_null.empty:
        raise DataValidationError("target column has no non-missing values")

    if pd.api.types.is_bool_dtype(y) or isinstance(y.dtype, pd.CategoricalDtype) or pd.api.types.is_object_dtype(y):
        n_unique = y_non_null.nunique()
        return TaskType.BINARY if n_unique <= 2 else TaskType.MULTICLASS

    if pd.api.types.is_numeric_dtype(y):
        n_unique = y_non_null.nunique()
        looks_discrete = n_unique <= MAX_CLASSIFICATION_UNIQUE and np.all(
            np.mod(y_non_null.astype(float), 1) == 0
        )
        if looks_discrete:
            return TaskType.BINARY if n_unique <= 2 else TaskType.MULTICLASS
        return TaskType.REGRESSION

    # datetime / other exotic target dtypes: treat as multiclass over their string repr
    return TaskType.MULTICLASS


def _classify_columns(df: pd.DataFrame, feature_columns: list[str]) -> dict[str, list[str]]:
    """Mutually-exclusive column classification: datetime > id_like > text > categorical > numeric."""
    remaining = list(feature_columns)
    n = len(df)

    datetime_cols: list[str] = []
    for c in list(remaining):
        s = df[c]
        if pd.api.types.is_datetime64_any_dtype(s):
            datetime_cols.append(c)
            remaining.remove(c)
        elif pd.api.types.is_object_dtype(s):
            sample = s.dropna()
            if len(sample) == 0:
                continue
            parsed = pd.to_datetime(sample, errors="coerce", format="mixed")
            if parsed.notna().mean() >= DATETIME_PARSE_SUCCESS_THRESHOLD:
                datetime_cols.append(c)
                remaining.remove(c)

    id_like_cols: list[str] = []
    for c in list(remaining):
        s = df[c]
        if n > 1 and s.nunique(dropna=False) == n and (
            pd.api.types.is_integer_dtype(s) or pd.api.types.is_object_dtype(s)
        ):
            id_like_cols.append(c)
            remaining.remove(c)

    text_cols: list[str] = []
    for c in list(remaining):
        s = df[c]
        if pd.api.types.is_object_dtype(s):
            sample = s.dropna().astype(str)
            if len(sample) == 0:
                continue
            avg_len = sample.str.len().mean()
            if avg_len is not None and avg_len > TEXT_AVG_LEN_THRESHOLD:
                text_cols.append(c)
                remaining.remove(c)

    categorical_cols: list[str] = []
    numeric_cols: list[str] = []
    for c in remaining:
        s = df[c]
        if pd.api.types.is_bool_dtype(s) or pd.api.types.is_object_dtype(s) or isinstance(s.dtype, pd.CategoricalDtype):
            categorical_cols.append(c)
        elif pd.api.types.is_numeric_dtype(s):
            numeric_cols.append(c)
        else:
            categorical_cols.append(c)

    return {
        "datetime": datetime_cols,
        "id_like": id_like_cols,
        "text": text_cols,
        "categorical": categorical_cols,
        "numeric": numeric_cols,
    }


def _missing_report(df: pd.DataFrame) -> dict[str, float]:
    frac = df.isna().mean()
    return {str(c): float(v) for c, v in frac.items() if v > 0}


def _duplicate_column_groups(df: pd.DataFrame, feature_columns: list[str]) -> list[list[str]]:
    buckets: dict[Any, list[str]] = {}
    for c in feature_columns:
        try:
            key = pd.util.hash_pandas_object(df[c], index=False).values.tobytes()
        except TypeError:
            continue
        buckets.setdefault(key, []).append(c)
    return [cols for cols in buckets.values() if len(cols) > 1]


def _constant_columns(df: pd.DataFrame, feature_columns: list[str]) -> list[str]:
    return [c for c in feature_columns if df[c].nunique(dropna=False) <= 1]


def _high_correlation_pairs(df: pd.DataFrame, numeric_columns: list[str]) -> list[tuple[str, str, float]]:
    if len(numeric_columns) < 2:
        return []
    corr = df[numeric_columns].corr(numeric_only=True).abs()
    pairs: list[tuple[str, str, float]] = []
    cols = corr.columns.tolist()
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            value = corr.iloc[i, j]
            if pd.notna(value) and value >= HIGH_CORR_THRESHOLD:
                pairs.append((cols[i], cols[j], float(value)))
    return pairs


def _outlier_columns(df: pd.DataFrame, numeric_columns: list[str]) -> dict[str, int]:
    result: dict[str, int] = {}
    for c in numeric_columns:
        s = df[c].dropna()
        if len(s) < 4:
            continue
        q1, q3 = s.quantile(0.25), s.quantile(0.75)
        iqr = q3 - q1
        if iqr == 0:
            continue
        lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        count = int(((s < lower) | (s > upper)).sum())
        if count > 0:
            result[c] = count
    return result


def _potential_leakage_columns(
    df: pd.DataFrame, target_columns: list[str], numeric_columns: list[str], task_type: TaskType
) -> list[str]:
    flagged: set[str] = set()

    for target in target_columns:
        target_name_lower = target.lower()

        if task_type in (TaskType.REGRESSION, TaskType.MULTIOUTPUT_REGRESSION, TaskType.RANKING):
            y = pd.to_numeric(df[target], errors="coerce")
        else:
            y = df[target].astype("category").cat.codes.astype(float)

        for c in numeric_columns:
            try:
                corr = df[c].corr(y)
            except Exception:
                continue
            if pd.notna(corr) and abs(corr) >= LEAKAGE_CORR_THRESHOLD:
                flagged.add(c)

        for c in numeric_columns:
            if target_name_lower in c.lower() or c.lower() in target_name_lower:
                flagged.add(c)

    return sorted(flagged)
