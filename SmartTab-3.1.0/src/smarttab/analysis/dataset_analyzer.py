"""Train-only dataset analysis and task inference."""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from smarttab.exceptions import DataValidationError
from smarttab.multimodal.detector import resolve_column_modalities

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
    target_column: str
    n_samples: int
    n_features: int
    feature_columns: list[str]

    target_columns: list[str] = field(default_factory=list)
    group_id_column: str | None = None
    source_n_samples: int | None = None
    holdout_n_samples: int | None = None
    duplicate_rows_removed: int = 0
    outlier_rows_removed: int = 0
    low_quality_rows_removed: int = 0
    conflicting_label_rows_removed: int = 0
    data_quality_report: dict[str, Any] = field(default_factory=dict)
    cleaning_report: dict[str, Any] = field(default_factory=dict)

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
    image_columns: list[str] = field(default_factory=list)
    audio_columns: list[str] = field(default_factory=list)
    video_columns: list[str] = field(default_factory=list)
    column_modalities: dict[str, str] = field(default_factory=dict)
    id_like_columns: list[str] = field(default_factory=list)

    outlier_columns: dict[str, int] = field(default_factory=dict)
    potential_leakage_columns: list[str] = field(default_factory=list)
    leakage_reasons: dict[str, list[str]] = field(default_factory=dict)

    class_distribution: dict[Any, int] | None = None
    class_imbalance_ratio: float | None = None
    is_imbalanced: bool = False

    n_groups: int | None = None
    group_size_stats: dict[str, float] | None = None

    @property
    def n_classes(self) -> int | None:
        return len(self.class_distribution) if self.class_distribution else None


def resolve_task_and_targets(
    df: pd.DataFrame,
    target: str | list[str],
    group_id: str | None = None,
    task_type: str | TaskType = "auto",
) -> tuple[TaskType, list[str]]:
    """Validate target columns and resolve the task.

    Numeric integer targets with more than two values are conservatively treated as
    regression in auto mode. Ambiguous multiclass integer labels should use
    ``task_type='multiclass'`` explicitly.
    """
    _validate_frame(df)
    target_columns = _normalize_target_columns(target)
    missing = [c for c in target_columns if c not in df.columns]
    if missing:
        raise DataValidationError(f"target column(s) not found in data: {missing}")
    if group_id is not None and group_id not in df.columns:
        raise DataValidationError(f"group_id column {group_id!r} not found in data")
    if group_id is not None and df[group_id].isna().any():
        raise DataValidationError(f"group_id column {group_id!r} must not contain missing values")

    for column in target_columns:
        if df[column].isna().all():
            raise DataValidationError(f"target column {column!r} is entirely missing")
        if df[column].isna().any():
            raise DataValidationError(
                f"target column {column!r} contains missing values; use target_missing='drop' or clean the data"
            )

    if isinstance(task_type, TaskType):
        resolved = task_type
    elif task_type == "auto":
        if group_id is not None:
            resolved = TaskType.RANKING
        elif len(target_columns) > 1:
            resolved = _infer_multi_target_task_type(df, target_columns)
        else:
            resolved = _infer_task_type(df[target_columns[0]])
    else:
        try:
            resolved = TaskType(str(task_type))
        except ValueError as exc:
            raise DataValidationError(f"unknown task_type {task_type!r}") from exc

    _validate_task_shape(df, target_columns, group_id, resolved)
    return resolved, target_columns


def analyze_dataset(
    df: pd.DataFrame,
    target: str | list[str],
    group_id: str | None = None,
    task_type: str | TaskType = "auto",
    modalities: dict[str, str] | str | None = None,
) -> DatasetProfile:
    """Profile a training DataFrame. Decisions derived here must be train-only."""
    resolved_task, target_columns = resolve_task_and_targets(df, target, group_id, task_type)
    excluded = set(target_columns) | ({group_id} if group_id else set())
    feature_columns = [str(c) for c in df.columns if c not in excluded]
    if not feature_columns:
        raise DataValidationError("no feature columns remain after excluding target/group columns")

    columns = _classify_columns(df, feature_columns, modalities)
    leakage_reasons = _potential_leakage_columns(
        df,
        target_columns,
        feature_columns,
        columns["numeric"],
        resolved_task,
        set(columns["image"] + columns["audio"] + columns["video"]),
    )
    profile = DatasetProfile(
        task_type=resolved_task,
        target_column=target_columns[0],
        target_columns=target_columns,
        group_id_column=group_id,
        n_samples=len(df),
        n_features=len(feature_columns),
        feature_columns=feature_columns,
        dtypes={str(c): str(df[c].dtype) for c in df.columns},
        missing_report=_missing_report(df),
        duplicate_row_count=_duplicate_row_count(df),
        duplicate_column_groups=_duplicate_column_groups(df, feature_columns),
        constant_columns=_constant_columns(df, feature_columns),
        high_correlation_pairs=_high_correlation_pairs(df, columns["numeric"]),
        cardinality={c: int(df[c].nunique(dropna=True)) for c in columns["categorical"]},
        numeric_columns=columns["numeric"],
        categorical_columns=columns["categorical"],
        datetime_columns=columns["datetime"],
        text_columns=columns["text"],
        image_columns=columns["image"],
        audio_columns=columns["audio"],
        video_columns=columns["video"],
        column_modalities=columns["column_modalities"],
        id_like_columns=columns["id_like"],
        outlier_columns=_outlier_columns(df, columns["numeric"]),
        potential_leakage_columns=sorted(leakage_reasons),
        leakage_reasons=leakage_reasons,
    )

    if resolved_task in (TaskType.BINARY, TaskType.MULTICLASS):
        counts = df[target_columns[0]].value_counts(dropna=True)
        profile.class_distribution = {str(k): int(v) for k, v in counts.items()}
        ratio = float(counts.min() / counts.max()) if len(counts) else 1.0
        profile.class_imbalance_ratio = ratio
        profile.is_imbalanced = ratio < IMBALANCE_RATIO_THRESHOLD
    elif resolved_task is TaskType.MULTILABEL:
        positive_counts = {
            column: int(_binary_codes(df[column]).sum()) for column in target_columns
        }
        profile.class_distribution = positive_counts
        rates = [count / len(df) for count in positive_counts.values()] if len(df) else []
        if rates:
            positive_rates = [rate for rate in rates if rate > 0]
            ratio = min(positive_rates) / max(positive_rates) if positive_rates else 1.0
            profile.class_imbalance_ratio = ratio
            profile.is_imbalanced = ratio < IMBALANCE_RATIO_THRESHOLD
    if group_id is not None:
        group_sizes = df.groupby(group_id, sort=False).size()
        profile.n_groups = int(len(group_sizes))
        profile.group_size_stats = {
            "min": float(group_sizes.min()),
            "max": float(group_sizes.max()),
            "mean": float(group_sizes.mean()),
        }
    return profile


def _validate_frame(df: pd.DataFrame) -> None:
    if not isinstance(df, pd.DataFrame):
        raise DataValidationError(f"data must be a pandas DataFrame, got {type(df).__name__}")
    if df.empty:
        raise DataValidationError("data is empty")
    if not df.columns.is_unique:
        duplicates = df.columns[df.columns.duplicated()].tolist()
        raise DataValidationError(f"data contains duplicate column names: {duplicates}")


def _normalize_target_columns(target: str | list[str]) -> list[str]:
    if isinstance(target, str):
        return [target]
    if isinstance(target, (list, tuple)) and target and all(isinstance(item, str) for item in target):
        if len(set(target)) != len(target):
            raise DataValidationError("target contains duplicate column names")
        return list(target)
    raise DataValidationError(f"target must be a column name or non-empty list of names, got {target!r}")


def _validate_task_shape(
    df: pd.DataFrame,
    target_columns: list[str],
    group_id: str | None,
    task_type: TaskType,
) -> None:
    if task_type in (TaskType.MULTILABEL, TaskType.MULTIOUTPUT_REGRESSION):
        if len(target_columns) < 2:
            raise DataValidationError(f"task_type={task_type.value!r} requires multiple target columns")
    elif len(target_columns) != 1:
        raise DataValidationError(f"task_type={task_type.value!r} requires exactly one target column")

    if task_type is TaskType.RANKING:
        if group_id is None:
            raise DataValidationError("ranking requires group_id")
        if not pd.api.types.is_numeric_dtype(df[target_columns[0]]):
            raise DataValidationError("ranking target must be numeric relevance values")
    elif task_type is TaskType.MULTILABEL:
        invalid = [column for column in target_columns if not _looks_binary(df[column])]
        if invalid:
            raise DataValidationError(f"multilabel targets must be binary-like; invalid columns: {invalid}")
    elif task_type in (TaskType.REGRESSION, TaskType.MULTIOUTPUT_REGRESSION):
        invalid = [column for column in target_columns if not pd.api.types.is_numeric_dtype(df[column])]
        if invalid:
            raise DataValidationError(f"regression targets must be numeric; invalid columns: {invalid}")
    elif task_type is TaskType.BINARY and df[target_columns[0]].nunique(dropna=True) != 2:
        raise DataValidationError("binary classification requires exactly two target classes")
    elif task_type is TaskType.MULTICLASS and df[target_columns[0]].nunique(dropna=True) < 3:
        raise DataValidationError("multiclass classification requires at least three target classes")


def _infer_multi_target_task_type(df: pd.DataFrame, target_columns: list[str]) -> TaskType:
    if all(_looks_binary(df[column]) for column in target_columns):
        return TaskType.MULTILABEL
    if all(pd.api.types.is_numeric_dtype(df[column]) for column in target_columns):
        return TaskType.MULTIOUTPUT_REGRESSION
    raise DataValidationError(
        "multiple targets must be all binary-like or all numeric; set task_type explicitly after correcting target types"
    )


def _looks_binary(series: pd.Series) -> bool:
    non_null = series.dropna()
    if non_null.empty or non_null.nunique() != 2:
        return False
    if pd.api.types.is_bool_dtype(series):
        return True
    normalized = set(_normalize_series(non_null).unique())
    accepted = {
        frozenset({"0", "1"}),
        frozenset({"false", "true"}),
        frozenset({"no", "yes"}),
        frozenset({"negative", "positive"}),
    }
    return frozenset(normalized) in accepted or len(normalized) == 2


def _binary_codes(series: pd.Series) -> np.ndarray:
    values = _normalize_series(series)
    unique = sorted(values.unique().tolist())
    if len(unique) != 2:
        raise DataValidationError(f"expected a binary target, got values {unique[:10]}")
    positive_tokens = {"1", "true", "yes", "positive"}
    if unique[1] in positive_tokens:
        positive = unique[1]
    elif unique[0] in positive_tokens:
        positive = unique[0]
    else:
        positive = unique[1]
    return (values == positive).astype("int8").to_numpy()


def _infer_task_type(y: pd.Series) -> TaskType:
    non_null = y.dropna()
    if non_null.empty:
        raise DataValidationError("target has no non-missing values")
    n_unique = non_null.nunique()
    if n_unique < 2:
        raise DataValidationError("target must contain at least two distinct values")
    if pd.api.types.is_bool_dtype(y) or isinstance(y.dtype, pd.CategoricalDtype) or pd.api.types.is_object_dtype(y):
        return TaskType.BINARY if n_unique == 2 else TaskType.MULTICLASS
    if pd.api.types.is_numeric_dtype(y):
        return TaskType.BINARY if n_unique == 2 else TaskType.REGRESSION
    return TaskType.MULTICLASS


def _classify_columns(
    df: pd.DataFrame,
    feature_columns: list[str],
    modalities: dict[str, str] | str | None = None,
) -> dict[str, list[str] | dict[str, str]]:
    column_modalities = resolve_column_modalities(df, feature_columns, modalities)
    image_columns = [c for c, m in column_modalities.items() if m == "image"]
    audio_columns = [c for c, m in column_modalities.items() if m == "audio"]
    video_columns = [c for c, m in column_modalities.items() if m == "video"]
    text_columns = [c for c, m in column_modalities.items() if m == "text"]
    multimodal_columns = set(column_modalities)
    remaining = [column for column in feature_columns if column not in multimodal_columns]
    n_rows = len(df)
    datetime_columns: list[str] = []
    for column in list(remaining):
        series = df[column]
        if pd.api.types.is_datetime64_any_dtype(series):
            datetime_columns.append(column)
            remaining.remove(column)
        elif pd.api.types.is_object_dtype(series) or pd.api.types.is_string_dtype(series):
            sample = series.dropna().head(2000)
            if sample.empty:
                continue
            parsed = pd.to_datetime(sample, errors="coerce", format="mixed")
            if parsed.notna().mean() >= DATETIME_PARSE_SUCCESS_THRESHOLD:
                datetime_columns.append(column)
                remaining.remove(column)

    id_like_columns: list[str] = []
    for column in list(remaining):
        series = df[column]
        try:
            unique_ratio = series.nunique(dropna=False) / max(n_rows, 1)
        except TypeError:
            continue
        name = str(column).lower()
        name_suggests_id = name == "id" or name.endswith("_id") or name.startswith("id_") or "uuid" in name
        if n_rows > 20 and unique_ratio >= 0.995 and (
            name_suggests_id or pd.api.types.is_integer_dtype(series) or pd.api.types.is_object_dtype(series)
        ):
            id_like_columns.append(column)
            remaining.remove(column)

    for column in list(remaining):
        series = df[column]
        if pd.api.types.is_object_dtype(series) or pd.api.types.is_string_dtype(series):
            sample = series.dropna().astype(str).head(2000)
            if not sample.empty and float(sample.str.len().mean()) > TEXT_AVG_LEN_THRESHOLD:
                text_columns.append(column)
                column_modalities[column] = "text"
                multimodal_columns.add(column)
                remaining.remove(column)

    categorical_columns: list[str] = []
    numeric_columns: list[str] = []
    for column in remaining:
        series = df[column]
        if pd.api.types.is_bool_dtype(series) or pd.api.types.is_object_dtype(series) or isinstance(series.dtype, pd.CategoricalDtype):
            categorical_columns.append(column)
        elif pd.api.types.is_numeric_dtype(series):
            numeric_columns.append(column)
        else:
            categorical_columns.append(column)
    return {
        "datetime": datetime_columns,
        "id_like": id_like_columns,
        "text": text_columns,
        "image": image_columns,
        "audio": audio_columns,
        "video": video_columns,
        "multimodal": sorted(multimodal_columns),
        "column_modalities": column_modalities,
        "categorical": categorical_columns,
        "numeric": numeric_columns,
    }


def _normalize_series(series: pd.Series) -> pd.Series:
    return series.astype("string").fillna("<missing>").str.strip().str.casefold()


def _missing_report(df: pd.DataFrame) -> dict[str, float]:
    fractions = df.isna().mean()
    return {str(column): float(value) for column, value in fractions.items() if value > 0}



def _duplicate_row_count(df: pd.DataFrame) -> int:
    try:
        return int(df.duplicated().sum())
    except TypeError:
        normalized = df.copy()
        for column in normalized.columns:
            normalized[column] = normalized[column].map(
                lambda value: repr(value) if isinstance(value, (np.ndarray, list, dict, tuple)) else value
            )
        return int(normalized.duplicated().sum())

def _duplicate_column_groups(df: pd.DataFrame, feature_columns: list[str]) -> list[list[str]]:
    buckets: dict[bytes, list[str]] = {}
    for column in feature_columns:
        try:
            key = pd.util.hash_pandas_object(df[column], index=False).values.tobytes()
        except (TypeError, ValueError):
            continue
        buckets.setdefault(key, []).append(column)
    return [columns for columns in buckets.values() if len(columns) > 1]


def _constant_columns(df: pd.DataFrame, feature_columns: list[str]) -> list[str]:
    result: list[str] = []
    for column in feature_columns:
        try:
            if df[column].nunique(dropna=False) <= 1:
                result.append(column)
        except TypeError:
            continue
    return result


def _high_correlation_pairs(df: pd.DataFrame, numeric_columns: list[str]) -> list[tuple[str, str, float]]:
    if len(numeric_columns) < 2:
        return []
    corr = df[numeric_columns].corr(numeric_only=True).abs()
    pairs: list[tuple[str, str, float]] = []
    for index, left in enumerate(corr.columns):
        for right in corr.columns[index + 1 :]:
            value = corr.loc[left, right]
            if pd.notna(value) and value >= HIGH_CORR_THRESHOLD:
                pairs.append((str(left), str(right), float(value)))
    return pairs


def _outlier_columns(df: pd.DataFrame, numeric_columns: list[str]) -> dict[str, int]:
    result: dict[str, int] = {}
    for column in numeric_columns:
        values = df[column].dropna()
        if len(values) < 4:
            continue
        q1, q3 = values.quantile(0.25), values.quantile(0.75)
        iqr = q3 - q1
        if not np.isfinite(iqr) or iqr == 0:
            continue
        count = int(((values < q1 - 1.5 * iqr) | (values > q3 + 1.5 * iqr)).sum())
        if count:
            result[column] = count
    return result


def _potential_leakage_columns(
    df: pd.DataFrame,
    target_columns: list[str],
    feature_columns: list[str],
    numeric_columns: list[str],
    task_type: TaskType,
    modality_columns: set[str] | None = None,
) -> dict[str, list[str]]:
    reasons: dict[str, set[str]] = {}

    def add(column: str, reason: str) -> None:
        reasons.setdefault(column, set()).add(reason)

    for target in target_columns:
        normalized_target = _normalize_series(df[target])
        target_name = target.casefold()
        target_unique = normalized_target.nunique(dropna=False)
        if task_type in (TaskType.REGRESSION, TaskType.MULTIOUTPUT_REGRESSION, TaskType.RANKING):
            numeric_target = pd.to_numeric(df[target], errors="coerce")
        else:
            numeric_target = normalized_target.astype("category").cat.codes.astype(float)

        for column in feature_columns:
            if modality_columns and column in modality_columns:
                # Raw media payloads are not compared as strings against the target.
                continue
            normalized_feature = _normalize_series(df[column])
            if normalized_feature.equals(normalized_target):
                add(column, f"exact copy of target {target!r}")
                continue

            feature_unique = normalized_feature.nunique(dropna=False)
            # A one-to-one relationship is meaningful leakage evidence only when the
            # target is genuinely discrete and feature values repeat.  Applying this
            # test to continuous targets makes every unique numeric feature look like
            # a deterministic code for every unique target value.
            repeated_feature_values = feature_unique <= max(20, int(len(df) * 0.5))
            discrete_target = target_unique <= 50
            max_small_cardinality = max(20, target_unique * 5)
            if discrete_target and repeated_feature_values and feature_unique <= max_small_cardinality:
                feature_to_target = (
                    pd.DataFrame({"feature": normalized_feature, "target": normalized_target})
                    .groupby("feature", dropna=False)["target"]
                    .nunique(dropna=False)
                    .max()
                )
                target_to_feature = (
                    pd.DataFrame({"feature": normalized_feature, "target": normalized_target})
                    .groupby("target", dropna=False)["feature"]
                    .nunique(dropna=False)
                    .max()
                )
                if feature_to_target == 1 and target_to_feature == 1:
                    add(column, f"one-to-one deterministic mapping with target {target!r}")

            column_name = str(column).casefold()
            if len(target_name) >= 3 and (
                target_name in column_name or column_name in target_name or f"{target_name}_label" in column_name
            ):
                add(column, f"column name overlaps target name {target!r}")

        for column in numeric_columns:
            try:
                correlation = pd.to_numeric(df[column], errors="coerce").corr(numeric_target)
            except (TypeError, ValueError):
                continue
            if pd.notna(correlation) and abs(correlation) >= LEAKAGE_CORR_THRESHOLD:
                add(column, f"absolute target correlation {abs(correlation):.6f}")

    return {column: sorted(values) for column, values in sorted(reasons.items())}
