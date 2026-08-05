"""Train-only data-quality audit with machine-readable findings."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


@dataclass(slots=True)
class DataQualityIssue:
    code: str
    severity: str
    message: str
    columns: list[str] = field(default_factory=list)
    count: int | None = None
    fraction: float | None = None
    recommended_action: str | None = None


@dataclass(slots=True)
class DataQualityReport:
    n_rows: int
    n_columns: int
    memory_mb: float
    quality_score: float
    severity_counts: dict[str, int]
    issues: list[DataQualityIssue]
    missing_by_column: dict[str, float]
    row_missing_quantiles: dict[str, float]
    numeric_summary: dict[str, dict[str, float]]
    categorical_summary: dict[str, dict[str, float]]
    modality_summary: dict[str, dict[str, float]]
    target_summary: dict[str, Any]
    recommendations: list[str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def __getitem__(self, key: str) -> Any:
        """Provide lightweight dictionary-style access for interactive use."""
        if not hasattr(self, key):
            raise KeyError(key)
        return getattr(self, key)

    def get(self, key: str, default: Any = None) -> Any:
        return getattr(self, key, default)

    def items(self):
        return self.to_dict().items()


def audit_data_quality(
    frame: pd.DataFrame,
    *,
    target_columns: list[str] | None = None,
    column_modalities: dict[str, str] | None = None,
    sample_media: int = 64,
) -> DataQualityReport:
    target_columns = list(target_columns or [])
    column_modalities = dict(column_modalities or {})
    issues: list[DataQualityIssue] = []
    n_rows = len(frame)
    n_columns = frame.shape[1]
    feature_columns = [column for column in frame.columns if column not in target_columns]
    missing = {str(column): float(frame[column].isna().mean()) for column in frame.columns}
    row_missing = frame[feature_columns].isna().mean(axis=1) if feature_columns else pd.Series(0.0, index=frame.index)

    try:
        duplicate_count = int(frame.duplicated().sum())
    except TypeError:
        comparable = frame.copy()
        for column in comparable.columns:
            comparable[column] = comparable[column].map(_hashable_value)
        duplicate_count = int(comparable.duplicated().sum())
    if duplicate_count:
        issues.append(DataQualityIssue(
            "duplicate_rows", "warning", f"{duplicate_count} exact duplicate rows were detected",
            count=duplicate_count, fraction=duplicate_count / max(n_rows, 1),
            recommended_action="Drop exact duplicates before splitting to prevent train/test contamination.",
        ))

    high_missing = [column for column, ratio in missing.items() if ratio >= 0.5]
    if high_missing:
        issues.append(DataQualityIssue(
            "high_missing_columns", "warning", "Columns with at least 50% missing values were detected",
            columns=high_missing, recommended_action="Confirm that these columns are useful or remove them.",
        ))
    almost_empty_rows = int((row_missing >= 0.98).sum())
    if almost_empty_rows:
        issues.append(DataQualityIssue(
            "almost_empty_rows", "error", f"{almost_empty_rows} rows have at least 98% missing features",
            count=almost_empty_rows, fraction=almost_empty_rows / max(n_rows, 1),
            recommended_action="Drop these rows or recover their source data.",
        ))

    numeric_summary: dict[str, dict[str, float]] = {}
    categorical_summary: dict[str, dict[str, float]] = {}
    modality_summary: dict[str, dict[str, float]] = {}

    for column in feature_columns:
        series = frame[column]
        modality = column_modalities.get(str(column))
        if modality:
            summary = _modality_summary(series, modality, sample_media)
            modality_summary[str(column)] = summary
            if summary.get("missing_or_empty_rate", 0.0) > 0.1:
                issues.append(DataQualityIssue(
                    "modality_missing", "warning", f"Raw {modality} values are frequently missing or empty",
                    columns=[str(column)], fraction=summary["missing_or_empty_rate"],
                    recommended_action="Use missing-modality training and investigate ingestion failures.",
                ))
            if summary.get("unreadable_path_rate", 0.0) > 0:
                issues.append(DataQualityIssue(
                    "unreadable_media_paths", "error", f"Some sampled {modality} paths are unreadable",
                    columns=[str(column)], fraction=summary["unreadable_path_rate"],
                    recommended_action="Repair or remove broken media references.",
                ))
            continue

        numeric = pd.to_numeric(series, errors="coerce")
        numeric_ratio = float(numeric.notna().mean())
        if pd.api.types.is_numeric_dtype(series) or numeric_ratio >= 0.95:
            finite = numeric.replace([np.inf, -np.inf], np.nan).dropna()
            if finite.empty:
                numeric_summary[str(column)] = {"finite_rate": 0.0}
                continue
            q1, median, q3 = finite.quantile([0.25, 0.5, 0.75]).tolist()
            iqr = q3 - q1
            outliers = ((finite < q1 - 1.5 * iqr) | (finite > q3 + 1.5 * iqr)).mean() if iqr > 0 else 0.0
            skew = float(finite.skew()) if len(finite) > 2 else 0.0
            numeric_summary[str(column)] = {
                "finite_rate": float(finite.size / max(n_rows, 1)),
                "mean": float(finite.mean()), "std": float(finite.std(ddof=0)),
                "min": float(finite.min()), "q25": float(q1), "median": float(median),
                "q75": float(q3), "max": float(finite.max()), "skew": skew,
                "outlier_rate_iqr": float(outliers), "zero_rate": float((finite == 0).mean()),
            }
            infinite_count = int(np.isinf(numeric.to_numpy(dtype=float, na_value=np.nan)).sum())
            if infinite_count:
                issues.append(DataQualityIssue(
                    "infinite_values", "error", "Infinite numeric values were detected",
                    columns=[str(column)], count=infinite_count,
                    recommended_action="Replace infinities with missing values before imputation.",
                ))
            if abs(skew) >= 3.0:
                issues.append(DataQualityIssue(
                    "extreme_skew", "info", "A strongly skewed numeric distribution was detected",
                    columns=[str(column)], recommended_action="Consider log1p or Yeo-Johnson transformation.",
                ))
            if float(outliers) >= 0.1:
                issues.append(DataQualityIssue(
                    "many_outliers", "warning", "A high IQR-outlier rate was detected",
                    columns=[str(column)], fraction=float(outliers),
                    recommended_action="Validate units and consider train-only winsorization.",
                ))
        else:
            values = series.astype("string")
            counts = values.value_counts(dropna=True)
            cardinality = int(counts.size)
            rare_rate = float(counts[counts < max(2, math.ceil(n_rows * 0.01))].sum() / max(n_rows, 1))
            singleton_rate = float((counts == 1).sum() / max(cardinality, 1))
            categorical_summary[str(column)] = {
                "cardinality": float(cardinality),
                "cardinality_ratio": float(cardinality / max(n_rows, 1)),
                "rare_row_rate": rare_rate,
                "singleton_category_rate": singleton_rate,
                "top_category_rate": float(counts.iloc[0] / max(n_rows, 1)) if cardinality else 0.0,
            }
            if cardinality > 128 and cardinality / max(n_rows, 1) > 0.2:
                issues.append(DataQualityIssue(
                    "high_cardinality", "warning", "A high-cardinality categorical column was detected",
                    columns=[str(column)],
                    recommended_action="Group rare categories or verify that this is not an identifier.",
                ))
            mixed_types = series.dropna().map(lambda value: type(value).__name__).nunique()
            if mixed_types > 1:
                issues.append(DataQualityIssue(
                    "mixed_python_types", "info", "Multiple Python value types occur in one column",
                    columns=[str(column)], recommended_action="Normalize values before categorical encoding.",
                ))

    target_summary = _target_summary(frame, target_columns, feature_columns, issues)
    severity_counts = {name: sum(issue.severity == name for issue in issues) for name in ("error", "warning", "info")}
    score = max(0.0, 100.0 - 15.0 * severity_counts["error"] - 6.0 * severity_counts["warning"] - 1.5 * severity_counts["info"])
    recommendations = list(dict.fromkeys(
        issue.recommended_action for issue in issues if issue.recommended_action
    ))
    return DataQualityReport(
        n_rows=n_rows,
        n_columns=n_columns,
        memory_mb=float(frame.memory_usage(index=True, deep=True).sum() / 1024**2),
        quality_score=float(score),
        severity_counts=severity_counts,
        issues=issues,
        missing_by_column=missing,
        row_missing_quantiles={
            "q50": float(row_missing.quantile(0.5)),
            "q90": float(row_missing.quantile(0.9)),
            "q99": float(row_missing.quantile(0.99)),
            "max": float(row_missing.max()) if len(row_missing) else 0.0,
        },
        numeric_summary=numeric_summary,
        categorical_summary=categorical_summary,
        modality_summary=modality_summary,
        target_summary=target_summary,
        recommendations=recommendations,
    )


def _target_summary(frame: pd.DataFrame, targets: list[str], features: list[str], issues: list[DataQualityIssue]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for target in targets:
        if target not in frame:
            continue
        series = frame[target]
        counts = series.value_counts(dropna=False)
        result[target] = {
            "missing_rate": float(series.isna().mean()),
            "unique": int(series.nunique(dropna=True)),
            "distribution": {str(key): int(value) for key, value in counts.head(30).items()},
        }
    if targets and features and len(frame) <= 100_000:
        conflicting = 0
        comparable = frame[features + targets].copy()
        for column in features:
            comparable[column] = comparable[column].map(_hashable_value)
        grouped = comparable.groupby(features, dropna=False, sort=False)[targets].nunique(dropna=False)
        if isinstance(grouped, pd.Series):
            conflicting = int((grouped > 1).sum())
        else:
            conflicting = int((grouped > 1).any(axis=1).sum())
        if conflicting:
            issues.append(DataQualityIssue(
                "conflicting_duplicate_labels", "error",
                f"{conflicting} duplicate feature patterns have conflicting targets",
                count=conflicting,
                recommended_action="Resolve contradictory labels before model fitting.",
            ))
    return result


def _is_empty(value: object) -> bool:
    if value is None:
        return True
    if isinstance(value, float) and np.isnan(value):
        return True
    if isinstance(value, (str, bytes, bytearray)):
        return len(value.strip() if isinstance(value, str) else value) == 0
    if isinstance(value, np.ndarray):
        return value.size == 0
    return False


def _modality_summary(series: pd.Series, modality: str, sample_media: int) -> dict[str, float]:
    values = series.tolist()
    empty = np.array([_is_empty(value) for value in values], dtype=bool)
    sample = [value for value, is_empty in zip(values, empty, strict=True) if not is_empty][:sample_media]
    path_values = [value for value in sample if isinstance(value, (str, Path))]
    path_like = [Path(value).expanduser() for value in path_values if modality != "text"]
    unreadable = sum(not path.is_file() for path in path_like)
    summary = {
        "missing_or_empty_rate": float(empty.mean()) if len(empty) else 0.0,
        "sampled_values": float(len(sample)),
        "unreadable_path_rate": float(unreadable / max(len(path_like), 1)) if path_like else 0.0,
    }
    if modality == "text":
        lengths = np.asarray([len(str(value)) for value in sample], dtype=float)
        if lengths.size:
            summary.update({
                "length_mean": float(lengths.mean()),
                "length_q90": float(np.quantile(lengths, 0.9)),
                "empty_text_rate": float(empty.mean()),
            })
    return summary


def _hashable_value(value: object) -> object:
    if isinstance(value, np.ndarray):
        array = np.asarray(value)
        return ("ndarray", array.shape, str(array.dtype), hash(array.tobytes()))
    if isinstance(value, (list, tuple, dict, bytearray, memoryview)):
        return repr(value)
    return value
