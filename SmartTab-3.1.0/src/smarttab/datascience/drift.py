"""Reference-based dataset and transformed-feature drift monitoring."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np
import pandas as pd

_EPS = 1e-6


@dataclass(slots=True)
class NumericReference:
    edges: list[float]
    proportions: list[float]
    missing_rate: float
    median: float
    iqr: float


@dataclass(slots=True)
class CategoricalReference:
    frequencies: dict[str, float]
    missing_rate: float
    other_rate: float


@dataclass(slots=True)
class DriftReference:
    raw_numeric: dict[str, NumericReference] = field(default_factory=dict)
    raw_categorical: dict[str, CategoricalReference] = field(default_factory=dict)
    transformed_numeric: dict[str, NumericReference] = field(default_factory=dict)
    n_rows: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def fit(
        cls,
        raw_frame: pd.DataFrame,
        transformed: pd.DataFrame,
        *,
        numeric_columns: list[str],
        categorical_columns: list[str],
        max_categories: int = 50,
        bins: int = 10,
    ) -> "DriftReference":
        reference = cls(n_rows=len(raw_frame))
        for column in numeric_columns:
            if column in raw_frame:
                reference.raw_numeric[column] = _numeric_reference(raw_frame[column], bins)
        for column in categorical_columns:
            if column in raw_frame:
                reference.raw_categorical[column] = _categorical_reference(raw_frame[column], max_categories)
        for column in transformed.columns:
            if pd.api.types.is_numeric_dtype(transformed[column]):
                reference.transformed_numeric[str(column)] = _numeric_reference(transformed[column], bins)
        return reference


def compare_drift(
    reference: DriftReference,
    raw_frame: pd.DataFrame,
    transformed: pd.DataFrame,
) -> dict[str, Any]:
    raw_results: dict[str, dict[str, float | str]] = {}
    transformed_results: dict[str, dict[str, float | str]] = {}
    scores: list[float] = []

    for column, ref in reference.raw_numeric.items():
        if column not in raw_frame:
            raw_results[column] = {"severity": "critical", "missing_column": 1.0}
            scores.append(1.0)
            continue
        result = _compare_numeric(ref, raw_frame[column])
        raw_results[column] = result
        scores.append(float(result["score"]))
    for column, ref in reference.raw_categorical.items():
        if column not in raw_frame:
            raw_results[column] = {"severity": "critical", "missing_column": 1.0}
            scores.append(1.0)
            continue
        result = _compare_categorical(ref, raw_frame[column])
        raw_results[column] = result
        scores.append(float(result["score"]))
    for column, ref in reference.transformed_numeric.items():
        if column not in transformed:
            continue
        result = _compare_numeric(ref, transformed[column])
        transformed_results[column] = result

    transformed_scores = [float(item["score"]) for item in transformed_results.values()]
    all_scores = scores + transformed_scores
    overall = float(np.mean(sorted(all_scores, reverse=True)[: max(1, min(20, len(all_scores)))]) if all_scores else 0.0)
    severity = _severity(overall)
    return {
        "n_reference_rows": reference.n_rows,
        "n_current_rows": len(raw_frame),
        "overall_score": overall,
        "severity": severity,
        "raw_columns": raw_results,
        "transformed_features": transformed_results,
        "drifted_raw_columns": sorted(
            [column for column, result in raw_results.items() if result.get("severity") in {"warning", "critical"}]
        ),
        "drifted_feature_count": int(sum(
            result.get("severity") in {"warning", "critical"} for result in transformed_results.values()
        )),
    }


def _numeric_reference(series: pd.Series, bins: int) -> NumericReference:
    numeric = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan)
    missing_rate = float(numeric.isna().mean())
    values = numeric.dropna().to_numpy(dtype=float)
    if values.size == 0:
        return NumericReference(edges=[-1.0, 1.0], proportions=[1.0], missing_rate=missing_rate, median=0.0, iqr=1.0)
    quantiles = np.linspace(0, 1, bins + 1)
    edges = np.unique(np.quantile(values, quantiles))
    if edges.size < 2:
        center = float(values[0])
        edges = np.array([center - 0.5, center + 0.5])
    edges[0] = -np.inf
    edges[-1] = np.inf
    counts, _ = np.histogram(values, bins=edges)
    proportions = (counts / max(counts.sum(), 1)).astype(float)
    q1, median, q3 = np.quantile(values, [0.25, 0.5, 0.75])
    return NumericReference(
        edges=[float(value) for value in edges],
        proportions=[float(value) for value in proportions],
        missing_rate=missing_rate,
        median=float(median),
        iqr=float(max(q3 - q1, _EPS)),
    )


def _categorical_reference(series: pd.Series, max_categories: int) -> CategoricalReference:
    missing_rate = float(series.isna().mean())
    values = series.astype("string").fillna("__missing__")
    frequencies = values.value_counts(normalize=True)
    top = frequencies.head(max_categories)
    known = {str(key): float(value) for key, value in top.items()}
    return CategoricalReference(
        frequencies=known,
        missing_rate=missing_rate,
        other_rate=float(max(0.0, 1.0 - sum(known.values()))),
    )


def _compare_numeric(reference: NumericReference, series: pd.Series) -> dict[str, float | str]:
    numeric = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan)
    missing_rate = float(numeric.isna().mean())
    values = numeric.dropna().to_numpy(dtype=float)
    if values.size:
        counts, _ = np.histogram(values, bins=np.asarray(reference.edges, dtype=float))
        current = counts / max(counts.sum(), 1)
        psi = _psi(np.asarray(reference.proportions), current)
        median_shift = abs(float(np.median(values)) - reference.median) / max(reference.iqr, _EPS)
    else:
        psi = 1.0
        median_shift = 5.0
    missing_shift = abs(missing_rate - reference.missing_rate)
    score = float(min(1.0, 0.65 * min(psi / 0.5, 1.0) + 0.20 * min(median_shift / 3.0, 1.0) + 0.15 * min(missing_shift / 0.25, 1.0)))
    return {
        "score": score,
        "severity": _severity(score),
        "psi": float(psi),
        "median_iqr_shift": float(median_shift),
        "missing_rate": missing_rate,
        "missing_rate_delta": float(missing_rate - reference.missing_rate),
    }


def _compare_categorical(reference: CategoricalReference, series: pd.Series) -> dict[str, float | str]:
    values = series.astype("string").fillna("__missing__")
    current_counts = values.value_counts(normalize=True)
    keys = list(reference.frequencies)
    ref = np.asarray([reference.frequencies[key] for key in keys] + [reference.other_rate], dtype=float)
    current_known = [float(current_counts.get(key, 0.0)) for key in keys]
    other = float(max(0.0, 1.0 - sum(current_known)))
    cur = np.asarray(current_known + [other], dtype=float)
    js = _jensen_shannon(ref, cur)
    unknown_rate = float(sum(value for key, value in current_counts.items() if str(key) not in reference.frequencies))
    missing_rate = float(series.isna().mean())
    missing_shift = abs(missing_rate - reference.missing_rate)
    score = float(min(1.0, 0.7 * min(js / 0.35, 1.0) + 0.2 * min(unknown_rate / 0.25, 1.0) + 0.1 * min(missing_shift / 0.25, 1.0)))
    return {
        "score": score,
        "severity": _severity(score),
        "jensen_shannon": float(js),
        "unknown_or_other_rate": unknown_rate,
        "missing_rate": missing_rate,
        "missing_rate_delta": float(missing_rate - reference.missing_rate),
    }


def _psi(expected: np.ndarray, actual: np.ndarray) -> float:
    expected = np.clip(expected.astype(float), _EPS, None)
    actual = np.clip(actual.astype(float), _EPS, None)
    expected /= expected.sum()
    actual /= actual.sum()
    return float(np.sum((actual - expected) * np.log(actual / expected)))


def _jensen_shannon(left: np.ndarray, right: np.ndarray) -> float:
    left = np.clip(left.astype(float), _EPS, None); left /= left.sum()
    right = np.clip(right.astype(float), _EPS, None); right /= right.sum()
    middle = 0.5 * (left + right)
    kl_left = np.sum(left * np.log(left / middle))
    kl_right = np.sum(right * np.log(right / middle))
    return float(np.sqrt(max(0.0, 0.5 * (kl_left + kl_right))))


def _severity(score: float) -> str:
    if score >= 0.65:
        return "critical"
    if score >= 0.30:
        return "warning"
    return "ok"
