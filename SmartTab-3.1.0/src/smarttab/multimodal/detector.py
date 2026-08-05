"""Conservative modality detection for raw DataFrame columns."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from smarttab.exceptions import ConfigurationError
from smarttab.multimodal.config import VALID_MODALITIES

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp", ".tif", ".tiff"}
AUDIO_EXTENSIONS = {".wav", ".flac", ".mp3", ".ogg", ".m4a", ".aac", ".wma"}
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".mpeg", ".mpg", ".m4v"}


def resolve_column_modalities(
    df: pd.DataFrame,
    feature_columns: list[str],
    declared: dict[str, str] | str | None,
) -> dict[str, str]:
    """Resolve explicit declarations first, then conservative content inference."""
    explicit: dict[str, str] = {}
    if declared in (None, "auto"):
        pass
    elif isinstance(declared, str):
        if declared not in VALID_MODALITIES:
            raise ConfigurationError(f"modality must be one of {VALID_MODALITIES} or 'auto'")
        if len(feature_columns) != 1:
            raise ConfigurationError("a scalar modality declaration requires exactly one feature column")
        explicit[feature_columns[0]] = declared
    elif isinstance(declared, dict):
        unknown_columns = sorted(set(declared) - set(feature_columns))
        if unknown_columns:
            raise ConfigurationError(f"modalities contains unknown feature columns: {unknown_columns}")
        for column, modality in declared.items():
            if modality not in VALID_MODALITIES:
                raise ConfigurationError(
                    f"modality for {column!r} must be one of {VALID_MODALITIES}, got {modality!r}"
                )
            explicit[str(column)] = modality
    else:
        raise ConfigurationError("modalities must be 'auto', a column-to-modality dictionary, or None")

    resolved: dict[str, str] = {}
    for column in feature_columns:
        if column in explicit:
            if explicit[column] != "tabular":
                resolved[column] = explicit[column]
            continue
        inferred = infer_series_modality(df[column])
        if inferred != "tabular":
            resolved[column] = inferred
    return resolved


def infer_series_modality(series: pd.Series) -> str:
    values = series.dropna().head(24).tolist()
    if not values:
        return "tabular"

    votes = {"image": 0, "audio": 0, "video": 0}
    for value in values:
        modality = infer_value_modality(value)
        if modality in votes:
            votes[modality] += 1
    threshold = max(1, int(np.ceil(len(values) * 0.7)))
    best = max(votes, key=votes.get)
    if votes[best] >= threshold:
        return best

    if pd.api.types.is_string_dtype(series) or series.dtype == object:
        text = series.dropna().astype(str)
        if len(text):
            avg_len = float(text.str.len().mean())
            unique_ratio = float(text.nunique() / max(1, len(text)))
            whitespace_ratio = float(text.str.contains(r"\s", regex=True).mean())
            if avg_len >= 30 or (avg_len >= 12 and unique_ratio >= 0.5 and whitespace_ratio >= 0.4):
                return "text"
    return "tabular"


def infer_value_modality(value: Any) -> str:
    if isinstance(value, (str, Path)):
        suffix = Path(str(value)).suffix.lower()
        if suffix in IMAGE_EXTENSIONS:
            return "image"
        if suffix in AUDIO_EXTENSIONS:
            return "audio"
        if suffix in VIDEO_EXTENSIONS:
            return "video"
        return "text" if len(str(value)) >= 30 else "tabular"
    if isinstance(value, np.ndarray):
        if value.ndim == 4:
            return "video"
        if value.ndim == 3 and value.shape[-1] in (1, 3, 4):
            return "image"
        if value.ndim in (1, 2):
            return "audio"
    if isinstance(value, tuple) and len(value) == 2:
        left, right = value
        if isinstance(left, (int, float)) and isinstance(right, np.ndarray):
            return "audio"
        if isinstance(right, (int, float)) and isinstance(left, np.ndarray):
            return "audio"
    if isinstance(value, list) and value:
        first = value[0]
        if isinstance(first, np.ndarray) and first.ndim in (2, 3):
            return "video"
    module = type(value).__module__
    name = type(value).__name__
    if module.startswith("PIL") and name == "Image":
        return "image"
    return "tabular"
