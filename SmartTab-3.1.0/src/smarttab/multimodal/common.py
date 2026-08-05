"""Shared feature extraction helpers."""

from __future__ import annotations

import hashlib
import math
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd


def entropy(values: np.ndarray, bins: int = 32) -> float:
    array = np.asarray(values)
    if array.size == 0:
        return 0.0
    if np.issubdtype(array.dtype, np.integer) and array.min(initial=0) >= 0 and array.max(initial=0) < 4096:
        counts = np.bincount(array.ravel().astype(int))
    else:
        counts, _ = np.histogram(array[np.isfinite(array)], bins=bins)
    counts = counts[counts > 0].astype(float)
    if counts.size == 0:
        return 0.0
    probabilities = counts / counts.sum()
    return float(-(probabilities * np.log2(probabilities)).sum())


def summarize(values: np.ndarray, prefix: str) -> dict[str, float]:
    array = np.asarray(values, dtype=float)
    array = array[np.isfinite(array)]
    if array.size == 0:
        return {f"{prefix}_{name}": 0.0 for name in ("mean", "std", "min", "max", "q25", "q50", "q75")}
    return {
        f"{prefix}_mean": float(array.mean()),
        f"{prefix}_std": float(array.std()),
        f"{prefix}_min": float(array.min()),
        f"{prefix}_max": float(array.max()),
        f"{prefix}_q25": float(np.quantile(array, 0.25)),
        f"{prefix}_q50": float(np.quantile(array, 0.50)),
        f"{prefix}_q75": float(np.quantile(array, 0.75)),
    }


def safe_float32_frame(matrix: np.ndarray, names: Iterable[str], index: pd.Index) -> pd.DataFrame:
    array = np.asarray(matrix, dtype=np.float32)
    array[~np.isfinite(array)] = np.nan
    return pd.DataFrame(array, columns=list(names), index=index)


def stable_signature(value: object) -> str:
    """Return a content-aware signature without materializing giant repr strings.

    Media caches must distinguish waveform/frame arrays that often have nearly
    identical truncated ``repr`` output. The incremental hasher also avoids a
    second full in-memory copy for large byte payloads.
    """
    hasher = hashlib.sha256()
    _update_signature(hasher, value)
    return hasher.hexdigest()


def _update_signature(hasher: Any, value: object) -> None:
    if value is None:
        hasher.update(b"none")
        return
    if isinstance(value, (bytes, bytearray, memoryview)):
        payload = memoryview(value)
        hasher.update(b"bytes|")
        hasher.update(str(len(payload)).encode())
        hasher.update(payload)
        return
    if isinstance(value, np.ndarray):
        array = np.ascontiguousarray(value)
        hasher.update(b"ndarray|")
        hasher.update(repr(array.shape).encode())
        hasher.update(str(array.dtype).encode())
        hasher.update(memoryview(array).cast("B"))
        return
    if isinstance(value, Path):
        _update_path_or_text(hasher, value)
        return
    if isinstance(value, str):
        _update_path_or_text(hasher, value)
        return
    if isinstance(value, dict):
        hasher.update(b"dict|")
        for key in sorted(value, key=lambda item: repr(item)):
            _update_signature(hasher, key)
            _update_signature(hasher, value[key])
        return
    if isinstance(value, (tuple, list)):
        hasher.update(type(value).__name__.encode() + b"|")
        hasher.update(str(len(value)).encode())
        for item in value:
            _update_signature(hasher, item)
        return
    if type(value).__module__.startswith("PIL"):
        hasher.update(b"pil|")
        _update_signature(hasher, np.asarray(value))
        return
    hasher.update(type(value).__qualname__.encode("utf-8", errors="replace"))
    hasher.update(b"|")
    hasher.update(repr(value).encode("utf-8", errors="replace"))


def _update_path_or_text(hasher: Any, value: str | Path) -> None:
    path = Path(value).expanduser()
    try:
        is_file = path.is_file()
    except OSError:
        is_file = False
    if is_file:
        stat = path.stat()
        hasher.update(b"file|")
        hasher.update(str(path.resolve()).encode("utf-8", errors="replace"))
        hasher.update(f"|{stat.st_size}|{stat.st_mtime_ns}".encode())
    else:
        hasher.update(b"text|")
        hasher.update(str(value).encode("utf-8", errors="replace"))


def pad_or_trim_features(values: np.ndarray, size: int) -> np.ndarray:
    array = np.asarray(values, dtype=np.float32).ravel()
    if len(array) >= size:
        return array[:size]
    return np.pad(array, (0, size - len(array)), constant_values=np.nan)


def linear_trend(values: np.ndarray) -> float:
    array = np.asarray(values, dtype=float)
    if len(array) < 2 or not np.isfinite(array).any():
        return 0.0
    x = np.arange(len(array), dtype=float)
    mask = np.isfinite(array)
    if mask.sum() < 2:
        return 0.0
    x = x[mask]
    y = array[mask]
    x = x - x.mean()
    denominator = float(np.dot(x, x))
    return float(np.dot(x, y - y.mean()) / denominator) if denominator else 0.0


def bounded_int(value: int | float, low: int, high: int) -> int:
    return int(max(low, min(high, math.floor(value))))
