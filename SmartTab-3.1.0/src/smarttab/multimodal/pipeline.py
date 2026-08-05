"""Fitted multimodal feature pipeline with a hard feature-space budget."""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
import warnings

import numpy as np
import pandas as pd
from sklearn.feature_selection import f_classif, f_regression

from smarttab.analysis.dataset_analyzer import TaskType
from smarttab.exceptions import DataValidationError
from smarttab.multimodal.adaptation import SupervisedFeatureAdapter
from smarttab.multimodal.audio import AudioFeatureExtractor
from smarttab.multimodal.base import BaseFeatureExtractor
from smarttab.multimodal.common import stable_signature
from smarttab.multimodal.config import FeatureSpaceConfig
from smarttab.multimodal.image import ImageFeatureExtractor
from smarttab.multimodal.text import TextFeatureExtractor
from smarttab.multimodal.video import VideoFeatureExtractor

_EXTRACTORS = {
    "text": TextFeatureExtractor,
    "image": ImageFeatureExtractor,
    "audio": AudioFeatureExtractor,
    "video": VideoFeatureExtractor,
}


@dataclass
class ModalityFeatureReport:
    column_modalities: dict[str, str] = field(default_factory=dict)
    allocated_features: dict[str, int] = field(default_factory=dict)
    generated_features: dict[str, int] = field(default_factory=dict)
    backend_used: dict[str, str] = field(default_factory=dict)
    adapted_features: dict[str, int] = field(default_factory=dict)
    selected_by_column: dict[str, int] = field(default_factory=dict)
    adapter_diagnostics: dict[str, dict[str, Any]] = field(default_factory=dict)
    errors: dict[str, list[str]] = field(default_factory=dict)
    selected_feature_count: int = 0
    dropped_by_global_budget: int = 0
    notes: list[str] = field(default_factory=list)


class MultiModalFeaturePipeline:
    """Convert raw modality columns into a bounded dense float32 feature frame."""

    def __init__(
        self,
        config: FeatureSpaceConfig,
        column_modalities: dict[str, str],
        task_type: TaskType,
    ) -> None:
        self.config = config
        self.column_modalities = dict(column_modalities)
        self.task_type = task_type
        self.extractors_: dict[str, BaseFeatureExtractor] = {}
        self.adapters_: dict[str, SupervisedFeatureAdapter] = {}
        self.local_selected_columns_: dict[str, list[str]] = {}
        self.feature_columns_: list[str] = []
        self.selected_columns_: list[str] = []
        self.report_ = ModalityFeatureReport(column_modalities=dict(column_modalities))
        self._memory_cache: OrderedDict[str, np.ndarray] = OrderedDict()
        self._fitted = False

    @property
    def source_columns(self) -> list[str]:
        return list(self.column_modalities)

    def fit_transform(self, df: pd.DataFrame, y: pd.Series | pd.DataFrame | None = None) -> pd.DataFrame:
        if not self.column_modalities:
            self._fitted = True
            return pd.DataFrame(index=df.index)
        present_modalities = sorted(set(self.column_modalities.values()))
        counts = {name: list(self.column_modalities.values()).count(name) for name in present_modalities}
        pieces: list[pd.DataFrame] = []
        for column, modality in self.column_modalities.items():
            group_limit = self.config.limit_for(modality, present_modalities)
            column_limit = self.config.column_limits.get(
                column, max(8, group_limit // max(1, counts[modality]))
            )
            params = dict(self.config.modality_params.get(modality, {}))
            params.update(self.config.modality_params.get(column, {}))
            requested_max = int(params.pop("max_features", column_limit))
            column_limit = max(8, min(column_limit, requested_max, self.config.total_features))
            extractor_class = _EXTRACTORS[modality]
            extractor_kwargs = {
                "max_features": column_limit,
                "backend": self.config.backend,
                "speed_accuracy": self.config.speed_accuracy,
                "allow_model_download": self.config.allow_model_download,
                "batch_size": self.config.batch_size,
                "workers": self.config.workers,
                "random_state": self.config.random_state,
                "error_policy": self.config.error_policy,
                "device": self.config.device,
            }
            extractor_kwargs.update(params)
            extractor = extractor_class(**extractor_kwargs)
            base_frame = extractor.fit_transform(df[column], y)
            self.extractors_[column] = extractor
            self._cache_rows(column, df[column], base_frame)
            frame = base_frame.copy()
            adapter = self._fit_adapter(column, frame, y, column_limit)
            if adapter is not None:
                adapted = adapter.transform(frame)
                frame = pd.concat([frame, adapted], axis=1)
                self.adapters_[column] = adapter
                self.report_.adapted_features[column] = adapted.shape[1]
                self.report_.adapter_diagnostics[column] = dict(adapter.diagnostics_)
            frame["source_missing"] = _source_missing_mask(df[column]).astype(np.float32)
            local_selected = self._rank_within_budget(frame, y, column_limit)
            self.local_selected_columns_[column] = local_selected
            self.report_.selected_by_column[column] = len(local_selected)
            self.report_.generated_features[column] = frame.shape[1]
            frame = frame.loc[:, local_selected]
            frame = frame.rename(columns={name: f"mm__{column}__{name}" for name in frame.columns})
            self.report_.allocated_features[column] = column_limit
            self.report_.backend_used[column] = getattr(extractor, "backend_used_", "classical")
            if extractor.errors_:
                self.report_.errors[column] = list(extractor.errors_)
            self.report_.notes.extend(getattr(extractor, "notes_", []))
            pieces.append(frame)
        result = pd.concat(pieces, axis=1) if pieces else pd.DataFrame(index=df.index)
        self.feature_columns_ = list(result.columns)
        self.selected_columns_ = self._select_within_budget(result, y)
        self.report_.selected_feature_count = len(self.selected_columns_)
        self.report_.dropped_by_global_budget = len(self.feature_columns_) - len(self.selected_columns_)
        self._fitted = True
        return result.loc[:, self.selected_columns_].astype(np.float32)

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self._fitted:
            raise RuntimeError("MultiModalFeaturePipeline must be fitted before transform()")
        if not self.column_modalities:
            return pd.DataFrame(index=df.index)
        pieces: list[pd.DataFrame] = []
        for column, modality in self.column_modalities.items():
            if column not in df.columns:
                raise DataValidationError(f"multimodal source column {column!r} is missing")
            extractor = self.extractors_[column]
            frame = self._transform_with_cache(column, df[column], extractor, modality)
            adapter = self.adapters_.get(column)
            if adapter is not None:
                frame = pd.concat([frame, adapter.transform(frame)], axis=1)
            frame["source_missing"] = _source_missing_mask(df[column]).astype(np.float32)
            local_columns = self.local_selected_columns_.get(column, list(frame.columns))
            missing_local = [name for name in local_columns if name not in frame]
            if missing_local:
                raise DataValidationError(
                    f"multimodal transform did not reproduce {column!r} features: {missing_local}"
                )
            frame = frame.loc[:, local_columns]
            frame = frame.rename(columns={name: f"mm__{column}__{name}" for name in frame.columns})
            pieces.append(frame)
        result = pd.concat(pieces, axis=1)
        missing = [name for name in self.selected_columns_ if name not in result]
        if missing:
            raise DataValidationError(f"multimodal transform did not reproduce features: {missing}")
        return result.loc[:, self.selected_columns_].astype(np.float32)

    def _select_within_budget(
        self,
        frame: pd.DataFrame,
        y: pd.Series | pd.DataFrame | None,
    ) -> list[str]:
        return self._rank_within_budget(frame, y, self.config.total_features)

    def _rank_within_budget(
        self,
        frame: pd.DataFrame,
        y: pd.Series | pd.DataFrame | None,
        budget: int,
    ) -> list[str]:
        budget = max(1, int(budget))
        if frame.shape[1] <= budget:
            return list(frame.columns)
        matrix = frame.replace([np.inf, -np.inf], np.nan)
        matrix = matrix.fillna(matrix.median(numeric_only=True)).fillna(0.0)
        values = matrix.to_numpy(dtype=np.float64)
        variances = np.nanvar(values, axis=0)
        scores = np.log1p(np.maximum(variances, 0.0))
        if y is not None and len(y) == len(frame):
            try:
                target_frame = y.to_frame() if isinstance(y, pd.Series) else pd.DataFrame(y)
                supervised_scores = []
                informative = np.isfinite(variances) & (variances > 1e-12)
                for target_column in target_frame.columns:
                    statistic = np.zeros(values.shape[1], dtype=np.float64)
                    if not informative.any():
                        supervised_scores.append(statistic)
                        continue
                    target = target_frame[target_column]
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore", RuntimeWarning)
                        warnings.simplefilter("ignore", UserWarning)
                        if self.task_type.is_classification:
                            target_values = pd.factorize(target.astype("string"), sort=True)[0]
                            partial, _ = f_classif(values[:, informative], target_values)
                        else:
                            target_values = pd.to_numeric(target, errors="coerce").fillna(0).to_numpy()
                            partial, _ = f_regression(values[:, informative], target_values)
                    statistic[informative] = np.nan_to_num(
                        partial, nan=0.0, posinf=0.0, neginf=0.0
                    )
                    supervised_scores.append(statistic)
                if supervised_scores:
                    supervised = np.max(np.vstack(supervised_scores), axis=0)
                    scores = scores + np.log1p(np.maximum(supervised, 0.0))
            except Exception as exc:
                self.report_.notes.append(
                    f"supervised multimodal feature ranking fell back to variance: {exc}"
                )
        mandatory = [
            index for index, name in enumerate(frame.columns)
            if name.endswith("source_missing")
        ][:budget]
        mandatory_set = set(mandatory)
        remaining_budget = max(0, budget - len(mandatory))
        candidate_order = [
            int(index) for index in np.argsort(-scores, kind="stable")
            if int(index) not in mandatory_set
        ][:remaining_budget]
        selected_indices = sorted(mandatory + candidate_order)
        return [str(frame.columns[index]) for index in selected_indices]

    def _fit_adapter(
        self,
        column: str,
        frame: pd.DataFrame,
        y: pd.Series | pd.DataFrame | None,
        column_limit: int,
    ) -> SupervisedFeatureAdapter | None:
        mode = self.config.supervised_adaptation
        if mode == "none" or y is None or len(frame) < self.config.adapter_min_samples:
            return None
        if mode == "auto" and self.config.speed_accuracy < 0.60:
            return None
        if frame.shape[1] < 4:
            return None
        if self.config.adapter_features == "auto":
            components = max(2, min(24, column_limit // 5, frame.shape[1] // 3))
        else:
            components = min(int(self.config.adapter_features), column_limit, frame.shape[1])
        if components < 1:
            return None
        adapter = SupervisedFeatureAdapter(
            task_type=self.task_type,
            n_components=components,
            max_fit_rows=max(1000, int(5_000 + 20_000 * self.config.speed_accuracy)),
            random_state=self.config.random_state,
        ).fit(frame, y)
        if not adapter.fitted:
            return None
        self.report_.notes.append(
            f"{column}: added {len(adapter.feature_names_)} bounded supervised adapter features"
        )
        return adapter

    def _cache_enabled(self, modality: str) -> bool:
        return bool(self.config.cache) and modality in {"image", "audio", "video"}

    def _cache_key(self, column: str, value: object, extractor: BaseFeatureExtractor) -> str:
        signature = stable_signature(value)
        model_name = getattr(extractor, "model_name", "")
        return f"{column}|{type(extractor).__name__}|{extractor.max_features}|{model_name}|{signature}"

    def _cache_rows(self, column: str, values: pd.Series, frame: pd.DataFrame) -> None:
        modality = self.column_modalities[column]
        if not self._cache_enabled(modality):
            return
        extractor = self.extractors_[column]
        for (_, value), row in zip(values.items(), frame.to_numpy(), strict=True):
            self._cache_put(self._cache_key(column, value, extractor), row)

    def _transform_with_cache(
        self,
        column: str,
        values: pd.Series,
        extractor: BaseFeatureExtractor,
        modality: str,
    ) -> pd.DataFrame:
        if not self._cache_enabled(modality):
            return extractor.transform(values)
        rows: list[np.ndarray | None] = []
        missing_positions: list[int] = []
        missing_values: list[object] = []
        keys: list[str] = []
        for position, value in enumerate(values.tolist()):
            key = self._cache_key(column, value, extractor)
            keys.append(key)
            cached = self._cache_get(key)
            rows.append(cached)
            if cached is None:
                missing_positions.append(position)
                missing_values.append(value)
        if missing_values:
            missing_frame = extractor.transform(pd.Series(missing_values, dtype=object)).to_numpy()
            for position, row in zip(missing_positions, missing_frame, strict=True):
                rows[position] = row
                self._cache_put(keys[position], row)
        return pd.DataFrame(np.vstack(rows), columns=extractor.feature_names_, index=values.index)

    def _cache_get(self, key: str) -> np.ndarray | None:
        if isinstance(self.config.cache, str):
            path = Path(self.config.cache).expanduser() / f"{stable_signature(key)}.npy"
            if path.exists():
                try:
                    return np.load(path, allow_pickle=False)
                except Exception:
                    path.unlink(missing_ok=True)
        row = self._memory_cache.get(key)
        if row is not None:
            self._memory_cache.move_to_end(key)
        return row

    def _cache_put(self, key: str, row: np.ndarray) -> None:
        array = np.asarray(row, dtype=np.float32)
        if isinstance(self.config.cache, str):
            directory = Path(self.config.cache).expanduser()
            directory.mkdir(parents=True, exist_ok=True)
            path = directory / f"{stable_signature(key)}.npy"
            if not path.exists():
                np.save(path, array, allow_pickle=False)
        self._memory_cache[key] = array
        self._memory_cache.move_to_end(key)
        while len(self._memory_cache) > 4096:
            self._memory_cache.popitem(last=False)

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        state["_memory_cache"] = OrderedDict()
        return state


def _source_missing_mask(values: pd.Series) -> np.ndarray:
    def missing(value: object) -> bool:
        if value is None:
            return True
        if isinstance(value, float) and np.isnan(value):
            return True
        if isinstance(value, str):
            return not value.strip()
        if isinstance(value, (bytes, bytearray, memoryview)):
            return len(value) == 0
        if isinstance(value, np.ndarray):
            return value.size == 0
        return False
    return np.asarray([missing(value) for value in values.tolist()], dtype=np.float32)
