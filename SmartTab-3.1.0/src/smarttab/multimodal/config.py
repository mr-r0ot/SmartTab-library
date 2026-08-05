"""Configuration primitives for bounded multimodal feature extraction."""

from __future__ import annotations

from dataclasses import dataclass, field
import os
from typing import Any

from smarttab.exceptions import ConfigurationError

VALID_MODALITIES = ("tabular", "text", "image", "audio", "video")
VALID_MULTIMODAL_BACKENDS = ("auto", "classical", "pretrained", "hybrid")
VALID_MEDIA_ERROR_POLICIES = ("error", "warn", "zero")
VALID_ADAPTATION_MODES = ("auto", "none", "pls")

_DEFAULT_WEIGHTS = {
    "text": 0.40,
    "image": 0.24,
    "audio": 0.18,
    "video": 0.18,
}


@dataclass(slots=True)
class FeatureSpaceConfig:
    """Resolved feature-space controls.

    ``total_features`` is a hard upper bound for generated multimodal features.
    Raw numeric/categorical tabular columns are not counted against this budget.
    """

    total_features: int
    modality_limits: dict[str, int] = field(default_factory=dict)
    column_limits: dict[str, int] = field(default_factory=dict)
    speed_accuracy: float = 0.5
    backend: str = "auto"
    allow_model_download: bool = False
    error_policy: str = "warn"
    batch_size: int = 32
    workers: int = 1
    cache: bool | str = False
    modality_params: dict[str, dict[str, Any]] = field(default_factory=dict)
    random_state: int = 42
    device: str = "auto"
    supervised_adaptation: str = "auto"
    adapter_features: str | int = "auto"
    adapter_min_samples: int = 80

    def limit_for(self, modality: str, present_modalities: list[str]) -> int:
        if modality in self.modality_limits:
            return max(1, min(self.total_features, int(self.modality_limits[modality])))
        active = [name for name in _DEFAULT_WEIGHTS if name in present_modalities]
        denominator = sum(_DEFAULT_WEIGHTS[name] for name in active) or 1.0
        share = _DEFAULT_WEIGHTS.get(modality, 0.1) / denominator
        return max(16, int(round(self.total_features * share)))


def resolve_feature_space_config(
    feature_budget: str | int | dict[str, int] | FeatureSpaceConfig,
    *,
    speed_accuracy: float,
    backend: str,
    allow_model_download: bool,
    error_policy: str,
    batch_size: str | int,
    cache: bool | str,
    modality_params: dict[str, dict[str, Any]] | None,
    random_state: int,
    workers: str | int = "auto",
    device: str = "auto",
    supervised_adaptation: str = "auto",
    adapter_features: str | int = "auto",
) -> FeatureSpaceConfig:
    if isinstance(feature_budget, FeatureSpaceConfig):
        _validate_resolved_config(feature_budget)
        return feature_budget
    if not 0.0 <= float(speed_accuracy) <= 1.0:
        raise ConfigurationError("speed_accuracy must be between 0.0 (speed) and 1.0 (accuracy)")
    if backend not in VALID_MULTIMODAL_BACKENDS:
        raise ConfigurationError(f"multimodal_backend must be one of {VALID_MULTIMODAL_BACKENDS}")
    if error_policy not in VALID_MEDIA_ERROR_POLICIES:
        raise ConfigurationError(f"media_error_policy must be one of {VALID_MEDIA_ERROR_POLICIES}")
    if not isinstance(allow_model_download, bool):
        raise ConfigurationError("allow_model_download must be True or False")
    if not isinstance(cache, (bool, str)):
        raise ConfigurationError("feature_cache must be True, False, or a cache directory path")
    if supervised_adaptation not in VALID_ADAPTATION_MODES:
        raise ConfigurationError(f"supervised_adaptation must be one of {VALID_ADAPTATION_MODES}")
    if adapter_features != "auto":
        try:
            resolved_adapter_features = int(adapter_features)
        except (TypeError, ValueError) as exc:
            raise ConfigurationError("adapter_features must be 'auto' or a positive integer") from exc
        if not 1 <= resolved_adapter_features <= 512:
            raise ConfigurationError("adapter_features must be between 1 and 512")
    else:
        resolved_adapter_features = "auto"
    if modality_params is not None and not isinstance(modality_params, dict):
        raise ConfigurationError("modality_params must be a dictionary or None")

    if feature_budget == "auto":
        total = int(round(256 + 768 * float(speed_accuracy)))
        limits: dict[str, int] = {}
        column_limits: dict[str, int] = {}
    elif isinstance(feature_budget, int):
        total = feature_budget
        limits = {}
        column_limits = {}
    elif isinstance(feature_budget, dict):
        total = int(feature_budget.get("total", sum(int(v) for k, v in feature_budget.items() if k != "total")))
        limits = {k: int(v) for k, v in feature_budget.items() if k in _DEFAULT_WEIGHTS}
        column_limits = {k: int(v) for k, v in feature_budget.items() if k not in {"total", *_DEFAULT_WEIGHTS}}
    else:
        raise ConfigurationError("feature_budget must be 'auto', a positive integer, or a modality dictionary")
    if not 32 <= int(total) <= 16384:
        raise ConfigurationError("feature_budget total must be between 32 and 16384")
    if any(value < 8 for value in [*limits.values(), *column_limits.values()]):
        raise ConfigurationError("each modality/column feature budget must be at least 8")
    if any(value > total for value in [*limits.values(), *column_limits.values()]):
        raise ConfigurationError("a modality/column feature budget cannot exceed total")

    if batch_size == "auto":
        resolved_batch = max(4, int(round(8 + (1.0 - float(speed_accuracy)) * 56)))
    else:
        try:
            resolved_batch = int(batch_size)
        except (TypeError, ValueError) as exc:
            raise ConfigurationError("batch_size must be 'auto' or a positive integer") from exc
        if not 1 <= resolved_batch <= 4096:
            raise ConfigurationError("batch_size must be between 1 and 4096")

    if workers == "auto":
        resolved_workers = max(1, min(8, (os.cpu_count() or 2) // 2))
    else:
        try:
            resolved_workers = int(workers)
        except (TypeError, ValueError) as exc:
            raise ConfigurationError("feature_workers must be 'auto' or a positive integer") from exc
        if not 1 <= resolved_workers <= 64:
            raise ConfigurationError("feature_workers must be between 1 and 64")

    return FeatureSpaceConfig(
        total_features=int(total),
        modality_limits=limits,
        column_limits=column_limits,
        speed_accuracy=float(speed_accuracy),
        backend=backend,
        allow_model_download=allow_model_download,
        error_policy=error_policy,
        batch_size=resolved_batch,
        workers=resolved_workers,
        cache=cache,
        modality_params=modality_params or {},
        random_state=random_state,
        device=device,
        supervised_adaptation=supervised_adaptation,
        adapter_features=resolved_adapter_features,
    )


def _validate_resolved_config(config: FeatureSpaceConfig) -> None:
    if not 32 <= int(config.total_features) <= 16384:
        raise ConfigurationError("FeatureSpaceConfig.total_features must be between 32 and 16384")
    if not 0.0 <= float(config.speed_accuracy) <= 1.0:
        raise ConfigurationError("FeatureSpaceConfig.speed_accuracy must be between 0.0 and 1.0")
    if config.backend not in VALID_MULTIMODAL_BACKENDS:
        raise ConfigurationError(
            f"FeatureSpaceConfig.backend must be one of {VALID_MULTIMODAL_BACKENDS}"
        )
    if config.supervised_adaptation not in VALID_ADAPTATION_MODES:
        raise ConfigurationError(
            f"FeatureSpaceConfig.supervised_adaptation must be one of {VALID_ADAPTATION_MODES}"
        )
    if config.adapter_features != "auto" and not 1 <= int(config.adapter_features) <= 512:
        raise ConfigurationError("FeatureSpaceConfig.adapter_features must be 'auto' or between 1 and 512")
    if int(config.adapter_min_samples) < 20:
        raise ConfigurationError("FeatureSpaceConfig.adapter_min_samples must be at least 20")
    if config.error_policy not in VALID_MEDIA_ERROR_POLICIES:
        raise ConfigurationError(
            f"FeatureSpaceConfig.error_policy must be one of {VALID_MEDIA_ERROR_POLICIES}"
        )
    if not 1 <= int(config.batch_size) <= 4096:
        raise ConfigurationError("FeatureSpaceConfig.batch_size must be between 1 and 4096")
    if not 1 <= int(config.workers) <= 64:
        raise ConfigurationError("FeatureSpaceConfig.workers must be between 1 and 64")
    limits = [*config.modality_limits.values(), *config.column_limits.values()]
    if any(int(value) < 8 for value in limits):
        raise ConfigurationError("FeatureSpaceConfig feature limits must be at least 8")
    if any(int(value) > int(config.total_features) for value in limits):
        raise ConfigurationError(
            "FeatureSpaceConfig modality/column limits cannot exceed total_features"
        )
