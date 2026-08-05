"""High-level data-science controls used by SmartTab's automatic pipeline."""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from typing import Any

from smarttab.exceptions import ConfigurationError


@dataclass(slots=True)
class DataScienceConfig:
    """Advanced controls with conservative automatic defaults.

    The object is intentionally compact: beginners can leave it at ``"auto"``;
    specialists can override individual policies through a dictionary or an
    explicit instance.
    """

    quality_audit: bool = True
    quality_policy: str = "warn"  # warn | strict | report
    conflicting_labels: str = "warn"  # warn | drop | error | keep
    row_missing_threshold: float = 0.98
    numeric_imputation: str = "median"  # median | mean | constant | knn | iterative
    numeric_fill_value: float = 0.0
    add_missing_indicators: bool = True
    rare_category_min_frequency: float = 0.01
    rare_category_max_categories: int = 128
    numeric_transform: str = "auto"  # auto | none | log1p | yeo-johnson
    skew_threshold: float = 1.5
    max_model_features: str | int = "auto"
    winsorize: str | float = "auto"  # auto | none | fraction in (0,.25)
    calibration: str = "auto"  # auto | none | sigmoid | isotonic
    calibration_fraction: float = 0.15
    conformal: str | bool = "auto"
    conformal_alpha: float = 0.10
    ood_detection: bool = True
    drift_monitoring: bool = True
    modality_dropout: str | float = "auto"
    modality_dropout_max_expansion: float = 0.20

    def __post_init__(self) -> None:
        if self.quality_policy not in {"warn", "strict", "report"}:
            raise ConfigurationError("quality_policy must be 'warn', 'strict', or 'report'")
        if self.conflicting_labels not in {"warn", "drop", "error", "keep"}:
            raise ConfigurationError("conflicting_labels must be warn, drop, error, or keep")
        if not 0.0 < float(self.row_missing_threshold) <= 1.0:
            raise ConfigurationError("row_missing_threshold must be in (0, 1]")
        if self.numeric_imputation not in {"median", "mean", "constant", "knn", "iterative"}:
            raise ConfigurationError(
                "numeric_imputation must be median, mean, constant, knn, or iterative"
            )
        if not 0.0 <= float(self.rare_category_min_frequency) < 0.5:
            raise ConfigurationError("rare_category_min_frequency must be in [0, 0.5)")
        if int(self.rare_category_max_categories) < 2:
            raise ConfigurationError("rare_category_max_categories must be >= 2")
        if self.numeric_transform not in {"auto", "none", "log1p", "yeo-johnson"}:
            raise ConfigurationError("numeric_transform must be auto, none, log1p, or yeo-johnson")
        if float(self.skew_threshold) <= 0:
            raise ConfigurationError("skew_threshold must be positive")
        if self.max_model_features != "auto":
            try:
                maximum = int(self.max_model_features)
            except (TypeError, ValueError) as exc:
                raise ConfigurationError("max_model_features must be 'auto' or an integer") from exc
            if maximum < 8:
                raise ConfigurationError("max_model_features must be >= 8")
        if self.winsorize not in {"auto", "none"}:
            try:
                fraction = float(self.winsorize)
            except (TypeError, ValueError) as exc:
                raise ConfigurationError("winsorize must be auto, none, or a fraction") from exc
            if not 0.0 < fraction < 0.25:
                raise ConfigurationError("numeric winsorize fraction must be in (0, 0.25)")
        if self.calibration not in {"auto", "none", "sigmoid", "isotonic"}:
            raise ConfigurationError("calibration must be auto, none, sigmoid, or isotonic")
        if not 0.05 <= float(self.calibration_fraction) <= 0.30:
            raise ConfigurationError("calibration_fraction must be in [0.05, 0.30]")
        if self.conformal not in {True, False, "auto"}:
            raise ConfigurationError("conformal must be True, False, or 'auto'")
        if not 0.01 <= float(self.conformal_alpha) <= 0.5:
            raise ConfigurationError("conformal_alpha must be in [0.01, 0.5]")
        if self.modality_dropout != "auto":
            try:
                dropout = float(self.modality_dropout)
            except (TypeError, ValueError) as exc:
                raise ConfigurationError("modality_dropout must be auto or a fraction") from exc
            if not 0.0 <= dropout <= 0.5:
                raise ConfigurationError("modality_dropout must be in [0, 0.5]")
        if not 0.0 <= float(self.modality_dropout_max_expansion) <= 1.0:
            raise ConfigurationError("modality_dropout_max_expansion must be in [0, 1]")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def resolve_data_science_config(value: str | dict[str, Any] | DataScienceConfig | None) -> DataScienceConfig:
    if value in (None, "auto"):
        return DataScienceConfig()
    if value == "minimal":
        return DataScienceConfig(
            quality_audit=True,
            conflicting_labels="keep",
            add_missing_indicators=False,
            rare_category_min_frequency=0.0,
            numeric_transform="none",
            winsorize="none",
            calibration="none",
            conformal=False,
            ood_detection=False,
            drift_monitoring=False,
            modality_dropout=0.0,
        )
    if isinstance(value, DataScienceConfig):
        return value
    if isinstance(value, dict):
        allowed = {field.name for field in fields(DataScienceConfig)}
        unknown = sorted(set(value) - allowed)
        if unknown:
            raise ConfigurationError(f"unknown data_science settings: {unknown}")
        return DataScienceConfig(**value)
    raise ConfigurationError("data_science must be 'auto', 'minimal', a dictionary, or DataScienceConfig")
