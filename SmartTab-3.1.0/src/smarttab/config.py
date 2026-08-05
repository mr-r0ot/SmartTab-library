"""Configuration and raw-data loading for :func:`smarttab.fit`."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from smarttab.exceptions import ConfigurationError, DataValidationError
from smarttab.optimization.threshold import DEFAULT_OBJECTIVE, VALID_OBJECTIVES
from smarttab.multimodal.config import FeatureSpaceConfig, resolve_feature_space_config
from smarttab.datascience.config import DataScienceConfig, resolve_data_science_config

DEFAULT_TEST_SIZE = 0.2

VALID_TASK_TYPES = (
    "auto",
    "binary",
    "multiclass",
    "regression",
    "multilabel",
    "multioutput_regression",
    "ranking",
)
VALID_ENSEMBLE_MODES = ("none", "voting", "stacking", "auto")
VALID_MODEL_MODES = ("auto", "catboost", "lightgbm")
VALID_CLEAN_MODES = ("auto", "minimal", "none")
VALID_SPLIT_STRATEGIES = ("auto", "random", "group", "stratified_group", "temporal")
VALID_SCHEMA_POLICIES = ("strict", "coerce")
VALID_LEAKAGE_POLICIES = ("drop", "error", "warn", "ignore")
VALID_DUPLICATE_POLICIES = ("drop", "keep", "error")
VALID_TARGET_MISSING_POLICIES = ("error", "drop")
VALID_XGBOOST_POLICIES = ("auto", "never", "always")
VALID_FUSION_STRATEGIES = ("auto", "early", "late", "hybrid")


@dataclass(slots=True)
class FitConfig:
    target: str | list[str]
    group_id: str | None = None
    task_type: str = "auto"
    split_strategy: str = "auto"
    time_column: str | None = None

    model: str = "auto"
    ensemble: str = "none"
    ensemble_models_limit: int = 5
    ensemble_min_gain: float = 0.001
    diversity_correlation_limit: float = 0.98
    meta_model: str = "auto"
    xgboost_policy: str = "auto"
    fusion: str = "auto"

    clean: str = "auto"
    missing: str = "auto"
    categorical: str = "auto"
    scaling: str = "auto"
    outlier: str = "auto"
    feature_selection: str = "auto"
    leakage_policy: str = "drop"
    duplicate_policy: str = "drop"
    target_missing: str = "error"
    schema_policy: str = "strict"

    modalities: dict[str, str] | str | None = "auto"
    feature_budget: str | int | dict[str, int] | FeatureSpaceConfig = "auto"
    speed_accuracy: float = 0.5
    multimodal_backend: str = "auto"
    allow_model_download: bool = False
    media_error_policy: str = "warn"
    feature_cache: bool | str = False
    batch_size: str | int = "auto"
    feature_workers: str | int = "auto"
    modality_params: dict[str, dict[str, Any]] | None = None
    supervised_adaptation: str = "auto"
    adapter_features: str | int = "auto"

    data_science: str | dict[str, Any] | DataScienceConfig = "auto"
    data_science_config: DataScienceConfig = field(init=False, repr=False)

    test_size: float = DEFAULT_TEST_SIZE
    validation: str = "auto"
    cv: str | int = "auto"
    optimize: bool = True
    optimizer: str = "auto"
    n_trials: str | int = "auto"
    n_estimators: str | int = "auto"
    timeout: float | None = None
    time_limit: float = 0

    threshold_optimization: bool = True
    objective: str = DEFAULT_OBJECTIVE

    device: str = "auto"
    cpu_threads: str | int = "auto"
    gpu_memory: str | float = "auto"
    ram_limit: str | float = "auto"
    metrics: str = "auto"
    params: dict[str, Any] | None = None

    report: bool = True
    explain: bool | str = "auto"
    static_charts: str | bool = "auto"
    random_state: int = 42
    verbose: int = 1

    def __post_init__(self) -> None:
        self._one_of("task_type", self.task_type, VALID_TASK_TYPES)
        self._one_of("model", self.model, VALID_MODEL_MODES)
        self._one_of("ensemble", self.ensemble, VALID_ENSEMBLE_MODES)
        self._one_of("xgboost_policy", self.xgboost_policy, VALID_XGBOOST_POLICIES)
        self._one_of("fusion", self.fusion, VALID_FUSION_STRATEGIES)
        if not isinstance(self.ensemble_models_limit, int) or not 1 <= self.ensemble_models_limit <= 10:
            raise ConfigurationError("ensemble_models_limit must be an integer between 1 and 10")
        if float(self.ensemble_min_gain) < 0:
            raise ConfigurationError("ensemble_min_gain must be >= 0")
        if not 0.0 < float(self.diversity_correlation_limit) <= 1.0:
            raise ConfigurationError("diversity_correlation_limit must be in (0, 1]")
        if self.meta_model not in ("auto", "catboost", "lightgbm", "linear"):
            raise ConfigurationError("meta_model must be 'auto', 'catboost', 'lightgbm', or 'linear'")
        self._one_of("clean", self.clean, VALID_CLEAN_MODES)
        self._one_of("split_strategy", self.split_strategy, VALID_SPLIT_STRATEGIES)
        self._one_of("schema_policy", self.schema_policy, VALID_SCHEMA_POLICIES)
        self._one_of("leakage_policy", self.leakage_policy, VALID_LEAKAGE_POLICIES)
        self._one_of("duplicate_policy", self.duplicate_policy, VALID_DUPLICATE_POLICIES)
        self._one_of("target_missing", self.target_missing, VALID_TARGET_MISSING_POLICIES)
        resolve_feature_space_config(
            self.feature_budget,
            speed_accuracy=self.speed_accuracy,
            backend=self.multimodal_backend,
            allow_model_download=self.allow_model_download,
            error_policy=self.media_error_policy,
            batch_size=self.batch_size,
            workers=self.feature_workers,
            cache=self.feature_cache,
            modality_params=self.modality_params,
            random_state=self.random_state,
            device=self.device,
            supervised_adaptation=self.supervised_adaptation,
            adapter_features=self.adapter_features,
        )
        self.data_science_config = resolve_data_science_config(self.data_science)

        if self.missing not in ("auto", "median", "mean", "constant", "knn", "iterative"):
            raise ConfigurationError("missing must be auto, median, mean, constant, knn, or iterative")
        if self.categorical not in ("auto", "native", "ordinal"):
            raise ConfigurationError("categorical must be 'auto', 'native', or 'ordinal'")
        if self.scaling not in ("auto", "none", "standard", "minmax", "robust"):
            raise ConfigurationError("scaling must be 'auto', 'none', 'standard', 'minmax', or 'robust'")
        if self.outlier not in ("auto", "keep", "remove", "clip"):
            raise ConfigurationError("outlier must be auto, keep, remove, or clip")
        if self.feature_selection not in ("auto", "none"):
            raise ConfigurationError("feature_selection must be 'auto' or 'none'")
        if self.validation not in ("auto", "kfold", "holdout"):
            raise ConfigurationError("validation must be 'auto', 'kfold', or 'holdout'")
        if self.optimizer not in ("auto", "tpe", "random"):
            raise ConfigurationError("optimizer must be 'auto', 'tpe', or 'random'")

        if not (0.0 < float(self.test_size) < 1.0):
            raise ConfigurationError(f"test_size must be between 0 and 1, got {self.test_size!r}")
        if float(self.time_limit) < 0:
            raise ConfigurationError(f"time_limit must be >= 0, got {self.time_limit!r}")
        if 0 < float(self.time_limit) < 2.0:
            raise ConfigurationError("time_limit must be 0 (unlimited) or at least 2 seconds")
        if self.timeout is not None and float(self.timeout) <= 0:
            raise ConfigurationError(f"timeout must be > 0 or None, got {self.timeout!r}")
        if self.objective not in VALID_OBJECTIVES:
            raise ConfigurationError(f"objective must be one of {VALID_OBJECTIVES}, got {self.objective!r}")
        if not isinstance(self.optimize, bool):
            raise ConfigurationError("optimize must be True or False")
        if not isinstance(self.threshold_optimization, bool):
            raise ConfigurationError("threshold_optimization must be True or False")
        if self.explain not in (True, False, "auto"):
            raise ConfigurationError("explain must be True, False, or 'auto'")
        if not isinstance(self.random_state, int):
            raise ConfigurationError("random_state must be an integer")
        if not isinstance(self.verbose, int) or self.verbose < 0:
            raise ConfigurationError("verbose must be a non-negative integer")
        if self.n_trials != "auto":
            try:
                n_trials = int(self.n_trials)
            except (TypeError, ValueError) as exc:
                raise ConfigurationError("n_trials must be 'auto' or a positive integer") from exc
            minimum_trials = 1 if self.optimize else 0
            if n_trials < minimum_trials:
                requirement = ">= 1 when optimize=True" if self.optimize else ">= 0"
                raise ConfigurationError(f"n_trials must be {requirement}")
        if self.n_estimators != "auto":
            try:
                n_estimators = int(self.n_estimators)
            except (TypeError, ValueError) as exc:
                raise ConfigurationError("n_estimators must be 'auto' or a positive integer") from exc
            if n_estimators < 1:
                raise ConfigurationError("n_estimators must be >= 1")
        if self.cv != "auto":
            try:
                cv = int(self.cv)
            except (TypeError, ValueError) as exc:
                raise ConfigurationError("cv must be 'auto' or an integer >= 2") from exc
            if cv < 2:
                raise ConfigurationError("cv must be >= 2")
        if self.static_charts not in (True, False, "auto"):
            raise ConfigurationError("static_charts must be True, False, or 'auto'")
        if self.gpu_memory != "auto":
            try:
                gpu_memory = float(self.gpu_memory)
            except (TypeError, ValueError) as exc:
                raise ConfigurationError(
                    "gpu_memory must be 'auto', a fraction in (0, 1], or an absolute MB value"
                ) from exc
            if gpu_memory <= 0:
                raise ConfigurationError("gpu_memory must be positive")
        if self.params is not None and not isinstance(self.params, dict):
            raise ConfigurationError("params must be a dictionary or None")

        if self.task_type == "ranking" and self.group_id is None:
            raise ConfigurationError("task_type='ranking' requires group_id=...")
        if self.split_strategy == "temporal" and self.time_column is None:
            raise ConfigurationError("split_strategy='temporal' requires time_column=...")
        if self.split_strategy in ("group", "stratified_group") and self.group_id is None:
            raise ConfigurationError(f"split_strategy={self.split_strategy!r} requires group_id=...")
        if self.ensemble != "none" and self.params is not None:
            raise ConfigurationError("params applies to a single model; use ensemble='none' or leave params=None")

    @staticmethod
    def _one_of(name: str, value: str, allowed: tuple[str, ...]) -> None:
        if value not in allowed:
            raise ConfigurationError(f"{name} must be one of {allowed}, got {value!r}")


SUPPORTED_DATA_EXTENSIONS = (
    ".csv",
    ".tsv",
    ".xlsx",
    ".xls",
    ".parquet",
    ".json",
    ".feather",
    ".pkl",
    ".pickle",
)


def load_data(data) -> pd.DataFrame:
    """Load a DataFrame or a supported local tabular file."""
    if isinstance(data, pd.DataFrame):
        return data.copy()

    if isinstance(data, (str, Path)):
        path = Path(data).expanduser()
        if not path.exists():
            raise DataValidationError(f"data file not found: {path}")
        if not path.is_file():
            raise DataValidationError(f"data path is not a file: {path}")
        suffix = path.suffix.lower()
        try:
            if suffix == ".csv":
                return pd.read_csv(path)
            if suffix == ".tsv":
                return pd.read_csv(path, sep="\t")
            if suffix in (".xlsx", ".xls"):
                return pd.read_excel(path)
            if suffix == ".parquet":
                return pd.read_parquet(path)
            if suffix == ".json":
                return pd.read_json(path)
            if suffix == ".feather":
                return pd.read_feather(path)
            if suffix in (".pkl", ".pickle"):
                return pd.read_pickle(path)
        except ImportError as exc:
            extra = "excel" if suffix in (".xlsx", ".xls") else "parquet"
            raise DataValidationError(
                f"Reading {suffix} requires an optional dependency; install smarttab[{extra}]"
            ) from exc
        except Exception as exc:
            raise DataValidationError(f"failed to read data file {path}: {exc}") from exc
        raise DataValidationError(
            f"unsupported file extension {suffix!r}; expected one of {SUPPORTED_DATA_EXTENSIONS}"
        )

    raise DataValidationError(
        "data must be a pandas DataFrame or a path to a supported tabular file; "
        f"got {type(data).__name__}"
    )
