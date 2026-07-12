"""fit() parameter container + raw-data loading (CSV/Excel/Parquet/DataFrame)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from smarttab.exceptions import ConfigurationError, DataValidationError
from smarttab.optimization.threshold import DEFAULT_OBJECTIVE, VALID_OBJECTIVES

DEFAULT_TEST_SIZE = 0.3
DEFAULT_THRESHOLD_MODELS = 4

VALID_ENSEMBLE_MODES = ("none", "voting", "stacking", "auto")


@dataclass
class FitConfig:
    target: str | list[str]
    group_id: str | None = None
    model: str = "auto"
    ensemble: str = "none"
    clean: str = "auto"
    missing: str = "auto"
    categorical: str = "auto"
    scaling: str = "auto"
    outlier: str = "auto"
    feature_selection: str = "auto"
    test_size: float = DEFAULT_TEST_SIZE
    validation: str = "auto"
    cv: str | int = "auto"
    optimize: bool = True
    optimizer: str = "auto"
    n_trials: str | int = "auto"
    timeout: float | None = None
    time_limit: float = 0
    threshold_optimization: bool = True
    objective: str = DEFAULT_OBJECTIVE
    multi_threshold_ensemble: bool = False
    threshold_models: int = DEFAULT_THRESHOLD_MODELS
    device: str = "auto"
    cpu_threads: str | int = "auto"
    gpu_memory: str = "auto"
    ram_limit: str | float = "auto"
    metrics: str = "auto"
    params: dict | None = None
    report: bool = True
    explain: bool = True
    random_state: int = 42
    verbose: int = 1

    def __post_init__(self) -> None:
        if not (0.0 < self.test_size < 1.0):
            raise ConfigurationError(f"test_size must be between 0 and 1 (exclusive), got {self.test_size!r}")
        if self.ensemble not in VALID_ENSEMBLE_MODES:
            raise ConfigurationError(f"ensemble must be one of {VALID_ENSEMBLE_MODES}, got {self.ensemble!r}")
        if self.time_limit < 0:
            raise ConfigurationError(f"time_limit must be >= 0 (0 = unlimited), got {self.time_limit!r}")
        if self.objective not in VALID_OBJECTIVES:
            raise ConfigurationError(f"objective must be one of {VALID_OBJECTIVES}, got {self.objective!r}")
        if self.threshold_models < 2:
            raise ConfigurationError(f"threshold_models must be >= 2, got {self.threshold_models!r}")


SUPPORTED_DATA_EXTENSIONS = (".csv", ".tsv", ".xlsx", ".xls", ".parquet", ".json", ".feather", ".pkl", ".pickle")


def load_data(data) -> pd.DataFrame:
    """Accept a DataFrame or a path to a CSV/TSV/Excel/Parquet/JSON/Feather/Pickle file."""
    if isinstance(data, pd.DataFrame):
        return data

    if isinstance(data, (str, Path)):
        path = Path(data)
        if not path.exists():
            raise DataValidationError(f"data file not found: {path}")
        suffix = path.suffix.lower()
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
        raise DataValidationError(
            f"Unsupported file extension {suffix!r}; expected one of {SUPPORTED_DATA_EXTENSIONS}"
        )

    raise DataValidationError(
        f"data must be a pandas DataFrame or a path to one of {SUPPORTED_DATA_EXTENSIONS}, got {type(data).__name__}"
    )
