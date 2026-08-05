from pathlib import Path

import pandas as pd
import pytest

from smarttab.config import DEFAULT_TEST_SIZE, FitConfig, load_data
from smarttab.exceptions import ConfigurationError, DataValidationError


def test_defaults_are_validated_and_explicit():
    config = FitConfig(target="y")
    assert config.test_size == pytest.approx(0.2)
    assert DEFAULT_TEST_SIZE == pytest.approx(0.2)
    assert config.optimize is True
    assert config.schema_policy == "strict"
    assert config.task_type == "auto"
    assert config.ensemble_models_limit == 5
    assert config.meta_model == "auto"
    assert config.fusion == "auto"
    assert config.explain == "auto"


@pytest.mark.parametrize("bad_value", [0.0, 1.0, -0.1, 1.5])
def test_invalid_test_size_raises(bad_value):
    with pytest.raises(ConfigurationError):
        FitConfig(target="y", test_size=bad_value)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"clean": "garbage"},
        {"outlier": "garbage"},
        {"categorical": "garbage"},
        {"schema_policy": "garbage"},
        {"n_trials": 0, "optimize": True},
        {"n_estimators": 0},
        {"cv": 1},
        {"time_limit": -1},
        {"time_limit": 1},
        {"ensemble_models_limit": 0},
        {"ensemble_models_limit": 11},
        {"ensemble_min_gain": -0.1},
        {"diversity_correlation_limit": 0},
        {"meta_model": "garbage"},
        {"fusion": "garbage"},
        {"explain": "garbage"},
    ],
)
def test_invalid_configuration_raises(kwargs):
    with pytest.raises(ConfigurationError):
        FitConfig(target="y", **kwargs)


def test_task_and_split_requirements():
    with pytest.raises(ConfigurationError):
        FitConfig(target="y", task_type="ranking")
    with pytest.raises(ConfigurationError):
        FitConfig(target="y", split_strategy="temporal")
    with pytest.raises(ConfigurationError):
        FitConfig(target="y", split_strategy="group")


def test_dataframe_is_copied_on_load():
    frame = pd.DataFrame({"a": [1, 2], "b": ["x", "y"]})
    loaded = load_data(frame)
    assert loaded.equals(frame)
    assert loaded is not frame


@pytest.mark.parametrize("suffix", [".csv", ".tsv", ".json", ".pkl"])
def test_load_data_core_formats(tmp_path, suffix):
    frame = pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
    path = tmp_path / f"data{suffix}"
    if suffix == ".csv":
        frame.to_csv(path, index=False)
    elif suffix == ".tsv":
        frame.to_csv(path, index=False, sep="\t")
    elif suffix == ".json":
        frame.to_json(path)
    else:
        frame.to_pickle(path)
    loaded = load_data(Path(path))
    assert list(loaded.columns) == ["a", "b"]
    assert len(loaded) == 3


@pytest.mark.optional
def test_load_data_parquet_and_feather_when_pyarrow_available(tmp_path):
    pytest.importorskip("pyarrow")
    frame = pd.DataFrame({"a": [1, 2], "b": ["x", "y"]})
    for suffix, writer in ((".parquet", frame.to_parquet), (".feather", frame.to_feather)):
        path = tmp_path / f"data{suffix}"
        writer(path)
        assert load_data(path).equals(frame)


def test_load_data_errors(tmp_path):
    with pytest.raises(DataValidationError):
        load_data(tmp_path / "missing.csv")
    unsupported = tmp_path / "data.txt"
    unsupported.write_text("x")
    with pytest.raises(DataValidationError):
        load_data(unsupported)
    with pytest.raises(DataValidationError):
        load_data(123)


def test_all_documented_threshold_objectives_are_accepted():
    for objective in ("f1", "precision", "recall", "accuracy", "balanced_accuracy", "mcc", "roc_auc"):
        assert FitConfig(target="y", objective=objective).objective == objective


def test_explain_modes_and_gpu_memory_validation():
    assert FitConfig(target="y", explain=True).explain is True
    assert FitConfig(target="y", explain=False).explain is False
    assert FitConfig(target="y", gpu_memory=0.5).gpu_memory == 0.5
    with pytest.raises(ConfigurationError):
        FitConfig(target="y", gpu_memory=0)


def test_zero_trials_is_valid_when_optimization_is_disabled():
    config = FitConfig(target="target", optimize=False, n_trials=0)
    assert config.n_trials == 0


def test_explicit_feature_space_config_is_accepted_and_validated():
    from smarttab.multimodal.config import FeatureSpaceConfig, resolve_feature_space_config

    explicit = FeatureSpaceConfig(
        total_features=144,
        modality_limits={"text": 96},
        column_limits={"review": 72},
        speed_accuracy=0.7,
        backend="classical",
        batch_size=12,
        workers=3,
    )
    resolved = resolve_feature_space_config(
        explicit,
        speed_accuracy=0.1,
        backend="auto",
        allow_model_download=False,
        error_policy="warn",
        batch_size="auto",
        workers="auto",
        cache=False,
        modality_params=None,
        random_state=42,
    )
    assert resolved is explicit
    assert resolved.total_features == 144
    assert resolved.workers == 3


@pytest.mark.parametrize(
    "overrides, message",
    [
        ({"speed_accuracy": -0.1}, "speed_accuracy"),
        ({"backend": "unknown"}, "multimodal_backend"),
        ({"error_policy": "silent"}, "media_error_policy"),
        ({"allow_model_download": "yes"}, "allow_model_download"),
        ({"cache": object()}, "feature_cache"),
        ({"modality_params": []}, "modality_params"),
        ({"batch_size": 0}, "batch_size"),
        ({"workers": 0}, "feature_workers"),
    ],
)
def test_feature_space_resolution_rejects_invalid_controls(overrides, message):
    from smarttab.multimodal.config import resolve_feature_space_config

    arguments = dict(
        feature_budget="auto",
        speed_accuracy=0.5,
        backend="auto",
        allow_model_download=False,
        error_policy="warn",
        batch_size="auto",
        workers="auto",
        cache=False,
        modality_params=None,
        random_state=42,
    )
    arguments.update(overrides)
    with pytest.raises(ConfigurationError, match=message):
        resolve_feature_space_config(**arguments)


def test_feature_budget_integer_dictionary_and_automatic_resolution():
    from smarttab.multimodal.config import resolve_feature_space_config

    common = dict(
        speed_accuracy=0.25,
        backend="classical",
        allow_model_download=False,
        error_policy="warn",
        batch_size="auto",
        workers="auto",
        cache=False,
        modality_params=None,
        random_state=7,
    )
    automatic = resolve_feature_space_config("auto", **common)
    explicit = resolve_feature_space_config(96, **common)
    mapping = resolve_feature_space_config(
        {"total": 120, "text": 80, "review": 64},
        **common,
    )
    assert automatic.total_features == 448
    assert automatic.batch_size >= 4
    assert automatic.workers >= 1
    assert explicit.total_features == 96
    assert mapping.modality_limits == {"text": 80}
    assert mapping.column_limits == {"review": 64}
    assert mapping.limit_for("text", ["text", "image"]) == 80
    assert mapping.limit_for("image", ["text", "image"]) >= 16


@pytest.mark.parametrize(
    "feature_budget",
    [31, 16385, {"total": 64, "text": 4}, {"total": 64, "image": 80}],
)
def test_feature_budget_bounds_are_enforced(feature_budget):
    from smarttab.multimodal.config import resolve_feature_space_config

    with pytest.raises(ConfigurationError):
        resolve_feature_space_config(
            feature_budget,
            speed_accuracy=0.5,
            backend="auto",
            allow_model_download=False,
            error_policy="warn",
            batch_size="auto",
            workers="auto",
            cache=False,
            modality_params=None,
            random_state=42,
        )


def test_invalid_prebuilt_feature_space_is_rejected():
    from smarttab.multimodal.config import FeatureSpaceConfig, resolve_feature_space_config

    invalid = FeatureSpaceConfig(total_features=20, batch_size=0, workers=0)
    with pytest.raises(ConfigurationError):
        resolve_feature_space_config(
            invalid,
            speed_accuracy=0.5,
            backend="auto",
            allow_model_download=False,
            error_policy="warn",
            batch_size="auto",
            workers="auto",
            cache=False,
            modality_params=None,
            random_state=42,
        )
