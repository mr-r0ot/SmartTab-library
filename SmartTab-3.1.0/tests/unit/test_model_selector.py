import pytest

from smarttab.analysis.dataset_analyzer import DatasetProfile, TaskType
from smarttab.exceptions import UnsupportedModelError
from smarttab.selection.model_selector import select_model


def _profile(n_samples, cardinality=None):
    return DatasetProfile(
        task_type=TaskType.BINARY,
        target_column="target",
        n_samples=n_samples,
        n_features=3,
        feature_columns=["a", "b", "c"],
        cardinality=cardinality or {},
    )


def test_auto_picks_lightgbm_for_small_numeric_dataset():
    model, notes = select_model(_profile(1000), model="auto")
    assert model == "lightgbm"


def test_auto_picks_lightgbm_for_large_dataset():
    model, notes = select_model(_profile(200_000), model="auto")
    assert model == "lightgbm"


def test_auto_picks_catboost_for_large_dataset_with_high_cardinality():
    model, notes = select_model(_profile(200_000, cardinality={"col": 500}), model="auto")
    assert model == "catboost"


def test_explicit_model_is_honored():
    model, notes = select_model(_profile(200_000), model="catboost")
    assert model == "catboost"


def test_unsupported_model_raises():
    with pytest.raises(UnsupportedModelError):
        select_model(_profile(1000), model="xgboost")


