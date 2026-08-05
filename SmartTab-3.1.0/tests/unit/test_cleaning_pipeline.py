import numpy as np
import pandas as pd
import pytest

from smarttab.analysis.dataset_analyzer import analyze_dataset
from smarttab.cleaning.pipeline import SmartCleaningPipeline
from smarttab.exceptions import DataValidationError


def _make_df(n=300, seed=0):
    rng = np.random.default_rng(seed)
    frame = pd.DataFrame(
        {
            "id": np.arange(n),
            "num_a": rng.normal(size=n),
            "num_b": rng.normal(size=n),
            "cat_a": rng.choice(["red", "green", "blue"], size=n).astype(object),
            "const_col": ["same"] * n,
            "signup_date": pd.date_range("2020-01-01", periods=n, freq="D").astype(str),
        }
    )
    frame.loc[rng.choice(n, size=20, replace=False), "num_a"] = np.nan
    frame.loc[rng.choice(n, size=15, replace=False), "cat_a"] = np.nan
    frame["target"] = rng.integers(0, 2, size=n)
    return frame


def _fit(**kwargs):
    frame = _make_df()
    profile = analyze_dataset(frame, "target")
    pipeline = SmartCleaningPipeline(**kwargs)
    transformed = pipeline.fit_transform(frame, frame["target"], profile)
    return frame, pipeline, transformed


def test_auto_drops_constant_and_id_columns():
    _, _, transformed = _fit()
    assert "const_col" not in transformed
    assert "id" not in transformed


def test_clean_modes_have_distinct_behavior():
    _, _, auto = _fit(clean="auto", feature_selection="none")
    _, _, minimal = _fit(clean="minimal", feature_selection="none")
    _, _, none = _fit(clean="none", feature_selection="none")
    assert "id" not in auto
    assert "id" in minimal
    assert "const_col" not in minimal
    assert "const_col" in none


def test_native_and_ordinal_categorical_modes():
    _, native_pipeline, native = _fit(categorical="native")
    assert isinstance(native["cat_a"].dtype, pd.CategoricalDtype)
    _, ordinal_pipeline, ordinal = _fit(categorical="ordinal")
    assert pd.api.types.is_integer_dtype(ordinal["cat_a"])
    assert native_pipeline.final_categorical_columns == ordinal_pipeline.final_categorical_columns


def test_missing_values_datetime_and_schema_replay():
    frame, pipeline, transformed = _fit()
    assert transformed.isna().sum().sum() == 0
    assert "signup_date" not in transformed
    assert "signup_date_year" in transformed
    new = _make_df(50, 99).drop(columns="target")
    replay = pipeline.transform(new)
    assert list(replay.columns) == list(transformed.columns)
    assert replay.isna().sum().sum() == 0


def test_unseen_category_uses_fitted_unknown_category():
    frame, pipeline, _ = _fit(categorical="native")
    new = frame.drop(columns="target").head(3).copy()
    new["cat_a"] = "unseen"
    transformed = pipeline.transform(new)
    assert "__unknown__" in transformed["cat_a"].cat.categories
    assert (transformed["cat_a"].astype(str) == "__unknown__").all()


def test_strict_schema_rejects_missing_and_extra_columns():
    frame, pipeline, _ = _fit(schema_policy="strict")
    features = frame.drop(columns="target")
    with pytest.raises(DataValidationError, match="missing required columns"):
        pipeline.transform(features.drop(columns="num_a"))
    with pytest.raises(DataValidationError, match="unexpected columns"):
        pipeline.transform(features.assign(extra=1))


def test_coerce_schema_uses_fitted_imputer_not_zero_fill():
    frame, pipeline, _ = _fit(schema_policy="coerce")
    expected_median = pipeline.numeric_imputer_.medians_["num_a"]
    transformed = pipeline.transform(frame.drop(columns=["target", "num_a"]).head(2))
    # no scaling under auto, so the inserted missing feature must equal the fitted median
    assert transformed["num_a"].tolist() == pytest.approx([expected_median, expected_median])


def test_near_duplicate_numeric_column_is_removed_only_in_auto():
    frame = _make_df()
    frame["num_a_dup"] = frame["num_a"] * 1.0000001
    profile = analyze_dataset(frame, "target")
    auto = SmartCleaningPipeline(feature_selection="auto").fit_transform(frame, frame["target"], profile)
    disabled = SmartCleaningPipeline(feature_selection="none").fit_transform(frame, frame["target"], profile)
    assert not ({"num_a", "num_a_dup"} <= set(auto.columns))
    assert {"num_a", "num_a_dup"} <= set(disabled.columns)
