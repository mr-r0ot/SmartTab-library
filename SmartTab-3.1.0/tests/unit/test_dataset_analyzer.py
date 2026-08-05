import numpy as np
import pandas as pd
import pytest

from smarttab.analysis.dataset_analyzer import TaskType, analyze_dataset
from smarttab.exceptions import DataValidationError


def _base_df(n=200, seed=0):
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "id": np.arange(n),
            "num_a": rng.normal(size=n),
            "num_b": rng.normal(size=n),
            "cat_a": rng.choice(["x", "y", "z"], size=n),
            "const_col": ["same"] * n,
        }
    )


def test_detects_regression_target():
    df = _base_df()
    df["target"] = np.random.default_rng(1).normal(size=len(df))
    profile = analyze_dataset(df, "target")
    assert profile.task_type is TaskType.REGRESSION


def test_detects_binary_classification_target():
    df = _base_df()
    df["target"] = np.random.default_rng(1).integers(0, 2, size=len(df))
    profile = analyze_dataset(df, "target")
    assert profile.task_type is TaskType.BINARY


def test_detects_multiclass_classification_target():
    df = _base_df()
    df["target"] = np.random.default_rng(1).integers(0, 4, size=len(df))
    profile = analyze_dataset(df, "target", task_type="multiclass")
    assert profile.task_type is TaskType.MULTICLASS


def test_constant_column_detected():
    df = _base_df()
    df["target"] = np.random.default_rng(1).integers(0, 2, size=len(df))
    profile = analyze_dataset(df, "target")
    assert "const_col" in profile.constant_columns


def test_id_like_column_detected():
    df = _base_df()
    df["target"] = np.random.default_rng(1).integers(0, 2, size=len(df))
    profile = analyze_dataset(df, "target")
    assert "id" in profile.id_like_columns


def test_duplicate_rows_counted():
    df = _base_df(n=50)
    df["target"] = np.random.default_rng(1).integers(0, 2, size=len(df))
    df = pd.concat([df, df.iloc[:5]], ignore_index=True)
    profile = analyze_dataset(df, "target")
    assert profile.duplicate_row_count == 5


def test_missing_values_reported():
    df = _base_df()
    df.loc[0:9, "num_a"] = np.nan
    df["target"] = np.random.default_rng(1).integers(0, 2, size=len(df))
    profile = analyze_dataset(df, "target")
    assert "num_a" in profile.missing_report
    assert profile.missing_report["num_a"] == pytest.approx(10 / len(df))


def test_high_correlation_pair_detected():
    n = 200
    df = _base_df(n=n)
    df["num_a_copy"] = df["num_a"] * 2 + 0.0001
    df["target"] = np.random.default_rng(1).integers(0, 2, size=n)
    profile = analyze_dataset(df, "target")
    pairs = {frozenset((a, b)) for a, b, _ in profile.high_correlation_pairs}
    assert frozenset(("num_a", "num_a_copy")) in pairs


def test_class_imbalance_detected():
    n = 500
    df = _base_df(n=n)
    target = np.zeros(n, dtype=int)
    target[:20] = 1
    df["target"] = target
    profile = analyze_dataset(df, "target")
    assert profile.is_imbalanced is True
    assert profile.class_imbalance_ratio == pytest.approx(20 / 480, abs=1e-6)


def test_leakage_column_detected_regression():
    n = 200
    df = _base_df(n=n)
    y = np.random.default_rng(2).normal(size=n)
    df["target"] = y
    df["leaky"] = y * 1.0000001
    profile = analyze_dataset(df, "target")
    assert "leaky" in profile.potential_leakage_columns


def test_missing_target_raises():
    df = _base_df()
    with pytest.raises(DataValidationError):
        analyze_dataset(df, "does_not_exist")


def test_empty_dataframe_raises():
    with pytest.raises(DataValidationError):
        analyze_dataset(pd.DataFrame(), "target")


def test_multilabel_detected_for_multiple_binary_targets():
    df = _base_df()
    df["label1"] = np.random.default_rng(1).integers(0, 2, size=len(df))
    df["label2"] = np.random.default_rng(2).integers(0, 2, size=len(df))
    profile = analyze_dataset(df, target=["label1", "label2"])
    assert profile.task_type is TaskType.MULTILABEL
    assert profile.target_columns == ["label1", "label2"]
    assert "label1" not in profile.feature_columns
    assert "label2" not in profile.feature_columns


def test_multioutput_regression_detected_for_multiple_continuous_targets():
    df = _base_df()
    df["t1"] = np.random.default_rng(1).normal(size=len(df))
    df["t2"] = np.random.default_rng(2).normal(size=len(df))
    profile = analyze_dataset(df, target=["t1", "t2"])
    assert profile.task_type is TaskType.MULTIOUTPUT_REGRESSION
    assert profile.target_columns == ["t1", "t2"]


def test_mixed_multi_target_types_raise():
    # a 3-category string column is neither binary-like nor numeric, so pairing it with a
    # continuous numeric column can't resolve to either multilabel or multi-output regression.
    df = _base_df()
    df["cat_target"] = np.random.default_rng(1).choice(["a", "b", "c"], size=len(df))
    df["t2"] = np.random.default_rng(2).normal(size=len(df))
    with pytest.raises(DataValidationError):
        analyze_dataset(df, target=["cat_target", "t2"])


def test_ranking_detected_when_group_id_given():
    n = 300
    df = _base_df(n=n)
    rng = np.random.default_rng(3)
    df["relevance"] = rng.integers(0, 4, size=n)
    df["query_id"] = rng.integers(0, 30, size=n)
    profile = analyze_dataset(df, target="relevance", group_id="query_id")
    assert profile.task_type is TaskType.RANKING
    assert profile.group_id_column == "query_id"
    assert "query_id" not in profile.feature_columns
    assert profile.n_groups == 30


def test_ranking_with_multiple_targets_raises():
    n = 100
    df = _base_df(n=n)
    rng = np.random.default_rng(3)
    df["relevance"] = rng.integers(0, 4, size=n)
    df["relevance2"] = rng.integers(0, 4, size=n)
    df["query_id"] = rng.integers(0, 10, size=n)
    with pytest.raises(DataValidationError):
        analyze_dataset(df, target=["relevance", "relevance2"], group_id="query_id")


def test_missing_group_id_column_raises():
    df = _base_df()
    df["relevance"] = np.random.default_rng(3).integers(0, 4, size=len(df))
    with pytest.raises(DataValidationError):
        analyze_dataset(df, target="relevance", group_id="does_not_exist")


def test_text_target_copy_and_encoded_copy_are_detected():
    n = 180
    rng = np.random.default_rng(44)
    frame = _base_df(n=n)
    target = pd.Series(rng.choice(["approved", "rejected"], size=n), name="target")
    frame["target"] = target
    frame["target_copy"] = target.str.upper().str.pad(12)
    frame["outcome_code"] = target.map({"approved": "A", "rejected": "R"})
    profile = analyze_dataset(frame, "target")
    assert {"target_copy", "outcome_code"} <= set(profile.potential_leakage_columns)
    assert any("exact copy" in reason for reason in profile.leakage_reasons["target_copy"])
    assert any("deterministic mapping" in reason for reason in profile.leakage_reasons["outcome_code"])


def test_unique_continuous_feature_is_not_misclassified_as_deterministic_leakage():
    n = 200
    rng = np.random.default_rng(45)
    frame = pd.DataFrame(
        {
            "unique_feature": np.linspace(-3.0, 4.0, n),
            "noise": rng.normal(size=n),
            "target": rng.normal(size=n),
        }
    )
    profile = analyze_dataset(frame, "target")
    assert "unique_feature" not in profile.potential_leakage_columns


def test_group_id_can_be_used_for_grouped_classification_when_task_is_explicit():
    frame = _base_df(n=120)
    frame["target"] = np.random.default_rng(46).integers(0, 2, size=len(frame))
    frame["patient_id"] = np.repeat(np.arange(30), 4)
    profile = analyze_dataset(frame, "target", group_id="patient_id", task_type="binary")
    assert profile.task_type is TaskType.BINARY
    assert profile.group_id_column == "patient_id"
