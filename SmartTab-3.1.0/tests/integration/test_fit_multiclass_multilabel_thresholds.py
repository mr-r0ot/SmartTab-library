import numpy as np
import pandas as pd
import pytest

import smarttab
from smarttab.analysis.dataset_analyzer import TaskType


def _multiclass(n=260, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 4))
    code = np.argmax(
        np.column_stack([X[:, 0], X[:, 1] + 0.2, -X[:, 0] - X[:, 1]]), axis=1
    )
    frame = pd.DataFrame(X, columns=["a", "b", "c", "d"])
    frame["label"] = np.array(["red", "green", "blue"])[code]
    return frame


def _multilabel(n=240, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 4))
    frame = pd.DataFrame(X, columns=["a", "b", "c", "d"])
    frame["label_a"] = (X[:, 0] + X[:, 1] > 0).astype(int)
    frame["label_b"] = (X[:, 2] - X[:, 3] > 0.2).astype(int)
    return frame


def test_multiclass_auto_string_target_and_argmax_contract(tmp_path):
    frame = _multiclass()
    model = smarttab.fit(
        frame,
        target="label",
        model="lightgbm",
        optimize=False,
        n_estimators=40,
        report=False,
        explain=False,
        static_charts=False,
        verbose=0,
    )
    assert model.task_type is TaskType.MULTICLASS
    X = frame.drop(columns="label").head(12)
    proba = model.predict_proba(X)
    predictions = model.predict(X)
    expected = model.target_encoder.inverse_transform(proba.argmax(axis=1))
    np.testing.assert_array_equal(predictions, expected)
    report = model.report(tmp_path / "multiclass")
    assert "f1_macro" in report["metrics"]
    assert "multi_threshold_ensemble" not in report


def test_numeric_multiclass_requires_explicit_override():
    frame = _multiclass()
    mapping = {"red": 0, "green": 1, "blue": 2}
    frame["label"] = frame["label"].map(mapping)
    auto = smarttab.fit(
        frame,
        target="label",
        model="lightgbm",
        optimize=False,
        n_estimators=20,
        report=False,
        explain=False,
        threshold_optimization=False,
        verbose=0,
    )
    assert auto.task_type is TaskType.REGRESSION
    explicit = smarttab.fit(
        frame,
        target="label",
        task_type="multiclass",
        model="lightgbm",
        optimize=False,
        n_estimators=20,
        report=False,
        explain=False,
        threshold_optimization=False,
        verbose=0,
    )
    assert explicit.task_type is TaskType.MULTICLASS


def test_multilabel_uses_one_threshold_per_label_and_roundtrips(tmp_path):
    frame = _multilabel()
    targets = ["label_a", "label_b"]
    model = smarttab.fit(
        frame,
        target=targets,
        task_type="multilabel",
        model="lightgbm",
        optimize=False,
        n_estimators=35,
        report=False,
        explain=False,
        verbose=0,
    )
    assert model.task_type is TaskType.MULTILABEL
    assert model.per_label_thresholds is not None
    assert len(model.per_label_thresholds) == 2
    X = frame.drop(columns=targets).head(10)
    predictions = model.predict(X)
    probabilities = model.predict_proba(X)
    assert predictions.shape == (10, 2)
    assert probabilities.shape == (10, 2)
    bundle = model.save(tmp_path / "multilabel.smarttab")
    loaded = smarttab.load(bundle, trusted=True)
    np.testing.assert_array_equal(predictions, loaded.predict(X))
    assert loaded.per_label_thresholds == pytest.approx(model.per_label_thresholds)
