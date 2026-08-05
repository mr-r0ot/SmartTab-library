"""End-to-end validation of supported prediction inputs and PredictionArray serialization."""

import json

import numpy as np
import pandas as pd
import pytest

import smarttab


def _make_binary_df(n=800, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 3))
    df = pd.DataFrame(X, columns=["a", "b", "c"])
    df["cat"] = rng.choice(["x", "y"], n)
    df["label"] = (df["a"] + (df["cat"] == "x").astype(int) > 0).astype(int)
    return df


@pytest.fixture(scope="module")
def binary_model():
    df = _make_binary_df()
    return smarttab.fit(df, target="label", optimize=False, verbose=0, report=False)


def test_predict_accepts_dataframe(binary_model):
    df = _make_binary_df(n=10, seed=1)
    X = df.drop(columns=["label"])
    result = binary_model.predict(X)
    assert isinstance(result, np.ndarray)
    assert len(result) == 10


def test_predict_accepts_single_dict(binary_model):
    sample = {"a": 0.5, "b": -0.2, "c": 0.1, "cat": "x"}
    result = binary_model.predict(sample)
    assert result.prediction in (0, 1)
    assert np.isscalar(result.prediction) or np.ndim(result.prediction) == 0


def test_predict_accepts_list_of_dicts(binary_model):
    samples = [
        {"a": 0.5, "b": -0.2, "c": 0.1, "cat": "x"},
        {"a": -0.9, "b": 0.1, "c": 0.0, "cat": "y"},
    ]
    result = binary_model.predict(samples)
    assert len(result) == 2


def test_predict_accepts_csv_path(binary_model, tmp_path):
    df = _make_binary_df(n=5, seed=2)
    X = df.drop(columns=["label"])
    csv_path = tmp_path / "new_rows.csv"
    X.to_csv(csv_path, index=False)
    result = binary_model.predict(str(csv_path))
    assert len(result) == 5


def test_predict_accepts_numpy_array_batch_and_single(binary_model):
    df = _make_binary_df(n=5, seed=3)
    X = df.drop(columns=["label"])
    batch_result = binary_model.predict(X.to_numpy())
    assert len(batch_result) == 5

    single_result = binary_model.predict(X.iloc[0].to_numpy())
    assert np.isscalar(single_result.prediction) or np.ndim(single_result.prediction) == 0


def test_predict_proba_accepts_dict_and_squeezes_single_sample(binary_model):
    sample = {"a": 0.5, "b": -0.2, "c": 0.1, "cat": "x"}
    proba = binary_model.predict_proba(sample)
    assert proba.shape == (2,)  # squeezed from (1, 2) since input was a single sample

    df = _make_binary_df(n=4, seed=4)
    batch_proba = binary_model.predict_proba(df.drop(columns=["label"]))
    assert batch_proba.shape == (4, 2)


# --------------------------------------------------------------------------- backward compatibility


def test_predict_result_is_still_a_plain_ndarray_for_isinstance_checks(binary_model):
    df = _make_binary_df(n=6, seed=5)
    result = binary_model.predict(df.drop(columns=["label"]))
    assert isinstance(result, np.ndarray)
    assert isinstance(result + 0, np.ndarray)
    for _ in result:  # iteration must yield individual predictions, not the wrapper itself
        pass
    assert list(result) == [int(v) for v in result]


def test_predict_result_csv_and_json(binary_model):
    df = _make_binary_df(n=3, seed=6)
    result = binary_model.predict(df.drop(columns=["label"]))
    lines = result.csv.strip().splitlines()
    assert lines[0] == "prediction"
    payload = json.loads(result.json)
    assert isinstance(payload, list) and len(payload) == 3


def test_predict_single_dict_json_is_flat_object(binary_model):
    sample = {"a": 0.5, "b": -0.2, "c": 0.1, "cat": "x"}
    result = binary_model.predict(sample)
    payload = json.loads(result.json)
    assert isinstance(payload, dict)
    assert "prediction" in payload


def test_removed_confidence_api_requires_predict_proba(binary_model):
    sample = {"a": 0.5, "b": -0.2, "c": 0.1, "cat": "x"}
    result = binary_model.predict(sample)
    with pytest.raises(Exception, match="predict_proba"):
        _ = result.probability
    assert binary_model.predict_proba(sample).shape == (2,)


def test_invalid_structures_and_schema_are_rejected(binary_model):
    from smarttab.exceptions import DataValidationError

    with pytest.raises(DataValidationError):
        binary_model.predict({})
    with pytest.raises(DataValidationError):
        binary_model.predict([[1, 2, 3, 4]])
    with pytest.raises(DataValidationError):
        binary_model.predict(np.array([1, 2, 3]))
    with pytest.raises(DataValidationError):
        binary_model.predict(pd.DataFrame([{"a": 1, "b": 2, "c": 3}]))
