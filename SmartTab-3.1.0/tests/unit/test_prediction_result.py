import json

import numpy as np
import pytest

from smarttab.exceptions import SmartTabError
from smarttab.prediction_result import PredictionArray, _predictions_to_frame


def test_prediction_array_is_numpy_compatible():
    result = PredictionArray([1, 0, 1])
    assert isinstance(result, np.ndarray)
    np.testing.assert_array_equal(result + 1, [2, 1, 2])
    assert result.prediction.tolist() == [1, 0, 1]
    assert result.label.tolist() == [1, 0, 1]


def test_single_prediction_is_scalar_convenience():
    result = PredictionArray(["yes"], single=True)
    assert result.prediction == "yes"
    assert json.loads(result.json) == {"prediction": "yes"}


def test_batch_and_multioutput_serialization():
    batch = PredictionArray([1, 0])
    assert batch.csv.splitlines()[0] == "prediction"
    assert len(json.loads(batch.json)) == 2
    multi = _predictions_to_frame(np.array([[1.0, 2.0], [3.0, 4.0]]))
    assert list(multi.columns) == ["output_0", "output_1"]


def test_probability_property_directs_to_predict_proba():
    with pytest.raises(SmartTabError, match="predict_proba"):
        _ = PredictionArray([1]).probability
