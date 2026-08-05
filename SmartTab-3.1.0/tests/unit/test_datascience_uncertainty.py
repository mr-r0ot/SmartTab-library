import numpy as np
import pandas as pd

from smarttab.datascience.uncertainty import ConformalPredictor, OODDetector, ProbabilityCalibrator


def test_probability_calibration_preserves_probability_contract():
    y = np.array([0, 0, 0, 1, 1, 1, 1, 0])
    p = np.array([0.05, 0.15, 0.4, 0.55, 0.65, 0.8, 0.95, 0.3])
    raw = np.column_stack([1 - p, p])
    calibrator = ProbabilityCalibrator(task="binary").fit(y, raw, method="sigmoid")
    calibrated = calibrator.transform(raw)
    assert calibrated.shape == raw.shape
    assert np.allclose(calibrated.sum(axis=1), 1.0)
    assert np.all((calibrated >= 0) & (calibrated <= 1))


def test_conformal_classification_sets_are_never_empty():
    y = np.array([0, 1, 0, 1, 0, 1])
    probabilities = np.array([
        [0.9, 0.1], [0.2, 0.8], [0.8, 0.2],
        [0.1, 0.9], [0.75, 0.25], [0.3, 0.7],
    ])
    conformal = ConformalPredictor(task="binary", alpha=0.1).fit(y, probabilities)
    sets = conformal.prediction_set(np.array([[0.5, 0.5], [0.01, 0.02]]))
    assert sets.any(axis=1).all()


def test_regression_conformal_interval_and_ood_detector():
    y = np.linspace(0, 1, 50)
    predictions = y + 0.05
    conformal = ConformalPredictor(task="regression").fit(y, predictions)
    lower, upper = conformal.interval(np.array([0.2, 0.7]))
    assert np.all(lower <= upper)

    train = pd.DataFrame({"a": np.linspace(-1, 1, 100), "b": np.linspace(0, 2, 100)})
    detector = OODDetector().fit(train)
    in_score = detector.score(pd.DataFrame({"a": [0.0], "b": [1.0]}))[0]
    out_score = detector.score(pd.DataFrame({"a": [100.0], "b": [-100.0]}))[0]
    assert out_score > in_score
