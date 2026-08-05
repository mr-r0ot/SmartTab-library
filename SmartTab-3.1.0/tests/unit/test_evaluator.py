import numpy as np
import pytest
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score

from smarttab.evaluation.evaluator import compute_metric, evaluate_classification, evaluate_regression


def test_evaluate_classification_matches_sklearn_accuracy():
    y_true = np.array([0, 1, 1, 0, 1, 0, 1, 1])
    y_pred = np.array([0, 1, 0, 0, 1, 1, 1, 1])
    results = evaluate_classification(y_true, y_pred)
    assert results["accuracy"] == pytest.approx(accuracy_score(y_true, y_pred))


def test_evaluate_classification_includes_roc_auc_when_proba_given():
    y_true = np.array([0, 1, 1, 0, 1, 0])
    y_pred = np.array([0, 1, 1, 0, 0, 0])
    y_proba = np.array(
        [[0.9, 0.1], [0.2, 0.8], [0.3, 0.7], [0.8, 0.2], [0.6, 0.4], [0.7, 0.3]]
    )
    results = evaluate_classification(y_true, y_pred, y_proba)
    assert "roc_auc" in results
    assert 0.0 <= results["roc_auc"] <= 1.0


def test_evaluate_regression_matches_sklearn():
    y_true = np.array([3.0, 5.0, 2.5, 7.0])
    y_pred = np.array([2.8, 5.5, 2.0, 6.5])
    results = evaluate_regression(y_true, y_pred)
    assert results["rmse"] == pytest.approx(np.sqrt(mean_squared_error(y_true, y_pred)))
    assert results["r2"] == pytest.approx(r2_score(y_true, y_pred))


def test_compute_metric_rmse():
    y_true = np.array([1.0, 2.0, 3.0])
    y_pred = np.array([1.5, 2.5, 2.5])
    value = compute_metric("rmse", y_true, y_pred)
    assert value == pytest.approx(np.sqrt(mean_squared_error(y_true, y_pred)))


def test_compute_metric_unknown_raises():
    with pytest.raises(ValueError):
        compute_metric("not_a_metric", np.array([1, 2]), np.array([1, 2]))
