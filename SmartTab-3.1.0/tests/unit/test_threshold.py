import numpy as np
import pytest

from smarttab.optimization.threshold import (
    DEFAULT_REJECT_THRESHOLD,
    DEFAULT_THRESHOLD,
    apply_threshold,
    optimize_multiclass_reject_threshold,
    optimize_per_label_thresholds,
    optimize_threshold,
)


def test_apply_threshold():
    np.testing.assert_array_equal(apply_threshold([0.2, 0.5, 0.9], 0.5), [0, 1, 1])


@pytest.mark.parametrize("objective", ["f1", "precision", "recall", "accuracy", "balanced_accuracy", "mcc"])
def test_threshold_objectives_return_valid_values(objective):
    y = np.array([0, 0, 0, 1, 1, 1])
    p = np.array([0.05, 0.2, 0.4, 0.45, 0.7, 0.9])
    threshold, score = optimize_threshold(y, p, objective)
    assert 0.01 <= threshold <= 0.99
    assert -1.0 <= score <= 1.0


def test_roc_auc_does_not_claim_threshold_optimization():
    y = np.array([0, 0, 1, 1])
    p = np.array([0.1, 0.4, 0.6, 0.9])
    threshold, score = optimize_threshold(y, p, "roc_auc")
    assert threshold == DEFAULT_THRESHOLD
    assert score == pytest.approx(1.0)


def test_single_class_falls_back():
    threshold, score = optimize_threshold(np.zeros(4), np.linspace(0, 1, 4))
    assert threshold == DEFAULT_THRESHOLD
    assert score == 0.0


def test_per_label_thresholds_validate_shape():
    y = np.array([[0, 1], [1, 0], [1, 1]])
    p = np.array([[0.1, 0.8], [0.7, 0.2], [0.9, 0.6]])
    assert len(optimize_per_label_thresholds(y, p)) == 2
    with pytest.raises(ValueError):
        optimize_per_label_thresholds(y, p[:, :1])


def test_multiclass_rejection_is_explicit_noop():
    y = np.array([0, 1, 2])
    proba = np.eye(3)
    threshold, score = optimize_multiclass_reject_threshold(y, proba)
    assert threshold == DEFAULT_REJECT_THRESHOLD
    assert score == pytest.approx(1.0)
