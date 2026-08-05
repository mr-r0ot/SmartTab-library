import numpy as np
import pandas as pd
import pytest

from smarttab.analysis.dataset_analyzer import DatasetProfile, TaskType
from smarttab.hardware.resource_planner import ResourcePlan
from smarttab.training.ensemble import (
    AUTO_MIN_RELATIVE_IMPROVEMENT,
    _allocate_trials,
    _relative_improvement,
    run_ensemble_decision_engine,
)


def _data(n=130):
    rng = np.random.default_rng(12)
    X = pd.DataFrame(rng.normal(size=(n, 3)), columns=["a", "b", "c"])
    y = (X.a - X.b * 0.3 > 0).astype(int).to_numpy()
    profile = DatasetProfile(
        task_type=TaskType.BINARY,
        target_column="target",
        target_columns=["target"],
        n_samples=n,
        n_features=3,
        feature_columns=list(X.columns),
        numeric_columns=list(X.columns),
    )
    return X, y, profile


def test_trial_allocation_never_inflates_explicit_budget():
    assert _allocate_trials(1, 2) == [1, 0]
    assert _allocate_trials(5, 2) == [3, 2]
    assert sum(_allocate_trials(7, 3)) == 7


def test_relative_improvement_supports_both_directions():
    assert _relative_improvement(0.8, 0.84, "maximize") == pytest.approx(0.05)
    assert _relative_improvement(10.0, 9.0, "minimize") == pytest.approx(0.1)
    assert AUTO_MIN_RELATIVE_IMPROVEMENT > 0


def test_auto_decision_returns_best_single_or_real_ensemble():
    X, y, profile = _data()
    result = run_ensemble_decision_engine(
        X,
        y,
        TaskType.BINARY,
        profile,
        ResourcePlan(cpu_threads=2, use_gpu=False, memory_budget_mb=4000),
        cat_features=[],
        optimize=False,
        n_trials=1,
        cv=2,
        xgboost_policy="never",
        threshold_optimization=False,
        verbose=0,
    )
    assert result.strategy in {"catboost", "lightgbm", "voting", "stacking"}
    assert any("OOF" in note or "selection scores" in note for note in result.notes)
    if result.used_ensemble:
        assert result.ensemble_info is not None
    else:
        assert result.ensemble_info is None
