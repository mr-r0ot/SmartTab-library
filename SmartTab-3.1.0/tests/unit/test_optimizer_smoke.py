import numpy as np
import pandas as pd
import pytest

from smarttab.analysis.dataset_analyzer import DatasetProfile, TaskType
from smarttab.hardware.profiler import CPUInfo, DiskInfo, GPUInfo, HardwareProfile, RAMInfo
from smarttab.hardware.resource_planner import ResourcePlan
from smarttab.optimization.optimizer import resolve_cv_splitter, resolve_n_trials, resolve_primary_metric, run_optimization


def _classification_data(n=200, seed=0):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame(rng.normal(size=(n, 4)), columns=["f1", "f2", "f3", "f4"])
    y = (X["f1"] + X["f2"] > 0).astype(int).to_numpy()
    profile = DatasetProfile(
        task_type=TaskType.BINARY, target_column="target", n_samples=n, n_features=4,
        feature_columns=list(X.columns), numeric_columns=list(X.columns),
    )
    return X, y, profile


def _resource_plan(cpu_threads=8):
    return ResourcePlan(cpu_threads=cpu_threads, use_gpu=False, memory_budget_mb=4000)


def test_resolve_primary_metric_defaults():
    _, _, profile = _classification_data()
    assert resolve_primary_metric(profile) == "roc_auc"


def test_resolve_n_trials_small_dataset_uses_more_trials():
    _, _, profile = _classification_data(n=500)
    plan = _resource_plan(cpu_threads=8)
    assert resolve_n_trials(profile, plan) == 8


def test_resolve_n_trials_capped_on_weak_hardware():
    _, _, profile = _classification_data(n=500)
    plan = _resource_plan(cpu_threads=2)
    assert resolve_n_trials(profile, plan) == 6


def test_resolve_cv_splitter_uses_fast_holdout_for_large_classification():
    _, y, profile = _classification_data(n=50000)
    splitter = resolve_cv_splitter(profile, y=y)
    assert splitter.get_n_splits() == 1


@pytest.mark.parametrize("model_name", ["catboost", "lightgbm"])
def test_run_optimization_end_to_end_smoke(model_name):
    X, y, profile = _classification_data(n=200)
    plan = _resource_plan()
    result = run_optimization(
        model_name=model_name, X=X, y=y, task_type=profile.task_type, profile=profile,
        resource_plan=plan, cat_features=[], n_trials=3, cv=3,
    )
    assert isinstance(result.best_params, dict)
    assert result.n_trials_run == 3
    assert result.best_n_estimators > 0
    assert 0.0 <= result.best_score <= 1.0
