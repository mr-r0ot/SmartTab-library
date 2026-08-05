import numpy as np
import pandas as pd

from smarttab.analysis.dataset_analyzer import DatasetProfile, TaskType
from smarttab.hardware.resource_planner import ResourcePlan
from smarttab.training.ensemble import StackingEnsemble, VotingEnsemble, train_voting_stacking_ensemble
from smarttab.training.trainer import predict, predict_proba


def _classification_data(n=140, seed=0):
    rng = np.random.default_rng(seed)
    features = pd.DataFrame(rng.normal(size=(n, 4)), columns=["a", "b", "c", "d"])
    target = (features["a"] + features["b"] * 0.5 > 0).astype(int).to_numpy()
    profile = DatasetProfile(
        task_type=TaskType.BINARY,
        target_column="target",
        target_columns=["target"],
        n_samples=n,
        n_features=4,
        feature_columns=list(features.columns),
        numeric_columns=list(features.columns),
    )
    return features, target, profile


def _resource_plan():
    return ResourcePlan(cpu_threads=2, use_gpu=False, memory_budget_mb=4000)


def test_explicit_voting_uses_catboost_and_lightgbm_and_refits():
    X, y, profile = _classification_data()
    result = train_voting_stacking_ensemble(
        X,
        y,
        TaskType.BINARY,
        profile,
        _resource_plan(),
        cat_features=[],
        optimize=False,
        n_trials=2,
        cv=2,
        strategy="voting",
        xgboost_policy="never",
        threshold_optimization=False,
        verbose=0,
    )
    assert result.strategy == "voting"
    assert set(result.base_params) == {"catboost", "lightgbm"}
    assert isinstance(result.estimator, VotingEnsemble)
    assert {entry[1] for entry in result.estimator.base_models} == {"catboost", "lightgbm"}
    assert len(result.members) == 2
    assert result.voting_weights
    assert np.isclose(sum(result.voting_weights.values()), 1.0)
    assert all("model_size_mb" in member for member in result.members)
    proba = predict_proba(result.estimator, result.strategy, X.head(10))
    assert proba.shape == (10, 2)
    np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-6)


def test_explicit_stacking_uses_a_real_meta_model():
    X, y, profile = _classification_data(seed=2)
    result = train_voting_stacking_ensemble(
        X,
        y,
        TaskType.BINARY,
        profile,
        _resource_plan(),
        cat_features=[],
        optimize=False,
        cv=2,
        strategy="stacking",
        xgboost_policy="never",
        threshold_optimization=False,
        verbose=0,
    )
    assert result.strategy == "stacking"
    assert isinstance(result.estimator, StackingEnsemble)
    assert result.meta_model_name in {"linear", "catboost", "lightgbm"}
    assert len(predict(result.estimator, result.strategy, X.head(5))) == 5


def test_hybrid_fusion_builds_bounded_modality_specialists():
    X, y, profile = _classification_data(n=150, seed=17)
    X = X.copy()
    for index in range(8):
        X[f"mm__text__f{index}"] = np.sin(X["a"] * (index + 1))
    profile.n_features = X.shape[1]
    profile.feature_columns = list(X.columns)
    profile.numeric_columns = list(X.columns)
    groups = {
        "all": list(X.columns),
        "tabular": ["a", "b", "c", "d"],
        "modality:text": [f"mm__text__f{index}" for index in range(8)],
    }
    result = train_voting_stacking_ensemble(
        X,
        y,
        TaskType.BINARY,
        profile,
        _resource_plan(),
        cat_features=[],
        optimize=False,
        cv=2,
        strategy="voting",
        xgboost_policy="never",
        threshold_optimization=False,
        ensemble_models_limit=3,
        feature_groups=groups,
        fusion="hybrid",
        verbose=0,
    )
    assert result.fusion_strategy == "hybrid"
    assert any(item.get("feature_group") == "modality:text" for item in result.candidates)
    for entry in result.estimator.base_models:
        assert len(entry) == 4
        subset = entry[3]
        if subset is not None:
            assert set(subset).issubset(X.columns)
    proba = result.estimator.predict_proba(X.head(7))
    assert proba.shape == (7, 2)
