import numpy as np
import pandas as pd
import pytest

import smarttab


def _make_imbalanced_noise_df(n=1000, minority_fraction=0.05, seed=0):
    """Imbalanced target with no real relationship to X — enough to exercise the
    class-weighting wiring without needing the model to actually learn anything."""
    rng = np.random.default_rng(seed)
    n_minority = int(n * minority_fraction)
    n_majority = n - n_minority
    X = rng.normal(size=(n, 4))
    df = pd.DataFrame(X, columns=[f"num_{i}" for i in range(4)])
    df["label"] = np.array([0] * n_majority + [1] * n_minority)
    return df.sample(frac=1.0, random_state=seed).reset_index(drop=True)


def _make_imbalanced_separable_df(n=1500, minority_fraction=0.05, seed=1):
    """Imbalanced target that *is* learnable from X, so a properly class-weighted
    model has a real shot at recognizing the minority class instead of collapsing
    to "always predict majority" (which would otherwise look like high accuracy)."""
    rng = np.random.default_rng(seed)
    n_minority = int(n * minority_fraction)
    n_majority = n - n_minority
    majority_X = rng.normal(loc=0.0, scale=1.0, size=(n_majority, 4))
    minority_X = rng.normal(loc=3.0, scale=1.0, size=(n_minority, 4))
    X = np.vstack([majority_X, minority_X])
    y = np.array([0] * n_majority + [1] * n_minority)
    df = pd.DataFrame(X, columns=[f"num_{i}" for i in range(4)])
    df["label"] = y
    return df.sample(frac=1.0, random_state=seed).reset_index(drop=True)


@pytest.mark.parametrize("optimize", [False, True])
def test_imbalance_applies_class_weight_regardless_of_optimize_path(optimize):
    df = _make_imbalanced_noise_df()
    model = smarttab.fit(
        df, target="label", optimize=optimize, n_trials=3, cv=3, verbose=0, report=False,
    )
    assert model.dataset_profile.is_imbalanced is True
    assert model.primary_metric != "accuracy"

    weight_key = "auto_class_weights" if model.model_name == "catboost" else "class_weight"
    assert weight_key in model.best_params


def test_imbalance_applies_class_weight_when_params_explicitly_given():
    df = _make_imbalanced_noise_df()
    model = smarttab.fit(
        df, target="label", model="lightgbm", params={"num_leaves": 15}, verbose=0, report=False,
    )
    assert model.best_params["num_leaves"] == 15
    assert model.best_params.get("class_weight") == "balanced"


def test_imbalanced_but_separable_model_recognizes_minority_class():
    df = _make_imbalanced_separable_df()
    model = smarttab.fit(df, target="label", optimize=False, verbose=0, report=False)
    preds = model.predict(df.drop(columns=["label"]))
    assert set(np.unique(preds)) == {0, 1}, "model collapsed to predicting a single class"

    metrics = model.evaluate(df.drop(columns=["label"]), df["label"])
    assert metrics["recall"] > 0.5, "class weighting should let the model actually catch minority cases"


def test_custom_test_size_is_honored():
    df = _make_imbalanced_noise_df(n=500, minority_fraction=0.3, seed=2)
    model = smarttab.fit(df, target="label", test_size=0.5, optimize=False, verbose=0, report=False)
    assert model.dataset_profile.source_n_samples == 500
    assert model.dataset_profile.n_samples == 250
    assert model.dataset_profile.holdout_n_samples == 250
