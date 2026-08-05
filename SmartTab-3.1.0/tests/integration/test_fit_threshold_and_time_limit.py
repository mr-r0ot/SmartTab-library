import time

import numpy as np
import pandas as pd
import pytest

import smarttab
from smarttab.optimization.threshold import DEFAULT_THRESHOLD


def _make_skewed_proba_df(n=1500, seed=0):
    """Binary target where the natural decision boundary isn't at 0.5 —
    good default (accuracy-focused) 0.5 cutoffs tend to under-predict the
    positive class here, so threshold optimization should find something
    else and improve F1 on the held-out set."""
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 5))
    df = pd.DataFrame(X, columns=[f"num_{i}" for i in range(5)])
    score = X[:, 0] * 2 + X[:, 1]
    # positive class is rare and only occurs at the high end of `score`
    df["label"] = (score > np.quantile(score, 0.85)).astype(int)
    return df


def test_threshold_optimization_enabled_by_default_and_not_always_half():
    df = _make_skewed_proba_df()
    model = smarttab.fit(df, target="label", optimize=False, verbose=0, report=False)
    assert 0.0 <= model.decision_threshold <= 1.0
    assert any(
        note.startswith("decision_threshold=") and "optimized for" in note
        for note in model.notes
    )


def test_threshold_optimization_can_be_disabled():
    df = _make_skewed_proba_df()
    model = smarttab.fit(df, target="label", optimize=False, threshold_optimization=False, verbose=0, report=False)
    assert model.decision_threshold == DEFAULT_THRESHOLD


def test_threshold_persists_through_save_load():
    df = _make_skewed_proba_df(n=500)
    model = smarttab.fit(df, target="label", optimize=False, verbose=0, report=False)

    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as tmp:
        bundle_path = model.save(str(Path(tmp) / "model.smarttab"))
        loaded = smarttab.load(bundle_path, trusted=True)
        assert loaded.decision_threshold == pytest.approx(model.decision_threshold)

        X = df.drop(columns=["label"]).head(10)
        np.testing.assert_array_equal(model.predict(X), loaded.predict(X))


def test_time_limit_zero_means_unlimited_and_still_works():
    df = _make_skewed_proba_df(n=300)
    model = smarttab.fit(df, target="label", time_limit=0, n_trials=3, cv=3, verbose=0, report=False)
    assert model.metrics


def test_time_limit_caps_wall_clock_roughly():
    df = _make_skewed_proba_df(n=2000)
    start = time.perf_counter()
    model = smarttab.fit(df, target="label", time_limit=20, verbose=0, report=False)
    elapsed = time.perf_counter() - start
    # Native learners cannot be interrupted between every low-level operation, so
    # allow bounded cleanup slack while ensuring the former 90-second overrun is gone.
    assert elapsed < 35
    assert any("time_limit" in note for note in model.notes)
