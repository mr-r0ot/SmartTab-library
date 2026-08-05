"""Integration coverage for removed pseudo-ensemble parameters and real optimization."""

import numpy as np
import pandas as pd
import pytest

import smarttab
from smarttab.exceptions import ConfigurationError


def _binary(n=260, seed=0, categorical=False):
    rng = np.random.default_rng(seed)
    frame = pd.DataFrame(rng.normal(size=(n, 4)), columns=["a", "b", "c", "d"])
    if categorical:
        frame["segment"] = rng.choice(["s1", "s2", "s3"], n)
        signal = frame.a + 0.6 * (frame.segment == "s1")
    else:
        signal = frame.a + 0.4 * frame.b - 0.2 * frame.c
    frame["label"] = (signal > 0).astype(int)
    return frame


@pytest.mark.parametrize(
    "kwargs",
    [
        {"multi_threshold_ensemble": True},
        {"threshold_models": 4},
        {"multi_threshold_ensemble": True, "threshold_models": 4},
    ],
)
def test_removed_pseudo_ensemble_parameters_fail_loudly(kwargs):
    with pytest.raises(ConfigurationError, match="removed parameter"):
        smarttab.fit(_binary(80), target="label", report=False, explain=False, verbose=0, **kwargs)


def test_optimize_false_runs_zero_search_trials():
    model = smarttab.fit(
        _binary(),
        target="label",
        model="lightgbm",
        optimize=False,
        n_estimators=35,
        threshold_optimization=False,
        report=False,
        explain=False,
        verbose=0,
    )
    joined = " ".join(model.notes)
    assert "zero search trials" in joined
    assert model.best_params


def test_optimize_true_respects_explicit_trial_count_and_is_baseline_aware():
    model = smarttab.fit(
        _binary(seed=5),
        target="label",
        model="lightgbm",
        optimize=True,
        n_trials=2,
        validation="holdout",
        threshold_optimization=False,
        report=False,
        explain=False,
        verbose=0,
    )
    joined = " ".join(model.notes)
    assert "/2 requested trials" in joined
    assert "baseline" in joined
    assert "retained" in joined or "rejected" in joined


def test_ensemble_auto_is_oof_based_and_has_no_threshold_ladder(tmp_path):
    frame = _binary(n=220, seed=9, categorical=True)
    model = smarttab.fit(
        frame,
        target="label",
        ensemble="auto",
        optimize=False,
        cv=2,
        xgboost_policy="never",
        threshold_optimization=False,
        report=False,
        explain=False,
        static_charts=False,
        verbose=0,
    )
    assert model.model_name in {"catboost", "lightgbm", "voting", "stacking"}
    assert not hasattr(model, "multi_threshold_ensemble")
    assert not hasattr(model, "threshold_ladder")
    assert any("OOF" in note or "selection scores" in note for note in model.notes)
    report = model.report(tmp_path / "report")
    assert "multi_threshold_ensemble" not in report
    assert "threshold_ladder" not in report
    assert "Multi-Threshold Ensemble" not in (tmp_path / "report" / "report.html").read_text()
