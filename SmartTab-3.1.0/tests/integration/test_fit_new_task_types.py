"""End-to-end coverage for the three task types added on top of the original
binary/multiclass/regression pipeline: multilabel classification,
multi-output regression, and ranking."""

import numpy as np
import pandas as pd
import pytest

import smarttab


def _make_multilabel_df(n=400, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 4))
    df = pd.DataFrame(X, columns=[f"num_{i}" for i in range(4)])
    df["cat"] = rng.choice(["x", "y", "z"], size=n)
    df["has_cats"] = (X[:, 0] > 0).astype(int)
    df["has_dogs"] = (X[:, 1] > 0.3).astype(int)
    return df


def _make_multioutput_df(n=400, seed=0):
    # Each target depends on a *combination* of features with real noise, not a near-1:1
    # rescaling of a single column — otherwise SmartTab's leakage protection correctly (and
    # intentionally) auto-drops the near-duplicate-of-target column, which is what a too-clean
    # synthetic signal like `price = num_0 * 3 + tiny_noise` would trigger.
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 4))
    noise = rng.normal(scale=1.5, size=(n, 2))
    df = pd.DataFrame(X, columns=[f"num_{i}" for i in range(4)])
    df["cat"] = rng.choice(["x", "y", "z"], size=n)
    df["price"] = X[:, 0] * 3.0 + X[:, 2] * 1.5 + noise[:, 0]
    df["demand"] = X[:, 1] * -2.0 + X[:, 3] * 1.0 + noise[:, 1]
    return df


def _make_ranking_df(n_groups=120, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for g in range(n_groups):
        n_items = rng.integers(3, 9)
        for _ in range(n_items):
            f1, f2 = rng.normal(), rng.normal()
            rel = int(np.clip(round(f1 * 1.5 + rng.normal(scale=0.3)), 0, 3))
            rows.append({"f1": f1, "f2": f2, "query_id": g, "relevance": rel})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- multilabel


@pytest.fixture(scope="module")
def multilabel_model():
    df = _make_multilabel_df()
    return smarttab.fit(df, target=["has_cats", "has_dogs"], optimize=False, verbose=0, report=False)


def test_multilabel_task_detected_and_trained(multilabel_model):
    assert multilabel_model.task_type.value == "multilabel"
    assert multilabel_model.model_name in ("catboost", "lightgbm")


def test_multilabel_predict_and_proba_shapes(multilabel_model):
    df = _make_multilabel_df(n=20, seed=9)
    X = df.drop(columns=["has_cats", "has_dogs"])
    preds = multilabel_model.predict(X)
    proba = multilabel_model.predict_proba(X)
    assert preds.shape == (20, 2)
    assert proba.shape == (20, 2)
    assert set(np.unique(preds)) <= {0, 1}


def test_multilabel_evaluate_and_report(multilabel_model, tmp_path):
    df = _make_multilabel_df(n=60, seed=3)
    X, y = df.drop(columns=["has_cats", "has_dogs"]), df[["has_cats", "has_dogs"]]
    metrics = multilabel_model.evaluate(X, y)
    for key in ("subset_accuracy", "hamming_loss", "f1_macro", "f1_micro"):
        assert key in metrics

    folder = tmp_path / "multilabel_report"
    report_dict = multilabel_model.report(str(folder))
    assert (folder / "report.html").exists()
    assert (folder / "report.json").exists()
    assert report_dict["task_type"] == "multilabel"


def test_multilabel_save_load_roundtrip(multilabel_model, tmp_path):
    df = _make_multilabel_df(n=15, seed=5)
    X = df.drop(columns=["has_cats", "has_dogs"])
    original = multilabel_model.predict(X)
    bundle_path = multilabel_model.save(str(tmp_path / "ml.smarttab"))
    loaded = smarttab.load(bundle_path, trusted=True)
    np.testing.assert_array_equal(original, loaded.predict(X))
    assert loaded.task_type.value == "multilabel"


# --------------------------------------------------------------------------- multi-output regression


@pytest.fixture(scope="module")
def multioutput_model():
    df = _make_multioutput_df()
    return smarttab.fit(df, target=["price", "demand"], optimize=False, verbose=0, report=False)


def test_multioutput_task_detected_and_trained(multioutput_model):
    assert multioutput_model.task_type.value == "multioutput_regression"


def test_multioutput_predict_shape_and_quality(multioutput_model):
    df = _make_multioutput_df(n=100, seed=11)
    X = df.drop(columns=["price", "demand"])
    preds = multioutput_model.predict(X)
    assert preds.shape == (100, 2)

    metrics = multioutput_model.evaluate(X, df[["price", "demand"]])
    assert "rmse_per_output" in metrics
    assert len(metrics["rmse_per_output"]) == 2
    assert metrics["r2"] > 0.5  # clean synthetic signal; sanity-check it actually learned


def test_multioutput_report_and_roundtrip(multioutput_model, tmp_path):
    df = _make_multioutput_df(n=40, seed=6)
    X, y = df.drop(columns=["price", "demand"]), df[["price", "demand"]]
    folder = tmp_path / "multioutput_report"
    report_dict = multioutput_model.report(str(folder), X, y)
    assert report_dict["task_type"] == "multioutput_regression"
    assert (folder / "charts").is_dir()

    original = multioutput_model.predict(X.head(5))
    bundle_path = multioutput_model.save(str(tmp_path / "mo.smarttab"))
    loaded = smarttab.load(bundle_path, trusted=True)
    np.testing.assert_allclose(original, loaded.predict(X.head(5)), rtol=1e-6)


# --------------------------------------------------------------------------- ranking


@pytest.fixture(scope="module")
def ranking_model():
    df = _make_ranking_df()
    return smarttab.fit(df, target="relevance", group_id="query_id", optimize=False, verbose=0, report=False)


def test_ranking_task_detected_and_trained(ranking_model):
    assert ranking_model.task_type.value == "ranking"
    assert ranking_model.dataset_profile.group_id_column == "query_id"


def test_ranking_predict_returns_scores(ranking_model):
    df = _make_ranking_df(n_groups=10, seed=2)
    X = df.drop(columns=["relevance", "query_id"])
    preds = ranking_model.predict(X)
    assert len(preds) == len(X)
    assert np.issubdtype(preds.dtype, np.floating)


def test_ranking_predict_proba_raises(ranking_model):
    df = _make_ranking_df(n_groups=5, seed=2)
    X = df.drop(columns=["relevance", "query_id"])
    from smarttab.exceptions import SmartTabError

    with pytest.raises(SmartTabError):
        ranking_model.predict_proba(X)


def test_ranking_evaluate_requires_groups(ranking_model):
    df = _make_ranking_df(n_groups=10, seed=4)
    X, y = df.drop(columns=["relevance", "query_id"]), df["relevance"]
    from smarttab.exceptions import SmartTabError

    with pytest.raises(SmartTabError):
        ranking_model.evaluate(X, y)  # no groups=...

    metrics = ranking_model.evaluate(X, y, groups=df["query_id"])
    assert "ndcg@10" in metrics


def test_ranking_report_and_roundtrip(ranking_model, tmp_path):
    df = _make_ranking_df(n_groups=15, seed=7)
    X, y = df.drop(columns=["relevance", "query_id"]), df["relevance"]
    folder = tmp_path / "ranking_report"
    report_dict = ranking_model.report(str(folder), X, y, groups=df["query_id"])
    assert report_dict["task_type"] == "ranking"
    assert (folder / "report.html").exists()

    sample = df.drop(columns=["relevance", "query_id"]).head(5)
    original = ranking_model.predict(sample)
    bundle_path = ranking_model.save(str(tmp_path / "rank.smarttab"))
    loaded = smarttab.load(bundle_path, trusted=True)
    np.testing.assert_allclose(original, loaded.predict(sample), rtol=1e-6)


def test_ensemble_rejected_for_new_task_types():
    from smarttab.exceptions import ConfigurationError

    df = _make_multilabel_df(n=50)
    with pytest.raises(ConfigurationError):
        smarttab.fit(df, target=["has_cats", "has_dogs"], ensemble="voting", verbose=0, report=False)


def test_multilabel_string_targets_return_original_label_space():
    frame = _make_multilabel_df(n=180, seed=17)
    frame["has_cats"] = frame["has_cats"].map({0: "no", 1: "yes"})
    frame["has_dogs"] = frame["has_dogs"].map({0: "absent", 1: "present"})
    model = smarttab.fit(
        frame,
        target=["has_cats", "has_dogs"],
        optimize=False,
        report=False,
        explain=False,
        verbose=0,
    )
    predictions = model.predict(frame.drop(columns=["has_cats", "has_dogs"]).head(20))
    assert set(predictions[:, 0]) <= {"no", "yes"}
    assert set(predictions[:, 1]) <= {"absent", "present"}
