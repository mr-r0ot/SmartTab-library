import numpy as np
import pandas as pd
import pytest

import smarttab


def _make_regression_df(n=400, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 5))
    noise = rng.normal(scale=0.1, size=n)
    y = X[:, 0] * 3.0 - X[:, 1] * 1.5 + noise
    df = pd.DataFrame(X, columns=[f"num_{i}" for i in range(5)])
    df["category"] = rng.choice(["a", "b", "c"], size=n)
    df.loc[rng.choice(n, size=15, replace=False), "num_0"] = np.nan
    df["price"] = y
    return df


@pytest.fixture(scope="module")
def fitted_model():
    df = _make_regression_df()
    return smarttab.fit(
        df, target="price", n_trials=3, cv=3, timeout=120, verbose=0,
        report=False,
    )


def test_fit_returns_regression_model(fitted_model):
    assert fitted_model.model_name in ("catboost", "lightgbm")
    assert fitted_model.task_type.value == "regression"


def test_predict_returns_floats(fitted_model):
    df = _make_regression_df(n=20, seed=99)
    preds = fitted_model.predict(df.drop(columns=["price"]))
    assert len(preds) == 20
    assert np.issubdtype(preds.dtype, np.floating)


def test_evaluate_returns_full_regression_metrics(fitted_model):
    df = _make_regression_df(n=50, seed=7)
    results = fitted_model.evaluate(df.drop(columns=["price"]), df["price"])
    for key in ("mae", "mse", "rmse", "r2", "median_ae"):
        assert key in results
    assert results["r2"] > 0.5  # signal is strong and noise is small; sanity check the pipeline learned something


def test_report_generates_html_json_with_regression_chart(fitted_model, tmp_path):
    df = _make_regression_df(n=50, seed=7)
    folder = tmp_path / "report_out"
    report_dict = fitted_model.report(str(folder), df.drop(columns=["price"]), df["price"])

    content = (folder / "report.html").read_text(encoding="utf-8")
    assert "Predicted vs Actual" in content
    assert (folder / "report.json").exists()
    assert report_dict["task_type"] == "regression"
    assert "rmse" in report_dict["metrics"]


def test_save_load_roundtrip_predictions_match(fitted_model, tmp_path):
    df = _make_regression_df(n=30, seed=3)
    X = df.drop(columns=["price"])

    original_preds = fitted_model.predict(X)
    bundle_path = fitted_model.save(str(tmp_path / "model.smarttab"))

    loaded = smarttab.load(bundle_path, trusted=True)
    loaded_preds = loaded.predict(X)

    np.testing.assert_allclose(original_preds, loaded_preds, rtol=1e-6)
