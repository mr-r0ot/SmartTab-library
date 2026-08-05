from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import smarttab


def _make_classification_df(n=400, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 5))
    y = (X[:, 0] + X[:, 1] * 0.5 > 0).astype(int)
    df = pd.DataFrame(X, columns=[f"num_{i}" for i in range(5)])
    df["category"] = rng.choice(["red", "green", "blue"], size=n)
    df["signup_date"] = pd.date_range("2021-01-01", periods=n, freq="D").astype(str)
    df.loc[rng.choice(n, size=15, replace=False), "num_0"] = np.nan
    df["label"] = y
    return df


@pytest.fixture(scope="module")
def fitted_model():
    df = _make_classification_df()
    return smarttab.fit(
        df, target="label", n_trials=3, cv=3, timeout=120, verbose=0,
        report=False,
    )


def test_fit_returns_model_with_supported_model_name(fitted_model):
    assert fitted_model.model_name in ("catboost", "lightgbm")
    assert fitted_model.task_type.value == "binary"


def test_predict_returns_original_label_space(fitted_model):
    df = _make_classification_df(n=20, seed=99)
    preds = fitted_model.predict(df.drop(columns=["label"]))
    assert len(preds) == 20
    assert set(np.unique(preds)) <= {0, 1}


def test_predict_proba_shape(fitted_model):
    df = _make_classification_df(n=20, seed=99)
    proba = fitted_model.predict_proba(df.drop(columns=["label"]))
    assert proba.shape == (20, 2)
    np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-5)


def test_evaluate_returns_full_classification_metrics(fitted_model):
    df = _make_classification_df(n=50, seed=7)
    results = fitted_model.evaluate(df.drop(columns=["label"]), df["label"])
    for key in ("accuracy", "precision", "recall", "f1", "f1_macro", "f1_weighted", "mcc", "cohen_kappa", "balanced_accuracy", "roc_auc"):
        assert key in results


def test_report_generates_html_json_and_chart_images(fitted_model, tmp_path):
    df = _make_classification_df(n=50, seed=7)
    folder = tmp_path / "report_out"
    report_dict = fitted_model.report(str(folder), df.drop(columns=["label"]), df["label"])

    html_path = folder / "report.html"
    json_path = folder / "report.json"
    assert html_path.exists()
    assert json_path.exists()

    content = html_path.read_text(encoding="utf-8")
    assert "SmartTab Report" in content
    assert "Evaluation Metrics" in content
    assert "Hardware Summary" in content
    assert "Feature Importance" in content

    assert report_dict["model_name"] == fitted_model.model_name
    assert "accuracy" in report_dict["metrics"]
    assert (folder / "charts").is_dir()
    assert report_dict["static_chart_export"]["status"] in {"ok", "partial", "failed", "unavailable", "disabled"}


def test_save_load_roundtrip_predictions_match(fitted_model, tmp_path):
    df = _make_classification_df(n=30, seed=3)
    X = df.drop(columns=["label"])

    original_preds = fitted_model.predict(X)
    bundle_path = fitted_model.save(str(tmp_path / "model.smarttab"))

    loaded = smarttab.load(bundle_path, trusted=True)
    loaded_preds = loaded.predict(X)

    np.testing.assert_array_equal(original_preds, loaded_preds)
    assert loaded.model_name == fitted_model.model_name


def test_report_folder_is_optional(fitted_model, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    report = fitted_model.report()
    assert Path(report["_paths"]["folder"]).exists()
