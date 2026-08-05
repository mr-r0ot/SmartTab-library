import numpy as np
import pandas as pd

import smarttab


def _classification_frame(seed=7):
    rng = np.random.default_rng(seed)
    n = 420
    x = rng.normal(size=n)
    category = np.where(x > 0, "positive", "negative").astype(object)
    category[rng.choice(n, 25, replace=False)] = None
    skewed = np.exp(rng.normal(size=n))
    skewed[rng.choice(n, 16, replace=False)] *= 100
    x[rng.choice(n, 20, replace=False)] = np.nan
    target = (np.nan_to_num(x) + rng.normal(scale=0.5, size=n) > 0).astype(int)
    return pd.DataFrame({"x": x, "skewed": skewed, "category": category, "target": target})


def test_data_science_pipeline_report_uncertainty_drift_and_roundtrip(tmp_path):
    frame = _classification_frame()
    model = smarttab.fit(
        frame,
        target="target",
        task_type="binary",
        optimize=False,
        n_trials=0,
        n_estimators=60,
        report=False,
        explain=False,
        device="cpu",
        data_science={
            "numeric_imputation": "mean",
            "rare_category_min_frequency": 0.03,
            "numeric_transform": "auto",
            "winsorize": 0.02,
            "calibration": "sigmoid",
            "conformal": True,
            "ood_detection": True,
            "drift_monitoring": True,
        },
        verbose=0,
    )
    assert model.data_quality_report["quality_score"] <= 100
    assert model.cleaning_pipeline.report_.numeric_imputation == "mean"
    assert model.probability_calibrator is not None
    assert model.conformal_predictor is not None
    assert model.ood_detector is not None
    assert model.drift_reference is not None
    assert model.predict_set(frame.drop(columns="target").head(5))

    shifted = frame.drop(columns="target").head(30).copy()
    shifted["x"] = shifted["x"].fillna(0) + 100
    assert model.drift_report(shifted)["overall_score"] > 0

    report = model.report(tmp_path / "report")
    assert report["data_quality_report"]
    assert report["cleaning_report"]
    assert report["uncertainty_info"]
    assert report["evaluation_drift_report"] is not None

    path = tmp_path / "model.smarttab"
    model.save(path)
    restored = smarttab.load(path, trusted=True)
    assert restored.predict_set(frame.drop(columns="target").head(3))
    assert restored.drift_reference is not None
