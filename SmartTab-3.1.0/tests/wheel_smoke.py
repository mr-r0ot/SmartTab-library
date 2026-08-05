"""Executed against an installed wheel, not the source checkout."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

import smarttab


def main() -> None:
    rng = np.random.default_rng(11)
    frame = pd.DataFrame(
        {
            "value": rng.normal(size=160),
            "category": rng.choice(["a", "b", "c"], size=160),
        }
    )
    frame["target"] = (frame["value"] + (frame["category"] == "a") * 0.4 > 0).astype(int)
    frame.loc[[2, 9, 31], "value"] = np.nan
    frame.loc[[5, 19], "category"] = "rare_code"
    audit = smarttab.audit(frame, target="target")
    assert audit.n_rows == len(frame)
    assert audit.quality_score <= 100.0
    model = smarttab.fit(
        frame,
        target="target",
        model="lightgbm",
        optimize=False,
        n_estimators=25,
        report=False,
        explain=False,
        static_charts=False,
        verbose=0,
        data_science={
            "calibration": "sigmoid",
            "conformal": True,
            "ood_detection": True,
            "drift_monitoring": True,
            "rare_category_min_frequency": 0.03,
        },
    )
    features = frame.drop(columns="target").head(8)
    original = model.predict(features)
    uncertainty = model.predict_with_uncertainty(features)
    assert len(uncertainty["prediction"]) == len(features)
    assert model.ood_score(features).shape == (len(features),)
    assert model.drift_report(features)["severity"] in {"ok", "warning", "critical"}
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        report = model.report(root / "report")
        assert (root / "report" / "report.html").exists()
        assert json.loads((root / "report" / "report.json").read_text()) == report
        assert report["data_quality_report"]
        assert report["cleaning_report"]
        assert report["uncertainty_info"]
        assert report["evaluation_drift_report"]
        bundle = model.save(root / "model.smarttab")
        loaded = smarttab.load(bundle, trusted=True)
        np.testing.assert_array_equal(original, loaded.predict(features))
        np.testing.assert_allclose(model.ood_score(features), loaded.ood_score(features))

        texts = [
            ("excellent reliable product " if index % 2 else "broken unreliable product ")
            + str(index)
            for index in range(40)
        ]
        labels = [index % 2 for index in range(40)]
        text_model = smarttab.fit_text(
            texts,
            labels,
            model="lightgbm",
            optimize=False,
            n_trials=0,
            n_estimators=15,
            feature_budget=48,
            speed_accuracy=0.1,
            report=False,
            explain=False,
            verbose=0,
            duplicate_policy="keep",
        )
        assert text_model.feature_space["generated_features"] <= 48
        assert text_model.transform_features(["one raw document"]).shape[1] == len(
            text_model.feature_names
        )
        text_bundle = text_model.save(root / "text.smarttab")
        loaded_text = smarttab.load(text_bundle, trusted=True)
        np.testing.assert_array_equal(
            text_model.predict(texts[:4]),
            loaded_text.predict(texts[:4]),
        )
    print(f"SmartTab wheel smoke test passed: {smarttab.__version__}")


if __name__ == "__main__":
    main()
