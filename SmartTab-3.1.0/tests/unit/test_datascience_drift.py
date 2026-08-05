import numpy as np
import pandas as pd

from smarttab.datascience.drift import DriftReference, compare_drift


def test_drift_reference_detects_numeric_and_categorical_shift():
    reference_raw = pd.DataFrame(
        {"x": np.linspace(0, 1, 200), "category": ["a", "b"] * 100}
    )
    reference_features = pd.DataFrame({"feature": np.linspace(-1, 1, 200)})
    reference = DriftReference.fit(
        reference_raw,
        reference_features,
        numeric_columns=["x"],
        categorical_columns=["category"],
    )
    current_raw = pd.DataFrame({"x": np.linspace(5, 8, 100), "category": ["new"] * 100})
    current_features = pd.DataFrame({"feature": np.linspace(10, 12, 100)})
    report = compare_drift(reference, current_raw, current_features)
    assert report["overall_score"] > 0.3
    assert report["severity"] in {"warning", "critical"}
    assert set(report["drifted_raw_columns"]) == {"category", "x"}
