import numpy as np
import pandas as pd

from smarttab.datascience.quality import audit_data_quality


def test_quality_audit_reports_actionable_data_problems(tmp_path):
    frame = pd.DataFrame(
        {
            "numeric": [1.0, np.inf, 3.0, 1000.0, np.nan, 1.0],
            "category": ["a", "b", "c", "d", "e", "a"],
            "text": ["valid", "", None, "another", "sample", "valid"],
            "image": [str(tmp_path / "missing.png"), None, "", "x.png", "y.png", str(tmp_path / "missing.png")],
            "target": [0, 1, 0, 1, 0, 0],
        }
    )
    report = audit_data_quality(
        frame,
        target_columns=["target"],
        column_modalities={"text": "text", "image": "image"},
    )
    codes = {issue.code for issue in report.issues}
    assert "infinite_values" in codes
    assert "modality_missing" in codes
    assert "unreadable_media_paths" in codes
    assert 0.0 <= report.quality_score <= 100.0
    assert report.recommendations


def test_quality_audit_handles_array_valued_rows_and_duplicates():
    image = np.zeros((4, 4, 3), dtype=np.uint8)
    frame = pd.DataFrame({"image": [image, image.copy()], "target": [1, 1]})
    report = audit_data_quality(
        frame,
        target_columns=["target"],
        column_modalities={"image": "image"},
    )
    assert any(issue.code == "duplicate_rows" for issue in report.issues)


def test_public_audit_returns_structured_report():
    import smarttab

    frame = pd.DataFrame({"x": [1.0, None, 3.0], "label": [0, 1, 0]})
    report = smarttab.audit(frame, target="label")
    assert report.n_rows == 3
    assert report["n_rows"] == 3
    assert report.get("quality_score") == report.quality_score
    assert report.to_dict()["target_summary"]["label"]["unique"] == 2
