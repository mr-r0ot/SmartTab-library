import json
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from smarttab.analysis.dataset_analyzer import analyze_dataset
from smarttab.cleaning.encoders import TargetLabelEncoder
from smarttab.cleaning.pipeline import SmartCleaningPipeline
from smarttab.exceptions import SmartTabError
from smarttab.hardware.profiler import CPUInfo, DiskInfo, GPUInfo, HardwareProfile, RAMInfo
from smarttab.hardware.resource_planner import ResourcePlan
from smarttab.persistence.serializer import BUNDLE_FORMAT_VERSION, load_bundle, save_bundle
from smarttab.training.trainer import build_estimator, fit_estimator, predict, predict_proba


def _fitted_bundle(model_name, tmp_path):
    rng = np.random.default_rng(0)
    frame = pd.DataFrame(
        {
            "num_a": rng.normal(size=120),
            "cat_a": rng.choice(["a", "b", "c"], size=120),
            "target": rng.integers(0, 2, size=120),
        }
    )
    profile = analyze_dataset(frame, "target")
    pipeline = SmartCleaningPipeline()
    X = pipeline.fit_transform(frame, frame["target"], profile)
    encoder = TargetLabelEncoder().fit(frame["target"])
    y = encoder.transform(frame["target"])
    estimator = build_estimator(
        model_name,
        params={},
        task_type=profile.task_type,
        n_estimators=15,
        cpu_threads=2,
        use_gpu=False,
        random_state=42,
    )
    fit_estimator(estimator, model_name, X, y, cat_features=pipeline.final_categorical_columns)
    hardware = HardwareProfile(
        cpu=CPUInfo(physical_cores=4, logical_cores=8),
        ram=RAMInfo(total_mb=16000, available_mb=8000),
        gpu=GPUInfo(available=False),
        disk=DiskInfo(kind="unknown"),
    )
    resource = ResourcePlan(cpu_threads=2, use_gpu=False, memory_budget_mb=4000)
    bundle = save_bundle(
        tmp_path / model_name,
        model_name=model_name,
        task_type=profile.task_type,
        estimator=estimator,
        cleaning_pipeline=pipeline,
        target_encoder=encoder,
        raw_feature_names=list(pipeline.raw_feature_columns_),
        feature_names=list(X.columns),
        cat_features=pipeline.final_categorical_columns,
        dataset_profile=profile,
        hardware_profile=hardware,
        resource_plan=resource,
        best_params={},
        primary_metric="roc_auc",
        metrics={"roc_auc": 0.5},
        class_labels=encoder.classes_.tolist(),
        timings={"training_seconds": 1.0},
        notes=["test"],
    )
    return Path(bundle), X, estimator


@pytest.mark.parametrize("model_name", ["catboost", "lightgbm"])
def test_roundtrip_predictions_and_probabilities(tmp_path, model_name):
    bundle, X, original = _fitted_bundle(model_name, tmp_path)
    with pytest.raises(SmartTabError, match="trusted=True"):
        load_bundle(bundle)
    loaded = load_bundle(bundle, trusted=True)
    np.testing.assert_array_equal(
        predict(original, model_name, X),
        predict(loaded["estimator"], model_name, X),
    )
    np.testing.assert_allclose(
        predict_proba(original, model_name, X),
        predict_proba(loaded["estimator"], model_name, X),
        rtol=1e-6,
    )
    assert loaded["raw_feature_names"] == ["num_a", "cat_a"]
    assert loaded["timings"]["training_seconds"] == 1.0


def test_bundle_has_versioned_manifest_and_hashes(tmp_path):
    bundle, _, _ = _fitted_bundle("lightgbm", tmp_path)
    with zipfile.ZipFile(bundle) as archive:
        manifest = json.loads(archive.read("manifest.json"))
        metadata = json.loads(archive.read("meta.json"))
    assert manifest["bundle_format_version"] == BUNDLE_FORMAT_VERSION
    assert metadata["bundle_format_version"] == BUNDLE_FORMAT_VERSION
    assert "pipeline.joblib" in manifest["files"]
    assert len(manifest["files"]["pipeline.joblib"]["sha256"]) == 64
    assert metadata["created_with"]["python"]


def test_integrity_mismatch_is_rejected_before_deserialization(tmp_path):
    bundle, _, _ = _fitted_bundle("lightgbm", tmp_path)
    tampered = tmp_path / "tampered.smarttab"
    with zipfile.ZipFile(bundle) as source, zipfile.ZipFile(tampered, "w", zipfile.ZIP_DEFLATED) as target:
        for info in source.infolist():
            payload = source.read(info.filename)
            if info.filename == "pipeline.joblib":
                payload += b"corruption"
            target.writestr(info, payload)
    with pytest.raises(SmartTabError, match="integrity check failed"):
        load_bundle(tampered, trusted=True)


def test_path_traversal_archive_is_rejected(tmp_path):
    malicious = tmp_path / "traversal.smarttab"
    with zipfile.ZipFile(malicious, "w") as archive:
        archive.writestr("../outside", b"x")
    with pytest.raises(SmartTabError, match="unsafe path"):
        load_bundle(malicious, trusted=True)


def test_future_format_is_rejected(tmp_path):
    bundle, _, _ = _fitted_bundle("catboost", tmp_path)
    future = tmp_path / "future.smarttab"
    with zipfile.ZipFile(bundle) as source:
        payloads = {info.filename: source.read(info.filename) for info in source.infolist()}
    metadata = json.loads(payloads["meta.json"])
    metadata["bundle_format_version"] = BUNDLE_FORMAT_VERSION + 1
    payloads["meta.json"] = json.dumps(metadata).encode()
    # Update the manifest hash so this reaches the format-version check.
    import hashlib
    manifest = json.loads(payloads["manifest.json"])
    manifest["files"]["meta.json"] = {
        "sha256": hashlib.sha256(payloads["meta.json"]).hexdigest(),
        "size": len(payloads["meta.json"]),
    }
    payloads["manifest.json"] = json.dumps(manifest).encode()
    with zipfile.ZipFile(future, "w", zipfile.ZIP_DEFLATED) as target:
        for name, payload in payloads.items():
            target.writestr(name, payload)
    with pytest.raises(SmartTabError, match="unsupported bundle format"):
        load_bundle(future, trusted=True)


def test_missing_bundle_raises(tmp_path):
    with pytest.raises(SmartTabError):
        load_bundle(tmp_path / "missing.smarttab", trusted=True)
