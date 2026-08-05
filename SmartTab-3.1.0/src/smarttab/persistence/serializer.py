"""Versioned, integrity-checked ``.smarttab`` model bundles.

A bundle contains joblib/pickle payloads for the fitted preprocessing pipeline
and, for some estimators, the estimator itself. Loading therefore requires an
explicit ``trusted=True`` acknowledgement. Integrity hashes detect corruption;
they do not make an untrusted pickle safe.
"""

from __future__ import annotations

import dataclasses
import hashlib
import importlib.metadata
import json
import os
import platform
import shutil
import stat
import sys
import tempfile
import zipfile
from pathlib import Path, PurePosixPath
from typing import Any

import joblib
import numpy as np

from smarttab import __version__
from smarttab.analysis.dataset_analyzer import DatasetProfile, TaskType
from smarttab.exceptions import SmartTabError
from smarttab.hardware.profiler import CPUInfo, DiskInfo, GPUInfo, HardwareProfile, RAMInfo
from smarttab.hardware.resource_planner import ResourcePlan
from smarttab.optimization.threshold import DEFAULT_OBJECTIVE, DEFAULT_REJECT_THRESHOLD, DEFAULT_THRESHOLD

BUNDLE_FORMAT_VERSION = 5
MAX_ARCHIVE_FILES = 200
MAX_UNCOMPRESSED_BYTES = 4 * 1024**3
MAX_COMPRESSION_RATIO = 1000.0
HASH_CHUNK_BYTES = 1024 * 1024


def save_bundle(
    path: str | Path,
    *,
    model_name: str,
    task_type: TaskType,
    estimator,
    cleaning_pipeline,
    target_encoder,
    raw_feature_names: list[str],
    feature_names: list[str],
    cat_features: list[str],
    dataset_profile: DatasetProfile,
    hardware_profile: HardwareProfile,
    resource_plan: ResourcePlan,
    best_params: dict,
    primary_metric: str,
    metrics: dict,
    class_labels: list | dict | None,
    ensemble_info: dict | None = None,
    decision_threshold: float = DEFAULT_THRESHOLD,
    reject_threshold: float = DEFAULT_REJECT_THRESHOLD,
    per_label_thresholds: list[float] | None = None,
    objective: str = DEFAULT_OBJECTIVE,
    timings: dict | None = None,
    notes: list[str] | None = None,
    static_charts: str | bool = "auto",
    probability_calibrator=None,
    conformal_predictor=None,
    ood_detector=None,
    drift_reference=None,
    data_science_config: dict | None = None,
    data_quality_report: dict | None = None,
    modality_dropout_info: dict | None = None,
) -> str:
    output_path = Path(path)
    if output_path.suffix != ".smarttab":
        output_path = output_path.with_suffix(".smarttab")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        model_records: list[dict[str, Any]] = []
        voting_weights = None

        if hasattr(estimator, "base_models"):
            base_dir = root / "base_models"
            base_dir.mkdir()
            for entry in estimator.base_models:
                if len(entry) == 4:
                    alias, base_model_name, base_estimator, feature_columns = entry
                else:
                    alias, base_model_name, base_estimator = entry
                    feature_columns = None
                safe_alias = _safe_component(alias)
                relative_base = Path("base_models") / safe_alias
                filename = _save_single_model(
                    base_model_name,
                    task_type,
                    base_estimator,
                    root / relative_base,
                )
                model_records.append(
                    {
                        "alias": alias,
                        "model_name": base_model_name,
                        "file": filename.relative_to(root).as_posix(),
                        "feature_columns": feature_columns,
                    }
                )
            if hasattr(estimator, "meta_model"):
                joblib.dump(estimator.meta_model, root / "meta_model.joblib", compress=3)
            if getattr(estimator, "weights", None) is not None:
                voting_weights = [float(value) for value in estimator.weights]
        else:
            filename = _save_single_model(model_name, task_type, estimator, root / "model")
            model_records.append(
                {"alias": model_name, "model_name": model_name, "file": filename.relative_to(root).as_posix()}
            )

        joblib.dump(cleaning_pipeline, root / "pipeline.joblib", compress=3)
        if target_encoder is not None:
            joblib.dump(target_encoder, root / "target_encoder.joblib", compress=3)
        diagnostics = {
            "probability_calibrator": probability_calibrator,
            "conformal_predictor": conformal_predictor,
            "ood_detector": ood_detector,
            "drift_reference": drift_reference,
        }
        if any(value is not None for value in diagnostics.values()):
            joblib.dump(diagnostics, root / "diagnostics.joblib", compress=3)

        metadata = {
            "bundle_format_version": BUNDLE_FORMAT_VERSION,
            "smarttab_version": __version__,
            "created_with": _runtime_manifest(),
            "model_name": model_name,
            "task_type": task_type.value,
            "model_records": model_records,
            "raw_feature_names": raw_feature_names,
            "feature_names": feature_names,
            "cat_features": cat_features,
            "best_params": best_params,
            "primary_metric": primary_metric,
            "metrics": metrics,
            "class_labels": class_labels,
            "ensemble_info": ensemble_info,
            "voting_weights": voting_weights,
            "decision_threshold": decision_threshold,
            "reject_threshold": reject_threshold,
            "per_label_thresholds": per_label_thresholds,
            "objective": objective,
            "timings": timings or {},
            "notes": notes or [],
            "static_charts": static_charts,
            "data_science_config": data_science_config or {},
            "data_quality_report": data_quality_report or {},
            "modality_dropout_info": modality_dropout_info or {},
            "dataset_profile": dataclasses.asdict(dataset_profile),
            "hardware_profile": dataclasses.asdict(hardware_profile),
            "resource_plan": dataclasses.asdict(resource_plan),
        }
        (root / "meta.json").write_text(
            json.dumps(metadata, indent=2, ensure_ascii=False, default=_json_default),
            encoding="utf-8",
        )

        files = {
            file.relative_to(root).as_posix(): {
                "sha256": _sha256(file),
                "size": file.stat().st_size,
            }
            for file in root.rglob("*")
            if file.is_file()
        }
        manifest = {
            "bundle_format_version": BUNDLE_FORMAT_VERSION,
            "integrity_scope": "accidental corruption detection; not authenticity",
            "files": files,
        }
        (root / "manifest.json").write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
        )

        temporary_output = output_path.with_suffix(output_path.suffix + ".tmp")
        try:
            with zipfile.ZipFile(temporary_output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
                for file in sorted(root.rglob("*")):
                    if file.is_file():
                        archive.write(file, arcname=file.relative_to(root).as_posix())
            os.replace(temporary_output, output_path)
        finally:
            temporary_output.unlink(missing_ok=True)
    return str(output_path)


def load_bundle(path: str | Path, *, trusted: bool = False) -> dict:
    path = Path(path)
    if not path.exists():
        raise SmartTabError(f"No SmartTab bundle found at {path}")
    if not trusted:
        raise SmartTabError(
            "SmartTab bundles contain joblib/pickle objects that may execute code while loading. "
            "Load only a bundle you created or obtained from a trusted source, then call "
            "smarttab.load(path, trusted=True)."
        )
    if not zipfile.is_zipfile(path):
        raise SmartTabError(f"{path} is not a valid .smarttab ZIP bundle")

    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        with zipfile.ZipFile(path, "r") as archive:
            _validate_archive_members(archive)
            _safe_extract(archive, root)
        manifest_path = root / "manifest.json"
        metadata_path = root / "meta.json"
        if not manifest_path.exists() or not metadata_path.exists():
            raise SmartTabError("invalid SmartTab bundle: manifest.json or meta.json is missing")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        _verify_manifest(root, manifest)
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        format_version = int(metadata.get("bundle_format_version", 0))
        if format_version != BUNDLE_FORMAT_VERSION:
            raise SmartTabError(
                f"unsupported bundle format {format_version}; this SmartTab version supports "
                f"format {BUNDLE_FORMAT_VERSION}. Re-save the model with a compatible version."
            )

        task_type = TaskType(metadata["task_type"])
        records = metadata.get("model_records") or []
        if not records:
            raise SmartTabError("invalid SmartTab bundle: no model records")
        loaded_models: list[tuple[str, str, object, list[str] | None]] = []
        for record in records:
            filename = _validated_relative_path(record["file"])
            loaded_models.append(
                (
                    record["alias"],
                    record["model_name"],
                    _load_single_model(record["model_name"], task_type, root / filename),
                    record.get("feature_columns"),
                )
            )

        ensemble_info = metadata.get("ensemble_info")
        model_name = metadata["model_name"]
        if ensemble_info is not None or model_name in {"voting", "stacking"}:
            from smarttab.training.ensemble import StackingEnsemble, VotingEnsemble

            strategy = (ensemble_info or {}).get("strategy", model_name)
            if strategy == "stacking":
                meta_path = root / "meta_model.joblib"
                if not meta_path.exists():
                    raise SmartTabError("invalid stacking bundle: meta_model.joblib is missing")
                meta_model = joblib.load(meta_path)
                estimator = StackingEnsemble(
                    loaded_models,
                    task_type,
                    meta_model,
                    (ensemble_info or {}).get("meta_model_name", "linear"),
                )
            elif strategy == "voting":
                estimator = VotingEnsemble(
                    loaded_models,
                    task_type,
                    weights=metadata.get("voting_weights"),
                )
            else:
                raise SmartTabError(f"invalid ensemble strategy {strategy!r} in bundle")
        else:
            estimator = loaded_models[0][2]

        pipeline_path = root / "pipeline.joblib"
        if not pipeline_path.exists():
            raise SmartTabError("invalid SmartTab bundle: pipeline.joblib is missing")
        cleaning_pipeline = joblib.load(pipeline_path)
        target_path = root / "target_encoder.joblib"
        target_encoder = joblib.load(target_path) if target_path.exists() else None
        diagnostics_path = root / "diagnostics.joblib"
        diagnostics = joblib.load(diagnostics_path) if diagnostics_path.exists() else {}

        return {
            "model_name": model_name,
            "task_type": task_type,
            "estimator": estimator,
            "cleaning_pipeline": cleaning_pipeline,
            "target_encoder": target_encoder,
            "raw_feature_names": metadata.get(
                "raw_feature_names", getattr(cleaning_pipeline, "raw_feature_columns_", [])
            ),
            "feature_names": metadata["feature_names"],
            "cat_features": metadata["cat_features"],
            "best_params": metadata["best_params"],
            "primary_metric": metadata["primary_metric"],
            "metrics": metadata["metrics"],
            "class_labels": metadata.get("class_labels"),
            "ensemble_info": ensemble_info,
            "decision_threshold": metadata.get("decision_threshold", DEFAULT_THRESHOLD),
            "reject_threshold": metadata.get("reject_threshold", DEFAULT_REJECT_THRESHOLD),
            "per_label_thresholds": metadata.get("per_label_thresholds"),
            "objective": metadata.get("objective", DEFAULT_OBJECTIVE),
            "timings": metadata.get("timings", {}),
            "notes": metadata.get("notes", []),
            "static_charts": metadata.get("static_charts", "auto"),
            "probability_calibrator": diagnostics.get("probability_calibrator"),
            "conformal_predictor": diagnostics.get("conformal_predictor"),
            "ood_detector": diagnostics.get("ood_detector"),
            "drift_reference": diagnostics.get("drift_reference"),
            "data_science_config": metadata.get("data_science_config", {}),
            "data_quality_report": metadata.get("data_quality_report", {}),
            "modality_dropout_info": metadata.get("modality_dropout_info", {}),
            "dataset_profile": _dataset_profile_from_dict(metadata["dataset_profile"]),
            "hardware_profile": _hardware_profile_from_dict(metadata["hardware_profile"]),
            "resource_plan": ResourcePlan(**metadata["resource_plan"]),
            "created_with": metadata.get("created_with", {}),
        }


def _save_single_model(model_name: str, task_type: TaskType, estimator, base_path: Path) -> Path:
    if task_type in {TaskType.MULTILABEL, TaskType.MULTIOUTPUT_REGRESSION}:
        path = Path(str(base_path) + ".joblib")
        joblib.dump(estimator, path, compress=3)
        return path
    if model_name == "catboost":
        path = Path(str(base_path) + ".cbm")
        estimator.save_model(str(path))
        return path
    if model_name in {"lightgbm", "xgboost"}:
        path = Path(str(base_path) + ".joblib")
        joblib.dump(estimator, path, compress=3)
        return path
    raise SmartTabError(f"Unknown model_name {model_name!r}")


def _load_single_model(model_name: str, task_type: TaskType, path: Path):
    if not path.exists():
        raise SmartTabError(f"model payload is missing: {path.name}")
    if task_type in {TaskType.MULTILABEL, TaskType.MULTIOUTPUT_REGRESSION}:
        return joblib.load(path)
    if model_name == "catboost":
        from catboost import CatBoostClassifier, CatBoostRanker, CatBoostRegressor

        if task_type is TaskType.RANKING:
            estimator = CatBoostRanker()
        elif task_type in {TaskType.REGRESSION, TaskType.MULTIOUTPUT_REGRESSION}:
            estimator = CatBoostRegressor()
        else:
            estimator = CatBoostClassifier()
        estimator.load_model(str(path))
        return estimator
    if model_name in {"lightgbm", "xgboost"}:
        return joblib.load(path)
    raise SmartTabError(f"Unknown model_name {model_name!r} in bundle")


def _validate_archive_members(archive: zipfile.ZipFile) -> None:
    members = archive.infolist()
    if len(members) > MAX_ARCHIVE_FILES:
        raise SmartTabError(f"bundle contains too many archive members ({len(members)})")
    total = 0
    for info in members:
        _validated_relative_path(info.filename)
        mode = (info.external_attr >> 16) & 0o170000
        if mode == stat.S_IFLNK:
            raise SmartTabError(f"bundle contains a symbolic link: {info.filename}")
        if info.is_dir():
            continue
        total += int(info.file_size)
        if total > MAX_UNCOMPRESSED_BYTES:
            raise SmartTabError("bundle exceeds the maximum uncompressed size")
        if info.file_size > 10 * 1024**2:
            ratio = info.file_size / max(info.compress_size, 1)
            if ratio > MAX_COMPRESSION_RATIO:
                raise SmartTabError(f"suspicious compression ratio for {info.filename}")
        if not _is_allowed_member(info.filename):
            raise SmartTabError(f"unexpected file in SmartTab bundle: {info.filename}")


def _safe_extract(archive: zipfile.ZipFile, root: Path) -> None:
    for info in archive.infolist():
        relative = _validated_relative_path(info.filename)
        destination = root / relative
        if info.is_dir():
            destination.mkdir(parents=True, exist_ok=True)
            continue
        destination.parent.mkdir(parents=True, exist_ok=True)
        with archive.open(info, "r") as source, destination.open("wb") as target:
            shutil.copyfileobj(source, target, length=HASH_CHUNK_BYTES)


def _verify_manifest(root: Path, manifest: dict) -> None:
    if int(manifest.get("bundle_format_version", 0)) != BUNDLE_FORMAT_VERSION:
        raise SmartTabError("bundle manifest version does not match this SmartTab release")
    records = manifest.get("files")
    if not isinstance(records, dict) or not records:
        raise SmartTabError("bundle manifest contains no file records")
    actual = {
        file.relative_to(root).as_posix()
        for file in root.rglob("*")
        if file.is_file() and file.name != "manifest.json"
    }
    expected = set(records)
    if actual != expected:
        missing = sorted(expected - actual)
        unexpected = sorted(actual - expected)
        raise SmartTabError(f"bundle file set mismatch; missing={missing}, unexpected={unexpected}")
    for relative, record in records.items():
        path = root / _validated_relative_path(relative)
        if path.stat().st_size != int(record["size"]):
            raise SmartTabError(f"bundle integrity check failed for {relative}: size mismatch")
        if _sha256(path) != record["sha256"]:
            raise SmartTabError(f"bundle integrity check failed for {relative}: SHA-256 mismatch")


def _validated_relative_path(value: str) -> Path:
    pure = PurePosixPath(value)
    if pure.is_absolute() or not pure.parts or any(part in {"", ".", ".."} for part in pure.parts):
        raise SmartTabError(f"unsafe path in SmartTab bundle: {value!r}")
    if "\\" in value or ":" in pure.parts[0]:
        raise SmartTabError(f"unsafe path in SmartTab bundle: {value!r}")
    return Path(*pure.parts)


def _is_allowed_member(value: str) -> bool:
    if value in {
        "manifest.json",
        "meta.json",
        "pipeline.joblib",
        "target_encoder.joblib",
        "meta_model.joblib",
        "diagnostics.joblib",
        "model.cbm",
        "model.joblib",
    }:
        return True
    path = PurePosixPath(value)
    return (
        len(path.parts) == 2
        and path.parts[0] == "base_models"
        and path.suffix in {".cbm", ".joblib"}
    )


def _safe_component(value: str) -> str:
    if not value or any(character not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for character in value):
        raise SmartTabError(f"unsafe model alias {value!r}")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(HASH_CHUNK_BYTES), b""):
            digest.update(block)
    return digest.hexdigest()


def _runtime_manifest() -> dict[str, Any]:
    packages = {}
    for name in ("numpy", "pandas", "scikit-learn", "catboost", "lightgbm", "xgboost", "optuna", "joblib"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            continue
    return {
        "python": sys.version.split()[0],
        "implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "packages": packages,
    }


def _json_default(value: Any):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _dataset_profile_from_dict(data: dict) -> DatasetProfile:
    payload = dict(data)
    payload["task_type"] = TaskType(payload["task_type"])
    payload["high_correlation_pairs"] = [tuple(pair) for pair in payload.get("high_correlation_pairs", [])]
    return DatasetProfile(**payload)


def _hardware_profile_from_dict(data: dict) -> HardwareProfile:
    return HardwareProfile(
        cpu=CPUInfo(**data["cpu"]),
        ram=RAMInfo(**data["ram"]),
        gpu=GPUInfo(**data["gpu"]),
        disk=DiskInfo(**data["disk"]),
    )
