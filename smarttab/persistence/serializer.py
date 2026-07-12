"""save()/load() — a self-contained ``.smarttab`` zip bundle.

CatBoost is saved through its native ``.cbm`` format, which is stable across
library versions. LightGBM's and XGBoost's sklearn wrappers are
joblib-serialized instead; this is the standard way to persist them but is
coupled to the library version used at save time — pin those versions if
this matters for your deployment. Everything else (cleaning pipeline,
target encoder, profiles, metadata) is plain, version-agnostic joblib/JSON.

Voting/stacking ensembles (``ensemble_info is not None``) are saved as their
individual base models under ``base_models/`` plus, for stacking, the
sklearn meta-learner — then reassembled into a
``training.ensemble.VotingEnsemble``/``StackingEnsemble`` on load.
"""

from __future__ import annotations

import dataclasses
import json
import tempfile
import zipfile
from pathlib import Path

import joblib

from smarttab import __version__
from smarttab.analysis.dataset_analyzer import DatasetProfile, TaskType
from smarttab.exceptions import SmartTabError
from smarttab.hardware.profiler import CPUInfo, DiskInfo, GPUInfo, HardwareProfile, RAMInfo
from smarttab.hardware.resource_planner import ResourcePlan
from smarttab.optimization.threshold import DEFAULT_OBJECTIVE, DEFAULT_REJECT_THRESHOLD, DEFAULT_THRESHOLD

BUNDLE_FORMAT_VERSION = 2


def save_bundle(path: str, *, model_name: str, task_type: TaskType, estimator, cleaning_pipeline,
                 target_encoder, feature_names: list[str], cat_features: list[str],
                 dataset_profile: DatasetProfile, hardware_profile: HardwareProfile,
                 resource_plan: ResourcePlan, best_params: dict, primary_metric: str,
                 metrics: dict, class_labels: list | None, ensemble_info: dict | None = None,
                 decision_threshold: float = DEFAULT_THRESHOLD, reject_threshold: float = DEFAULT_REJECT_THRESHOLD,
                 per_label_thresholds: list[float] | None = None, objective: str = DEFAULT_OBJECTIVE,
                 multi_threshold_ensemble: bool = False, threshold_ladder: list[dict] | None = None) -> str:
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)

        voting_weights = None
        if ensemble_info is not None:
            base_dir = tmp_path / "base_models"
            base_dir.mkdir()
            for base_name, base_estimator in estimator.base_models:
                _save_single_model(base_name, task_type, base_estimator, base_dir / base_name)
            if ensemble_info["strategy"] == "stacking":
                joblib.dump(estimator.meta_model, tmp_path / "meta_model.joblib")
            elif getattr(estimator, "weights", None) is not None:
                voting_weights = [float(w) for w in estimator.weights]
        else:
            _save_single_model(model_name, task_type, estimator, tmp_path / "model")

        joblib.dump(cleaning_pipeline, tmp_path / "pipeline.joblib")
        if target_encoder is not None:
            joblib.dump(target_encoder, tmp_path / "target_encoder.joblib")

        meta = {
            "bundle_format_version": BUNDLE_FORMAT_VERSION,
            "smarttab_version": __version__,
            "model_name": model_name,
            "task_type": task_type.value,
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
            "multi_threshold_ensemble": multi_threshold_ensemble,
            "threshold_ladder": threshold_ladder,
            "dataset_profile": dataclasses.asdict(dataset_profile),
            "hardware_profile": dataclasses.asdict(hardware_profile),
            "resource_plan": dataclasses.asdict(resource_plan),
        }
        (tmp_path / "meta.json").write_text(json.dumps(meta, indent=2, default=str), encoding="utf-8")

        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(output_path, "w", zipfile.ZIP_DEFLATED) as zf:
            for file in tmp_path.rglob("*"):
                if file.is_file():
                    zf.write(file, arcname=file.relative_to(tmp_path))

    return str(output_path)


def load_bundle(path: str) -> dict:
    path = Path(path)
    if not path.exists():
        raise SmartTabError(f"No SmartTab bundle found at {path}")

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        with zipfile.ZipFile(path, "r") as zf:
            zf.extractall(tmp_path)

        meta = json.loads((tmp_path / "meta.json").read_text(encoding="utf-8"))
        model_name = meta["model_name"]
        task_type = TaskType(meta["task_type"])
        ensemble_info = meta.get("ensemble_info")

        if ensemble_info is not None:
            from smarttab.training.ensemble import StackingEnsemble, VotingEnsemble

            base_dir = tmp_path / "base_models"
            base_models = [
                (base_name, _load_single_model(base_name, task_type, base_dir / base_name))
                for base_name in ensemble_info["base_params"].keys()
            ]
            if ensemble_info["strategy"] == "stacking":
                meta_model = joblib.load(tmp_path / "meta_model.joblib")
                estimator = StackingEnsemble(base_models, task_type, meta_model)
            else:
                estimator = VotingEnsemble(base_models, task_type, weights=meta.get("voting_weights"))
        else:
            estimator = _load_single_model(model_name, task_type, tmp_path / "model")

        cleaning_pipeline = joblib.load(tmp_path / "pipeline.joblib")
        target_encoder_path = tmp_path / "target_encoder.joblib"
        target_encoder = joblib.load(target_encoder_path) if target_encoder_path.exists() else None

        dataset_profile = _dataset_profile_from_dict(meta["dataset_profile"])
        hardware_profile = _hardware_profile_from_dict(meta["hardware_profile"])
        resource_plan = ResourcePlan(**meta["resource_plan"])

        return {
            "model_name": model_name,
            "task_type": task_type,
            "estimator": estimator,
            "cleaning_pipeline": cleaning_pipeline,
            "target_encoder": target_encoder,
            "feature_names": meta["feature_names"],
            "cat_features": meta["cat_features"],
            "best_params": meta["best_params"],
            "primary_metric": meta["primary_metric"],
            "metrics": meta["metrics"],
            "class_labels": meta["class_labels"],
            "ensemble_info": ensemble_info,
            "decision_threshold": meta.get("decision_threshold", DEFAULT_THRESHOLD),
            "reject_threshold": meta.get("reject_threshold", DEFAULT_REJECT_THRESHOLD),
            "per_label_thresholds": meta.get("per_label_thresholds"),
            "objective": meta.get("objective", DEFAULT_OBJECTIVE),
            "multi_threshold_ensemble": meta.get("multi_threshold_ensemble", False),
            "threshold_ladder": meta.get("threshold_ladder"),
            "dataset_profile": dataset_profile,
            "hardware_profile": hardware_profile,
            "resource_plan": resource_plan,
        }


def _save_single_model(model_name: str, task_type: TaskType, estimator, base_path: Path) -> None:
    if task_type in (TaskType.MULTILABEL, TaskType.MULTIOUTPUT_REGRESSION):
        # a sklearn MultiOutputClassifier/Regressor wrapper, not a raw catboost/lightgbm object —
        # no native save_model() to use regardless of model_name.
        joblib.dump(estimator, str(base_path) + ".joblib")
        return
    if model_name == "catboost":
        estimator.save_model(str(base_path) + ".cbm")
    elif model_name in ("lightgbm", "xgboost"):
        joblib.dump(estimator, str(base_path) + ".joblib")
    else:
        raise SmartTabError(f"Unknown model_name {model_name!r}")


def _load_single_model(model_name: str, task_type: TaskType, base_path: Path):
    if task_type in (TaskType.MULTILABEL, TaskType.MULTIOUTPUT_REGRESSION):
        return joblib.load(str(base_path) + ".joblib")
    if model_name == "catboost":
        from catboost import CatBoostClassifier, CatBoostRanker, CatBoostRegressor

        if task_type is TaskType.RANKING:
            cls = CatBoostRanker
        elif task_type is TaskType.REGRESSION:
            cls = CatBoostRegressor
        else:
            cls = CatBoostClassifier
        estimator = cls()
        estimator.load_model(str(base_path) + ".cbm")
        return estimator
    if model_name in ("lightgbm", "xgboost"):
        return joblib.load(str(base_path) + ".joblib")
    raise SmartTabError(f"Unknown model_name {model_name!r} in bundle metadata")


def _dataset_profile_from_dict(data: dict) -> DatasetProfile:
    data = dict(data)
    data["task_type"] = TaskType(data["task_type"])
    data["high_correlation_pairs"] = [tuple(p) for p in data.get("high_correlation_pairs", [])]
    return DatasetProfile(**data)


def _hardware_profile_from_dict(data: dict) -> HardwareProfile:
    return HardwareProfile(
        cpu=CPUInfo(**data["cpu"]),
        ram=RAMInfo(**data["ram"]),
        gpu=GPUInfo(**data["gpu"]),
        disk=DiskInfo(**data["disk"]),
    )
