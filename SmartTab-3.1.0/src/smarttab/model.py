"""The fitted model object returned by :func:`smarttab.fit` and :func:`smarttab.load`."""

from __future__ import annotations

import dataclasses
import pickle
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from smarttab.analysis.dataset_analyzer import DatasetProfile, TaskType
from smarttab.cleaning.encoders import MultiLabelTargetEncoder, TargetLabelEncoder
from smarttab.cleaning.pipeline import SmartCleaningPipeline
from smarttab.config import load_data
from smarttab.datascience.drift import compare_drift
from smarttab.datascience.quality import audit_data_quality
from smarttab.evaluation.evaluator import (
    evaluate_classification,
    evaluate_multilabel,
    evaluate_multioutput_regression,
    evaluate_ranking,
    evaluate_regression,
)
from smarttab.exceptions import DataValidationError, SmartTabError
from smarttab.hardware.profiler import HardwareProfile
from smarttab.hardware.resource_planner import ResourcePlan
from smarttab.optimization.threshold import (
    DEFAULT_OBJECTIVE,
    DEFAULT_REJECT_THRESHOLD,
    DEFAULT_THRESHOLD,
    apply_threshold,
)
from smarttab.persistence.serializer import save_bundle
from smarttab.prediction_result import PredictionArray
from smarttab.reporting.report_generator import ReportContext, generate_report
from smarttab.training.trainer import predict, predict_proba


class SmartTabModel:
    """A complete preprocessing + model + metadata bundle.

    Input validation is performed against the raw training schema before the
    fitted cleaning pipeline is replayed. The fitted model therefore never
    silently substitutes a zero-filled synthetic row for malformed input.
    """

    def __init__(
        self,
        *,
        model_name: str,
        task_type: TaskType,
        estimator: Any,
        cleaning_pipeline: SmartCleaningPipeline,
        target_encoder: TargetLabelEncoder | MultiLabelTargetEncoder | None,
        raw_feature_names: list[str],
        feature_names: list[str],
        cat_features: list[str],
        best_params: dict,
        primary_metric: str,
        metrics: dict,
        feature_importance: pd.DataFrame,
        dataset_profile: DatasetProfile,
        hardware_profile: HardwareProfile,
        resource_plan: ResourcePlan,
        class_labels: list | dict | None,
        timings: dict,
        notes: list[str],
        shap_importance: pd.DataFrame | None = None,
        ensemble_info: dict | None = None,
        decision_threshold: float = DEFAULT_THRESHOLD,
        reject_threshold: float = DEFAULT_REJECT_THRESHOLD,
        per_label_thresholds: list[float] | None = None,
        objective: str = DEFAULT_OBJECTIVE,
        static_charts: str | bool = "auto",
        probability_calibrator: Any = None,
        conformal_predictor: Any = None,
        ood_detector: Any = None,
        drift_reference: Any = None,
        data_science_config: dict | None = None,
        data_quality_report: dict | None = None,
        modality_dropout_info: dict | None = None,
    ) -> None:
        self.model_name = model_name
        self.task_type = task_type
        self.estimator = estimator
        self.cleaning_pipeline = cleaning_pipeline
        self.target_encoder = target_encoder
        self.raw_feature_names = list(raw_feature_names)
        self.feature_names = list(feature_names)
        self.cat_features = list(cat_features)
        self.best_params = dict(best_params)
        self.primary_metric = primary_metric
        self.metrics = dict(metrics)
        self.feature_importance = feature_importance
        self.dataset_profile = dataset_profile
        self.hardware_profile = hardware_profile
        self.resource_plan = resource_plan
        self.class_labels = class_labels
        self.timings = dict(timings)
        self.notes = list(notes)
        self.shap_importance = shap_importance
        self.ensemble_info = ensemble_info
        self.decision_threshold = float(decision_threshold)
        self.reject_threshold = float(reject_threshold)
        self.per_label_thresholds = per_label_thresholds
        self.objective = objective
        self.static_charts = static_charts
        self.probability_calibrator = probability_calibrator
        self.conformal_predictor = conformal_predictor
        self.ood_detector = ood_detector
        self.drift_reference = drift_reference
        self.data_science_config = dict(data_science_config or {})
        self.data_quality_report = dict(data_quality_report or {})
        self.modality_dropout_info = dict(modality_dropout_info or {})
        self._last_eval: dict | None = None
        self._report_folder: str | None = None

    def _clean(self, frame: pd.DataFrame) -> pd.DataFrame:
        return self.cleaning_pipeline.transform(frame)

    def _single_raw_modality(self) -> tuple[str, str] | None:
        modalities = getattr(self.cleaning_pipeline, "raw_column_modalities_", {})
        if len(self.raw_feature_names) == 1 and self.raw_feature_names[0] in modalities:
            column = self.raw_feature_names[0]
            return column, modalities[column]
        return None

    def _coerce_predict_input(self, X: Any) -> tuple[pd.DataFrame, bool]:
        """Normalize tabular or direct raw-modality prediction input."""
        direct = self._single_raw_modality()
        if isinstance(X, pd.DataFrame):
            if len(X) == 0:
                raise DataValidationError("prediction input contains no rows")
            return X.copy(), False
        if isinstance(X, (str, Path)):
            if direct is not None:
                return pd.DataFrame({direct[0]: [X]}), True
            frame = load_data(X)
            if frame.empty:
                raise DataValidationError("prediction file contains no rows")
            return frame, False
        if isinstance(X, dict):
            if direct is not None and direct[0] not in X:
                return pd.DataFrame({direct[0]: [X]}), True
            if not X:
                raise DataValidationError(
                    f"single-row prediction input is empty; expected columns {self.raw_feature_names}"
                )
            return pd.DataFrame([X]), True
        if isinstance(X, np.ndarray):
            array = np.asarray(X)
            if direct is not None:
                column, modality = direct
                if modality == "image":
                    if array.ndim in (2, 3):
                        return pd.DataFrame({column: [array]}), True
                    if array.ndim == 4:
                        return pd.DataFrame({column: list(array)}), False
                elif modality == "video" and array.ndim == 4:
                    return pd.DataFrame({column: [array]}), True
                elif modality == "audio":
                    if array.ndim == 1:
                        return pd.DataFrame({column: [array]}), True
                    if array.ndim == 2:
                        return pd.DataFrame({column: list(array)}), False
            if array.ndim == 1:
                if len(array) != len(self.raw_feature_names):
                    raise DataValidationError(
                        f"1-D prediction array has {len(array)} values; expected "
                        f"{len(self.raw_feature_names)} in raw feature order {self.raw_feature_names}"
                    )
                return pd.DataFrame([array], columns=self.raw_feature_names).infer_objects(), True
            if array.ndim == 2:
                if array.shape[0] == 0:
                    raise DataValidationError("prediction array contains no rows")
                if array.shape[1] != len(self.raw_feature_names):
                    raise DataValidationError(
                        f"prediction array has {array.shape[1]} columns; expected "
                        f"{len(self.raw_feature_names)} in raw feature order {self.raw_feature_names}"
                    )
                return pd.DataFrame(array, columns=self.raw_feature_names).infer_objects(), False
            raise DataValidationError("prediction arrays have an unsupported shape for this model")
        if isinstance(X, (list, tuple)):
            if not X:
                raise DataValidationError("prediction input contains no rows")
            if all(isinstance(row, dict) for row in X):
                return pd.DataFrame(list(X)), False
            if direct is not None:
                column, modality = direct
                if modality == "audio" and len(X) == 2 and isinstance(X[0], (int, float)):
                    return pd.DataFrame({column: [X]}), True
                if modality == "video" and len(X) == 2 and isinstance(X[0], (int, float)):
                    return pd.DataFrame({column: [X]}), True
                if modality == "video" and isinstance(X[0], np.ndarray) and X[0].ndim in (2, 3):
                    return pd.DataFrame({column: [X]}), True
                return pd.DataFrame({column: list(X)}), False
            raise DataValidationError(
                "list/tuple prediction input must be a batch of dictionaries; "
                "direct raw lists are supported only by single-modality models"
            )
        if direct is not None:
            return pd.DataFrame({direct[0]: [X]}), True
        raise DataValidationError(
            "unsupported prediction input type; expected DataFrame, file path, dict, "
            "list of dicts, NumPy array, or a direct raw-modality value"
        )

    def transform_features(self, X: Any) -> pd.DataFrame:
        """Return the exact bounded feature matrix consumed by CatBoost/LightGBM."""
        frame, _ = self._coerce_predict_input(X)
        return self._clean(frame)

    @property
    def feature_space(self) -> dict[str, Any]:
        """Describe the fitted raw modalities and generated feature budget."""
        multimodal = getattr(self.cleaning_pipeline, "multimodal_pipeline_", None)
        if multimodal is None:
            return {"modalities": {}, "generated_features": 0, "total_model_features": len(self.feature_names)}
        return {
            "modalities": dict(multimodal.column_modalities),
            "generated_features": len(multimodal.selected_columns_),
            "feature_budget": multimodal.config.total_features,
            "allocated_features": dict(multimodal.report_.allocated_features),
            "generated_by_column": dict(multimodal.report_.generated_features),
            "backend_requested": multimodal.config.backend,
            "backend_used": dict(multimodal.report_.backend_used),
            "supervised_adaptation": multimodal.config.supervised_adaptation,
            "adapted_features": dict(multimodal.report_.adapted_features),
            "adapter_diagnostics": dict(multimodal.report_.adapter_diagnostics),
            "speed_accuracy": multimodal.config.speed_accuracy,
            "batch_size": multimodal.config.batch_size,
            "feature_workers": multimodal.config.workers,
            "feature_groups": {
                name: len(columns)
                for name, columns in getattr(self.cleaning_pipeline, "feature_groups", {}).items()
            },
            "fusion_strategy": (self.ensemble_info or {}).get("fusion_strategy", "early"),
            "total_model_features": len(self.feature_names),
            "errors": dict(multimodal.report_.errors),
            "notes": list(multimodal.report_.notes),
        }

    def _probabilities_clean(self, X_clean: pd.DataFrame) -> np.ndarray:
        probabilities = np.asarray(predict_proba(self.estimator, self.model_name, X_clean))
        if self.probability_calibrator is not None:
            probabilities = self.probability_calibrator.transform(probabilities)
        return probabilities

    def _predict_encoded(self, X_clean: pd.DataFrame) -> np.ndarray:
        if self.task_type is TaskType.BINARY:
            proba = self._probabilities_clean(X_clean)
            return apply_threshold(proba[:, 1], self.decision_threshold)
        if self.task_type is TaskType.MULTILABEL:
            proba = self._probabilities_clean(X_clean)
            thresholds = self.per_label_thresholds or [DEFAULT_THRESHOLD] * proba.shape[1]
            if len(thresholds) != proba.shape[1]:
                raise SmartTabError("stored multilabel threshold count does not match model outputs")
            return np.column_stack(
                [apply_threshold(proba[:, index], threshold) for index, threshold in enumerate(thresholds)]
            )
        if self.task_type is TaskType.MULTICLASS:
            return self._probabilities_clean(X_clean).argmax(axis=1)
        return np.asarray(predict(self.estimator, self.model_name, X_clean))

    def predict(self, X: Any) -> PredictionArray:
        """Predict labels or numeric outputs after strict raw-schema validation."""
        frame, is_single = self._coerce_predict_input(X)
        encoded = self._predict_encoded(self._clean(frame))
        if self.task_type in {TaskType.BINARY, TaskType.MULTICLASS}:
            if not isinstance(self.target_encoder, TargetLabelEncoder):
                raise SmartTabError("classification target encoder is unavailable")
            encoded = self.target_encoder.inverse_transform(encoded)
        elif self.task_type is TaskType.MULTILABEL:
            if not isinstance(self.target_encoder, MultiLabelTargetEncoder):
                raise SmartTabError("multilabel target encoder is unavailable")
            encoded = self.target_encoder.inverse_transform(encoded)
        return PredictionArray(encoded, single=is_single)

    def predict_proba(self, X: Any) -> np.ndarray:
        if not self.task_type.is_classification:
            raise SmartTabError("predict_proba() is available only for classification tasks")
        frame, is_single = self._coerce_predict_input(X)
        probabilities = self._probabilities_clean(self._clean(frame))
        return probabilities[0] if is_single else probabilities

    def predict_interval(self, X: Any) -> tuple[np.ndarray, np.ndarray]:
        """Return conformal prediction intervals for regression models."""
        if self.conformal_predictor is None or self.task_type not in {
            TaskType.REGRESSION, TaskType.MULTIOUTPUT_REGRESSION
        }:
            raise SmartTabError("conformal regression intervals are unavailable for this model")
        frame, is_single = self._coerce_predict_input(X)
        values = np.asarray(predict(self.estimator, self.model_name, self._clean(frame)))
        lower, upper = self.conformal_predictor.interval(values)
        if is_single:
            return np.asarray(lower)[0], np.asarray(upper)[0]
        return lower, upper

    def predict_set(self, X: Any):
        """Return conformal class sets instead of forcing one overconfident class."""
        if self.conformal_predictor is None or self.task_type not in {
            TaskType.BINARY, TaskType.MULTICLASS, TaskType.MULTILABEL
        }:
            raise SmartTabError("conformal prediction sets are unavailable for this model")
        probabilities = np.asarray(self.predict_proba(X))
        if probabilities.ndim == 1:
            probabilities = probabilities.reshape(1, -1)
        if self.task_type is TaskType.MULTILABEL:
            return self.conformal_predictor.multilabel_sets(probabilities)
        masks = self.conformal_predictor.prediction_set(probabilities)
        labels = (
            np.asarray(self.target_encoder.classes_)
            if isinstance(self.target_encoder, TargetLabelEncoder) else np.arange(masks.shape[1])
        )
        return [labels[row].tolist() for row in masks]

    def ood_score(self, X: Any) -> np.ndarray | float:
        """Score how far inputs are from the transformed training distribution."""
        if self.ood_detector is None:
            raise SmartTabError("OOD detection was disabled or unavailable during fitting")
        frame, is_single = self._coerce_predict_input(X)
        scores = np.asarray(self.ood_detector.score(self._clean(frame)), dtype=float)
        return float(scores[0]) if is_single else scores

    def predict_with_uncertainty(self, X: Any) -> dict[str, Any]:
        """Return predictions, calibrated probabilities/intervals, and OOD scores."""
        result: dict[str, Any] = {"prediction": np.asarray(self.predict(X)).tolist()}
        if self.task_type.is_classification:
            result["probabilities"] = np.asarray(self.predict_proba(X)).tolist()
            if self.conformal_predictor is not None:
                result["prediction_set"] = self.predict_set(X)
        elif self.conformal_predictor is not None:
            lower, upper = self.predict_interval(X)
            result["interval_lower"] = np.asarray(lower).tolist()
            result["interval_upper"] = np.asarray(upper).tolist()
        if self.ood_detector is not None:
            result["ood_score"] = np.asarray(self.ood_score(X)).tolist()
            result["ood_threshold"] = float(self.ood_detector.score_threshold_)
        return result

    def data_quality(self, X: Any, y: Any = None) -> dict:
        """Audit new raw data with the same schema and modality declarations."""
        frame, _ = self._coerce_predict_input(X)
        targets: list[str] = []
        if y is not None:
            frame = frame.reset_index(drop=True).copy()
            values = np.asarray(y)
            if values.ndim == 1:
                frame["__audit_target__"] = values
                targets = ["__audit_target__"]
            else:
                targets = []
                for index in range(values.shape[1]):
                    name = f"__audit_target_{index}__"
                    frame[name] = values[:, index]
                    targets.append(name)
        report = audit_data_quality(
            frame,
            target_columns=targets,
            column_modalities=getattr(self.cleaning_pipeline, "raw_column_modalities_", {}),
        )
        return report.to_dict()

    def drift_report(self, X: Any) -> dict:
        """Compare raw and transformed inference data with the training reference."""
        if self.drift_reference is None:
            raise SmartTabError("drift monitoring was disabled during fitting")
        frame, _ = self._coerce_predict_input(X)
        transformed = self._clean(frame)
        return compare_drift(self.drift_reference, frame, transformed)

    def _uncertainty_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_proba: np.ndarray | None,
        clean: pd.DataFrame,
    ) -> dict[str, float]:
        results: dict[str, float] = {}
        if self.ood_detector is not None:
            scores = np.asarray(self.ood_detector.score(clean), dtype=float)
            results.update({
                "ood_score_mean": float(scores.mean()),
                "ood_score_p95": float(np.quantile(scores, 0.95)),
                "ood_flag_rate": float((scores >= self.ood_detector.score_threshold_).mean()),
            })
        if self.conformal_predictor is None:
            return results
        truth = np.asarray(y_true)
        if self.task_type in {TaskType.BINARY, TaskType.MULTICLASS} and y_proba is not None:
            sets = self.conformal_predictor.prediction_set(y_proba)
            results["conformal_coverage"] = float(
                sets[np.arange(len(truth)), truth.astype(int)].mean()
            )
            results["conformal_set_size_mean"] = float(sets.sum(axis=1).mean())
        elif self.task_type is TaskType.MULTILABEL and y_proba is not None:
            sets = self.conformal_predictor.multilabel_sets(y_proba)
            covered = np.where(truth.astype(int) == 1, sets["positive_possible"], sets["negative_possible"])
            results["conformal_label_coverage"] = float(covered.mean())
            results["conformal_ambiguous_rate"] = float(
                (sets["positive_possible"] & sets["negative_possible"]).mean()
            )
        elif self.task_type in {TaskType.REGRESSION, TaskType.MULTIOUTPUT_REGRESSION}:
            lower, upper = self.conformal_predictor.interval(np.asarray(y_pred))
            results["conformal_coverage"] = float(np.mean((truth >= lower) & (truth <= upper)))
            results["conformal_interval_width_mean"] = float(np.mean(upper - lower))
        return results

    def evaluate(self, X: Any, y: Any, groups: Any = None) -> dict:
        frame, _ = self._coerce_predict_input(X)
        clean = self._clean(frame)
        started = time.perf_counter()
        y_pred = self._predict_encoded(clean)
        prediction_seconds = time.perf_counter() - started
        groups_array = np.asarray(groups) if groups is not None else None

        if self.task_type in {TaskType.BINARY, TaskType.MULTICLASS}:
            if not isinstance(self.target_encoder, TargetLabelEncoder):
                raise SmartTabError("classification target encoder is unavailable")
            y_true = self.target_encoder.transform(pd.Series(y).reset_index(drop=True))
            y_proba = self._probabilities_clean(clean)
            metrics = evaluate_classification(y_true, y_pred, y_proba)
        elif self.task_type is TaskType.MULTILABEL:
            if isinstance(self.target_encoder, MultiLabelTargetEncoder):
                if isinstance(y, pd.DataFrame):
                    target_frame = y.reset_index(drop=True)
                else:
                    array = np.asarray(y)
                    if array.ndim != 2 or array.shape[1] != len(self.target_encoder.columns_):
                        raise DataValidationError(
                            f"multilabel target must have shape (n_samples, {len(self.target_encoder.columns_)})"
                        )
                    target_frame = pd.DataFrame(array, columns=self.target_encoder.columns_)
                y_true = self.target_encoder.transform(target_frame)
            else:
                y_true = np.asarray(y, dtype=int)
            y_proba = self._probabilities_clean(clean)
            metrics = evaluate_multilabel(y_true, y_pred, y_proba)
        elif self.task_type is TaskType.MULTIOUTPUT_REGRESSION:
            y_true = np.asarray(y, dtype=float)
            y_proba = None
            metrics = evaluate_multioutput_regression(y_true, y_pred)
        elif self.task_type is TaskType.RANKING:
            if groups_array is None or len(groups_array) != len(frame):
                raise DataValidationError(
                    "ranking evaluation requires one groups value for every input row"
                )
            y_true = np.asarray(y, dtype=float)
            y_proba = None
            metrics = evaluate_ranking(y_true, y_pred, groups_array)
        else:
            y_true = np.asarray(y, dtype=float)
            y_proba = None
            metrics = evaluate_regression(y_true, y_pred)

        if len(np.asarray(y_true)) != len(frame):
            raise DataValidationError("X and y contain different numbers of rows")
        metrics.update(self._uncertainty_metrics(y_true, y_pred, y_proba, clean))
        evaluation_quality = audit_data_quality(
            frame, column_modalities=getattr(self.cleaning_pipeline, "raw_column_modalities_", {})
        ).to_dict()
        evaluation_drift = (
            compare_drift(self.drift_reference, frame, clean)
            if self.drift_reference is not None else None
        )
        self._last_eval = {
            "y_true": y_true,
            "y_pred": y_pred,
            "y_proba": y_proba,
            "groups": groups_array,
            "metrics": metrics,
            "prediction_seconds": prediction_seconds,
            "data_quality_report": evaluation_quality,
            "drift_report": evaluation_drift,
        }
        return metrics

    def _uncertainty_info(self) -> dict[str, Any]:
        return {
            "calibration": getattr(self.probability_calibrator, "diagnostics_", None),
            "conformal": {
                "enabled": self.conformal_predictor is not None,
                "task": getattr(self.conformal_predictor, "task", None),
                "alpha": getattr(self.conformal_predictor, "alpha", None),
            },
            "ood": {
                "enabled": self.ood_detector is not None,
                "score_threshold": getattr(self.ood_detector, "score_threshold_", None),
            },
            "drift_monitoring": self.drift_reference is not None,
        }

    def report(
        self,
        folder: str | Path | None = None,
        X: Any = None,
        y: Any = None,
        groups: Any = None,
    ) -> dict:
        if folder is None:
            folder = self._report_folder or (
                f"smarttab_reports/{self.model_name}_{time.strftime('%Y%m%d_%H%M%S')}"
            )
        if (X is None) != (y is None):
            raise DataValidationError("report() requires both X and y when fresh evaluation data is supplied")
        if X is not None:
            self.evaluate(X, y, groups=groups)
        if self._last_eval is None:
            raise SmartTabError(
                "report() has no evaluation data; call report(folder, X, y) or evaluate(X, y) first"
            )
        try:
            model_size = len(pickle.dumps(self.estimator, protocol=pickle.HIGHEST_PROTOCOL))
        except Exception:
            model_size = -1
        timings = {**self.timings, "prediction_seconds": self._last_eval["prediction_seconds"]}
        context = ReportContext(
            dataset_profile=self.dataset_profile,
            hardware_profile=self.hardware_profile,
            resource_plan=self.resource_plan,
            model_name=self.model_name,
            task_type=self.task_type,
            best_params=self.best_params,
            primary_metric=self.primary_metric,
            metrics=self._last_eval["metrics"],
            feature_importance=self.feature_importance,
            y_true=self._last_eval["y_true"],
            y_pred=self._last_eval["y_pred"],
            y_proba=self._last_eval["y_proba"],
            timings=timings,
            model_size_bytes=model_size,
            n_final_features=len(self.feature_names),
            class_labels=self.class_labels,
            shap_importance=self.shap_importance,
            ensemble_info=self.ensemble_info,
            decision_threshold=self.decision_threshold,
            reject_threshold=self.reject_threshold,
            per_label_thresholds=self.per_label_thresholds,
            objective=self.objective,
            notes=self.notes,
            static_charts=self.static_charts,
            multimodal_info=self.feature_space,
            data_quality_report=self.data_quality_report,
            cleaning_report=dataclasses.asdict(self.cleaning_pipeline.report_),
            uncertainty_info=self._uncertainty_info(),
            modality_dropout_info=self.modality_dropout_info,
            evaluation_quality_report=self._last_eval.get("data_quality_report"),
            evaluation_drift_report=self._last_eval.get("drift_report"),
        )
        result = generate_report(str(folder), context)
        self._report_folder = str(folder)
        return result

    def save(self, path: str | Path) -> str:
        return save_bundle(
            path,
            model_name=self.model_name,
            task_type=self.task_type,
            estimator=self.estimator,
            cleaning_pipeline=self.cleaning_pipeline,
            target_encoder=self.target_encoder,
            raw_feature_names=self.raw_feature_names,
            feature_names=self.feature_names,
            cat_features=self.cat_features,
            dataset_profile=self.dataset_profile,
            hardware_profile=self.hardware_profile,
            resource_plan=self.resource_plan,
            best_params=self.best_params,
            primary_metric=self.primary_metric,
            metrics=self.metrics,
            class_labels=self.class_labels,
            ensemble_info=self.ensemble_info,
            decision_threshold=self.decision_threshold,
            reject_threshold=self.reject_threshold,
            per_label_thresholds=self.per_label_thresholds,
            objective=self.objective,
            timings=self.timings,
            notes=self.notes,
            static_charts=self.static_charts,
            probability_calibrator=self.probability_calibrator,
            conformal_predictor=self.conformal_predictor,
            ood_detector=self.ood_detector,
            drift_reference=self.drift_reference,
            data_science_config=self.data_science_config,
            data_quality_report=self.data_quality_report,
            modality_dropout_info=self.modality_dropout_info,
        )
