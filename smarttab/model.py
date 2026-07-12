"""``SmartTabModel`` — the object returned by ``fit()`` and ``load()``."""

from __future__ import annotations

import pickle
import time

import numpy as np
import pandas as pd

from smarttab.analysis.dataset_analyzer import DatasetProfile, TaskType
from smarttab.cleaning.encoders import TargetLabelEncoder
from smarttab.cleaning.pipeline import SmartCleaningPipeline
from smarttab.config import load_data
from smarttab.evaluation.evaluator import (
    evaluate_classification,
    evaluate_multilabel,
    evaluate_multioutput_regression,
    evaluate_ranking,
    evaluate_regression,
)
from smarttab.exceptions import SmartTabError
from smarttab.hardware.profiler import HardwareProfile
from smarttab.hardware.resource_planner import ResourcePlan
from smarttab.optimization.threshold import DEFAULT_OBJECTIVE, DEFAULT_REJECT_THRESHOLD, DEFAULT_THRESHOLD, apply_threshold
from smarttab.persistence.serializer import save_bundle
from smarttab.prediction_result import PredictionArray, PredictionWithConfidence
from smarttab.reporting.report_generator import ReportContext, generate_report
from smarttab.training.trainer import predict, predict_proba


class SmartTabModel:
    def __init__(
        self,
        *,
        model_name: str,
        task_type: TaskType,
        estimator,
        cleaning_pipeline: SmartCleaningPipeline,
        target_encoder: TargetLabelEncoder | None,
        feature_names: list[str],
        cat_features: list[str],
        best_params: dict,
        primary_metric: str,
        metrics: dict,
        feature_importance: pd.DataFrame,
        dataset_profile: DatasetProfile,
        hardware_profile: HardwareProfile,
        resource_plan: ResourcePlan,
        class_labels: list | None,
        timings: dict,
        notes: list[str],
        shap_importance: pd.DataFrame | None = None,
        ensemble_info: dict | None = None,
        decision_threshold: float = DEFAULT_THRESHOLD,
        reject_threshold: float = DEFAULT_REJECT_THRESHOLD,
        per_label_thresholds: list[float] | None = None,
        objective: str = DEFAULT_OBJECTIVE,
        multi_threshold_ensemble: bool = False,
        threshold_ladder: list[dict] | None = None,
    ) -> None:
        self.model_name = model_name
        self.task_type = task_type
        self.estimator = estimator
        self.cleaning_pipeline = cleaning_pipeline
        self.target_encoder = target_encoder
        self.feature_names = feature_names
        self.cat_features = cat_features
        self.best_params = best_params
        self.primary_metric = primary_metric
        self.metrics = metrics
        self.feature_importance = feature_importance
        self.dataset_profile = dataset_profile
        self.hardware_profile = hardware_profile
        self.resource_plan = resource_plan
        self.class_labels = class_labels
        self.timings = timings
        self.notes = notes
        self.shap_importance = shap_importance
        self.ensemble_info = ensemble_info
        self.decision_threshold = decision_threshold
        self.reject_threshold = reject_threshold
        self.per_label_thresholds = per_label_thresholds
        self.objective = objective
        self.multi_threshold_ensemble = multi_threshold_ensemble
        self.threshold_ladder = threshold_ladder

        # In-memory only (not persisted by save()): lets report() be called
        # again in the same session without re-evaluating.
        self._last_eval: dict | None = None
        self._report_folder: str | None = None

    def _clean(self, X: pd.DataFrame) -> pd.DataFrame:
        return self.cleaning_pipeline.transform(X)

    def _coerce_predict_input(self, X) -> tuple[pd.DataFrame, bool]:
        """Normalizes whatever ``predict()``/``predict_proba()`` were handed into
        ``(DataFrame, is_single_sample)``. Accepts:

        - a ``DataFrame`` (used as-is)
        - a path (``str``) to any file ``load_data()`` supports — CSV, TSV, Excel,
          Parquet, JSON, Feather, Pickle
        - a single ``dict`` (one sample; keys must match the training feature names)
        - a list/tuple of ``dict`` (a batch of samples)
        - a ``numpy.ndarray`` — 1-D for a single sample, 2-D for a batch — columns are
          assumed to be in ``self.feature_names`` order
        """
        if isinstance(X, pd.DataFrame):
            return X, False
        if isinstance(X, str):
            return load_data(X), False
        if isinstance(X, dict):
            return pd.DataFrame([X]), True
        if isinstance(X, np.ndarray):
            # a mixed-type array (e.g. numbers + strings) forces every column to object
            # dtype; infer_objects() recovers proper numeric dtypes so cleaning/imputation
            # behaves the same as it would for a DataFrame built with correct dtypes.
            if X.ndim == 1:
                return pd.DataFrame([X], columns=self.feature_names).infer_objects(), True
            return pd.DataFrame(X, columns=self.feature_names).infer_objects(), False
        if isinstance(X, (list, tuple)) and X and isinstance(X[0], dict):
            return pd.DataFrame(list(X)), False
        return pd.DataFrame(X), False

    def _predict_encoded(self, X_clean: pd.DataFrame) -> np.ndarray:
        """Predict class codes / regression values from already-cleaned features.

        Binary classification goes through ``self.decision_threshold`` and multi-label goes
        through ``self.per_label_thresholds`` (one per label) rather than each library's
        built-in (always-0.5) cutoff, so threshold optimization actually takes effect on every
        prediction, not just the metrics computed right after fit(). Multi-class's
        ``reject_threshold`` deliberately does *not* affect this method — it only ever changes
        ``predict()``'s output, and only when ``multi_threshold_ensemble=True`` (see
        ``predict()``), so plain argmax stays the default here exactly as before this feature
        existed.
        """
        if self.task_type is TaskType.BINARY:
            proba = predict_proba(self.estimator, self.model_name, X_clean)
            return apply_threshold(proba[:, 1], self.decision_threshold)
        if self.task_type is TaskType.MULTILABEL and self.per_label_thresholds is not None:
            proba = predict_proba(self.estimator, self.model_name, X_clean)
            return np.stack(
                [apply_threshold(proba[:, i], t) for i, t in enumerate(self.per_label_thresholds)], axis=1,
            )
        return predict(self.estimator, self.model_name, X_clean)

    def _confidence_from_ladder(self, proba_positive: np.ndarray, predicted_codes: np.ndarray) -> np.ndarray:
        """Binary: fraction of ``self.threshold_ladder`` cutoffs (lenient -> strict) that agree
        with each row's final prediction — e.g. a positive call that also clears the strictest
        threshold is high-confidence; one that only clears the most lenient threshold is a
        borderline, low-confidence call."""
        ladder_thresholds = np.array([point["threshold"] for point in self.threshold_ladder])
        n = len(ladder_thresholds)
        cleared = (proba_positive[:, None] >= ladder_thresholds[None, :]).sum(axis=1)
        return np.where(predicted_codes == 1, cleared / n, (n - cleared) / n)

    def _multiclass_confidence_from_ladder(self, max_proba: np.ndarray) -> np.ndarray:
        """Multi-class: fraction of the reject-threshold ladder (lenient -> strict) that still
        accepts the row — there's no "direction" to agree with here (unlike binary), since every
        ladder level is just a stricter bar for the same top predicted class."""
        ladder_thresholds = np.array([point["threshold"] for point in self.threshold_ladder])
        n = len(ladder_thresholds)
        cleared = (max_proba[:, None] >= ladder_thresholds[None, :]).sum(axis=1)
        return cleared / n

    def _multilabel_confidence_from_ladder(self, proba: np.ndarray, predicted_codes: np.ndarray) -> np.ndarray:
        """Multi-label: the binary ladder-agreement formula, run independently per label
        column. ``self.threshold_ladder`` is a list of one ladder per label."""
        confidence = np.zeros_like(proba, dtype=float)
        for i, label_ladder in enumerate(self.threshold_ladder):
            ladder_thresholds = np.array([point["threshold"] for point in label_ladder])
            n = len(ladder_thresholds)
            cleared = (proba[:, i : i + 1] >= ladder_thresholds[None, :]).sum(axis=1)
            confidence[:, i] = np.where(predicted_codes[:, i] == 1, cleared / n, (n - cleared) / n)
        return confidence

    def predict(self, X):
        """``X`` accepts a ``DataFrame``, a path to a CSV/Parquet/JSON/etc. file, a single
        ``dict`` (one sample), a list of ``dict``, or a ``numpy.ndarray`` — see
        ``_coerce_predict_input``. The return value is always a real ``np.ndarray`` (or,
        when ``multi_threshold_ensemble=True``, a real ``(labels, confidence)`` tuple) —
        every existing usage keeps working — but it also exposes ``.prediction``/``.label``/
        ``.probability`` and ``.csv``/``.json`` for convenience; see documents.md.
        """
        X_df, is_single = self._coerce_predict_input(X)
        X_clean = self._clean(X_df)

        # When multi_threshold_ensemble is in play, derive both the hard prediction and the
        # confidence score from a single predict_proba() call per branch, rather than calling
        # _predict_encoded() (which itself calls predict_proba() for BINARY/MULTILABEL) and then
        # predict_proba() again separately — that would run inference on the estimator twice.
        if self.multi_threshold_ensemble and self.threshold_ladder:
            if self.task_type is TaskType.BINARY:
                proba_positive = predict_proba(self.estimator, self.model_name, X_clean)[:, 1]
                codes = apply_threshold(proba_positive, self.decision_threshold)
                confidence = self._confidence_from_ladder(proba_positive, codes)
                labels = self.target_encoder.inverse_transform(codes) if self.target_encoder is not None else codes
                return PredictionWithConfidence(labels, confidence, single=is_single)

            if self.task_type is TaskType.MULTICLASS:
                proba = predict_proba(self.estimator, self.model_name, X_clean)
                max_proba = proba.max(axis=1)
                predicted_class = proba.argmax(axis=1)
                accepted = max_proba >= self.reject_threshold
                confidence = self._multiclass_confidence_from_ladder(max_proba)
                labels = np.empty(len(predicted_class), dtype=object)
                if self.target_encoder is not None:
                    labels[accepted] = self.target_encoder.inverse_transform(predicted_class[accepted])
                else:
                    labels[accepted] = predicted_class[accepted]
                labels[~accepted] = None  # below reject_threshold: too uncertain to call
                return PredictionWithConfidence(labels, confidence, single=is_single)

            if self.task_type is TaskType.MULTILABEL:
                proba = predict_proba(self.estimator, self.model_name, X_clean)
                if self.per_label_thresholds is not None:
                    codes = np.stack(
                        [apply_threshold(proba[:, i], t) for i, t in enumerate(self.per_label_thresholds)], axis=1,
                    )
                else:
                    codes = (proba >= 0.5).astype(int)
                confidence = self._multilabel_confidence_from_ladder(proba, codes)
                return PredictionWithConfidence(codes, confidence, single=is_single)

        codes = self._predict_encoded(X_clean)
        if self.task_type.is_classification and self.target_encoder is not None:
            return PredictionArray(self.target_encoder.inverse_transform(codes), single=is_single)
        return PredictionArray(codes, single=is_single)

    def predict_proba(self, X) -> np.ndarray:
        """Accepts the same input types as ``predict()`` — see ``_coerce_predict_input``."""
        if not self.task_type.is_classification:
            raise SmartTabError("predict_proba() is only available for classification tasks")
        X_df, is_single = self._coerce_predict_input(X)
        X_clean = self._clean(X_df)
        proba = predict_proba(self.estimator, self.model_name, X_clean)
        return proba[0] if is_single else proba

    def evaluate(self, X: pd.DataFrame, y, groups=None) -> dict:
        """For ``task_type is TaskType.RANKING``, ``groups`` (the query/group id per row of
        ``X``) is required — NDCG is only meaningful within a group."""
        X_clean = self._clean(X)
        start = time.perf_counter()
        y_pred = self._predict_encoded(X_clean)
        prediction_seconds = time.perf_counter() - start
        groups_array = np.asarray(groups) if groups is not None else None

        if self.task_type in (TaskType.BINARY, TaskType.MULTICLASS):
            y_true = self.target_encoder.transform(pd.Series(y))
            y_proba = predict_proba(self.estimator, self.model_name, X_clean)
            metrics = evaluate_classification(y_true, y_pred, y_proba)
        elif self.task_type is TaskType.MULTILABEL:
            y_true = np.asarray(y, dtype=int)
            y_proba = predict_proba(self.estimator, self.model_name, X_clean)
            metrics = evaluate_multilabel(y_true, y_pred)
        elif self.task_type is TaskType.MULTIOUTPUT_REGRESSION:
            y_true = np.asarray(y, dtype=float)
            y_proba = None
            metrics = evaluate_multioutput_regression(y_true, y_pred)
        elif self.task_type is TaskType.RANKING:
            if groups_array is None:
                raise SmartTabError(
                    "evaluate() on a ranking model requires groups=... (the query/group id "
                    "for each row of X) — NDCG can only be computed within a group."
                )
            y_true = np.asarray(y, dtype=float)
            y_proba = None
            metrics = evaluate_ranking(y_true, y_pred, groups_array)
        else:
            y_true = np.asarray(y, dtype=float)
            y_proba = None
            metrics = evaluate_regression(y_true, y_pred)

        self._last_eval = {
            "y_true": y_true, "y_pred": y_pred, "y_proba": y_proba, "groups": groups_array,
            "metrics": metrics, "prediction_seconds": prediction_seconds,
        }
        return metrics

    def report(self, folder: str, X: pd.DataFrame | None = None, y=None, groups=None) -> dict:
        """Write ``folder/report.html``, ``folder/report.json``, and ``folder/charts/*.png``.

        Returns the same data as a plain dict (also what ends up in report.json).
        ``folder`` is required and will be created if it doesn't exist.
        """
        if X is not None and y is not None:
            self.evaluate(X, y, groups=groups)

        if self._last_eval is None:
            raise SmartTabError(
                "report() needs evaluation data. Either call model.evaluate(X, y) first, "
                "or call model.report(folder, X, y). (A freshly loaded model has no cached holdout data.)"
            )

        timings = {**self.timings, "prediction_seconds": self._last_eval["prediction_seconds"]}

        ctx = ReportContext(
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
            model_size_bytes=len(pickle.dumps(self.estimator)),
            n_final_features=len(self.feature_names),
            class_labels=self.class_labels,
            shap_importance=self.shap_importance,
            ensemble_info=self.ensemble_info,
            decision_threshold=self.decision_threshold,
            reject_threshold=self.reject_threshold,
            per_label_thresholds=self.per_label_thresholds,
            objective=self.objective,
            multi_threshold_ensemble=self.multi_threshold_ensemble,
            threshold_ladder=self.threshold_ladder,
            notes=self.notes,
        )
        report_dict = generate_report(folder, ctx)
        self._report_folder = folder
        return report_dict

    def save(self, path: str) -> str:
        saved_path = save_bundle(
            path,
            model_name=self.model_name, task_type=self.task_type, estimator=self.estimator,
            cleaning_pipeline=self.cleaning_pipeline, target_encoder=self.target_encoder,
            feature_names=self.feature_names, cat_features=self.cat_features,
            dataset_profile=self.dataset_profile, hardware_profile=self.hardware_profile,
            resource_plan=self.resource_plan, best_params=self.best_params,
            primary_metric=self.primary_metric, metrics=self.metrics, class_labels=self.class_labels,
            ensemble_info=self.ensemble_info, decision_threshold=self.decision_threshold,
            reject_threshold=self.reject_threshold, per_label_thresholds=self.per_label_thresholds,
            objective=self.objective, multi_threshold_ensemble=self.multi_threshold_ensemble,
            threshold_ladder=self.threshold_ladder,
        )
        return saved_path
