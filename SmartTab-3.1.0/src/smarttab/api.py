"""Public :func:`fit` and :func:`load` entry points."""

from __future__ import annotations

import dataclasses
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import (
    GroupShuffleSplit,
    StratifiedGroupKFold,
    train_test_split,
)

from smarttab.analysis.dataset_analyzer import (
    DatasetProfile,
    TaskType,
    analyze_dataset,
    resolve_task_and_targets,
)
from smarttab.cleaning.encoders import MultiLabelTargetEncoder, TargetLabelEncoder
from smarttab.cleaning.outliers import compute_outlier_keep_mask
from smarttab.cleaning.pipeline import SmartCleaningPipeline
from smarttab.config import FitConfig, load_data
from smarttab.deadline import FitDeadline, TimeLimitExceeded
from smarttab.datascience.drift import DriftReference, compare_drift
from smarttab.datascience.quality import DataQualityReport, audit_data_quality
from smarttab.datascience.uncertainty import ConformalPredictor, OODDetector, ProbabilityCalibrator
from smarttab.evaluation.evaluator import (
    evaluate_classification,
    evaluate_multilabel,
    evaluate_multioutput_regression,
    evaluate_ranking,
    evaluate_regression,
)
from smarttab.exceptions import ConfigurationError, DataValidationError
from smarttab.explainability.explainer import get_feature_importance, get_shap_importance
from smarttab.hardware.profiler import profile_hardware
from smarttab.hardware.resource_planner import resolve_resource_plan
from smarttab.logging_utils import get_logger, set_verbosity
from smarttab.model import SmartTabModel
from smarttab.multimodal.config import resolve_feature_space_config
from smarttab.multimodal.detector import resolve_column_modalities
from smarttab.optimization.optimizer import resolve_class_weight_params, run_optimization
from smarttab.optimization.search_spaces import default_params
from smarttab.optimization.threshold import (
    DEFAULT_REJECT_THRESHOLD,
    DEFAULT_THRESHOLD,
    apply_threshold,
    optimize_per_label_thresholds,
    optimize_threshold,
    probe_and_optimize_multilabel_thresholds,
    probe_and_optimize_threshold,
)
from smarttab.persistence.serializer import load_bundle
from smarttab.selection.model_selector import select_model
from smarttab.training.ensemble import (
    run_ensemble_decision_engine,
    train_voting_stacking_ensemble,
)
from smarttab.training.trainer import (
    PeakMemoryTracker,
    build_estimator,
    fit_estimator,
    predict,
    predict_proba,
    sort_by_group,
    verify_gpu_usable,
)

ENSEMBLE_SUPPORTED_TASK_TYPES = (TaskType.BINARY, TaskType.MULTICLASS, TaskType.REGRESSION)
REMOVED_PARAMETERS = {"multi_threshold_ensemble", "threshold_models"}
MIN_ENSEMBLE_TIME_BUDGET = 8.0


def fit(
    data,
    target: str | list[str] | None = None,
    group_id: str | None = None,
    *,
    y=None,
    modality: str = "auto",
    modalities: dict[str, str] | str | None = None,
    **kwargs,
) -> SmartTabModel:
    data, target, resolved_modalities = _prepare_fit_input(
        data, target=target, y=y, modality=modality, modalities=modalities
    )

    removed = sorted(REMOVED_PARAMETERS.intersection(kwargs))
    if removed:
        raise ConfigurationError(
            f"removed parameter(s): {removed}. Multi-threshold pseudo-ensembling was removed; "
            "use ensemble='auto' for a real OOF voting/stacking ensemble."
        )

    config = FitConfig(target=target, group_id=group_id, modalities=resolved_modalities, **kwargs)
    set_verbosity(config.verbose)
    logger = get_logger()
    deadline = FitDeadline(float(config.time_limit))

    df = load_data(data)
    source_n_samples = len(df)
    target_columns = [target] if isinstance(target, str) else list(target)
    df, dropped_target_rows = _handle_missing_targets(df, target_columns, config.target_missing)
    df, duplicate_rows_removed = _handle_duplicate_rows(df, config.duplicate_policy)
    df, conflicting_label_rows_removed = _handle_conflicting_labels(
        df, target_columns, config.group_id, config.data_science_config.conflicting_labels
    )

    task_type, target_columns = resolve_task_and_targets(
        df,
        config.target,
        config.group_id,
        config.task_type,
    )
    if config.ensemble != "none" and task_type not in ENSEMBLE_SUPPORTED_TASK_TYPES:
        raise ConfigurationError(
            f"ensemble={config.ensemble!r} supports binary, multiclass, and regression only; "
            f"resolved task_type={task_type.value!r}"
        )

    logger.info(
        "resolved task_type=%s; source rows=%d; usable rows=%d",
        task_type.value,
        source_n_samples,
        len(df),
    )
    if dropped_target_rows:
        logger.info("dropped %d row(s) with missing target values", dropped_target_rows)
    if duplicate_rows_removed:
        logger.info("dropped %d exact duplicate row(s) before splitting", duplicate_rows_removed)
    if conflicting_label_rows_removed:
        logger.info("dropped %d row(s) with conflicting duplicate labels", conflicting_label_rows_removed)

    train_df, test_df, group_train, group_test = _split_train_test(
        df,
        task_type,
        target_columns,
        config,
    )
    train_df, group_train, low_quality_rows_removed = _drop_low_quality_training_rows(
        train_df, target_columns, config.group_id, group_train,
        config.data_science_config.row_missing_threshold,
    )

    logger.info("Stage 1/9: train-only dataset analysis and quality audit")
    profile = analyze_dataset(
        train_df,
        config.target,
        config.group_id,
        task_type,
        modalities=config.modalities,
    )
    profile.source_n_samples = source_n_samples
    profile.holdout_n_samples = len(test_df)
    profile.duplicate_rows_removed = duplicate_rows_removed
    profile.low_quality_rows_removed = low_quality_rows_removed
    profile.conflicting_label_rows_removed = conflicting_label_rows_removed
    quality_report = audit_data_quality(
        train_df, target_columns=target_columns, column_modalities=profile.column_modalities
    )
    profile.data_quality_report = quality_report.to_dict()
    _enforce_quality_policy(quality_report, config.data_science_config.quality_policy)

    if config.outlier == "remove":
        keep_mask = compute_outlier_keep_mask(train_df, profile.numeric_columns)
        removed_outliers = int((~keep_mask).sum())
        if removed_outliers:
            train_df = train_df.loc[keep_mask].reset_index(drop=True)
            if group_train is not None:
                group_train = group_train[keep_mask.to_numpy()]
            profile = analyze_dataset(
                train_df,
                config.target,
                config.group_id,
                task_type,
                modalities=config.modalities,
            )
            profile.source_n_samples = source_n_samples
            profile.holdout_n_samples = len(test_df)
            profile.duplicate_rows_removed = duplicate_rows_removed
            profile.low_quality_rows_removed = low_quality_rows_removed
            profile.conflicting_label_rows_removed = conflicting_label_rows_removed
            profile.outlier_rows_removed = removed_outliers
            quality_report = audit_data_quality(
                train_df, target_columns=target_columns, column_modalities=profile.column_modalities
            )
            profile.data_quality_report = quality_report.to_dict()
            logger.info("removed %d training-only outlier row(s)", removed_outliers)

    logger.info("Stage 2/9: fitting cleaning pipeline on training rows")
    feature_space_config = resolve_feature_space_config(
        config.feature_budget,
        speed_accuracy=config.speed_accuracy,
        backend=config.multimodal_backend,
        allow_model_download=config.allow_model_download,
        error_policy=config.media_error_policy,
        batch_size=config.batch_size,
        workers=config.feature_workers,
        cache=config.feature_cache,
        modality_params=config.modality_params,
        random_state=config.random_state,
        device=config.device,
        supervised_adaptation=config.supervised_adaptation,
        adapter_features=config.adapter_features,
    )
    pipeline = SmartCleaningPipeline(
        clean=config.clean,
        missing=config.missing,
        categorical=config.categorical,
        scaling=config.scaling,
        feature_selection=config.feature_selection,
        outlier=config.outlier,
        leakage_policy=config.leakage_policy,
        schema_policy=config.schema_policy,
        feature_space_config=feature_space_config,
        data_science_config=config.data_science_config,
    )
    X_train = pipeline.fit_transform(train_df, train_df[target_columns], profile)
    X_test = pipeline.transform(test_df[profile.feature_columns])
    cat_features = pipeline.final_categorical_columns
    target_encoder, y_train, y_test, class_labels = _extract_targets(
        train_df,
        test_df,
        profile,
    )
    profile.cleaning_report = dataclasses.asdict(pipeline.report_)
    X_reference = X_train.copy()
    raw_reference = train_df[profile.feature_columns].reset_index(drop=True).copy()
    X_train, y_train, group_train, X_calibration, y_calibration = _split_uncertainty_holdout(
        X_train, y_train, group_train, task_type, config
    )
    X_train, y_train, group_train, dropout_info = _augment_modality_dropout(
        X_train, y_train, group_train, pipeline.feature_groups, config,
    )

    logger.info("Stage 3/9: hardware profiling")
    hardware = profile_hardware()
    resource_plan = resolve_resource_plan(
        hardware,
        profile,
        device=config.device,
        cpu_threads=config.cpu_threads,
        gpu_memory=config.gpu_memory,
        ram_limit=config.ram_limit,
    )

    logger.info("Stage 4/9: model strategy selection")
    notes: list[str] = list(resource_plan.notes)
    notes.append(f"data_quality_score={quality_report.quality_score:.1f}/100")
    if low_quality_rows_removed:
        notes.append(f"removed {low_quality_rows_removed} almost-empty training rows")
    if conflicting_label_rows_removed:
        notes.append(f"removed {conflicting_label_rows_removed} rows with contradictory duplicate labels")
    if dropout_info.get("augmented_rows", 0):
        notes.append(
            f"missing-modality robustness added {dropout_info['augmented_rows']} training rows"
        )
    ensemble_info: dict | None = None
    decision_threshold = DEFAULT_THRESHOLD
    reject_threshold = DEFAULT_REJECT_THRESHOLD
    per_label_thresholds: list[float] | None = None

    requested_ensemble = config.ensemble
    if requested_ensemble != "none" and deadline.enabled:
        remaining = deadline.remaining() or 0.0
        if remaining < MIN_ENSEMBLE_TIME_BUDGET:
            notes.append(
                f"ensemble skipped because only {remaining:.1f}s remained under time_limit"
            )
            requested_ensemble = "none"

    if requested_ensemble == "none":
        model_name, model_notes = select_model(profile, config.model)
        notes.extend(model_notes)
        if resource_plan.use_gpu:
            resource_plan.use_gpu = verify_gpu_usable(
                model_name,
                task_type,
                resource_plan.cpu_threads,
                config.random_state,
            )
        estimator, best_params, primary_metric, training_seconds, memory_info, threshold_notes = _train_single_model(
            model_name,
            X_train,
            y_train,
            profile,
            resource_plan,
            cat_features,
            group_train,
            config,
            deadline,
        )
        notes.extend(threshold_notes)
        if config.threshold_optimization and task_type is TaskType.BINARY:
            if not deadline.expired():
                decision_threshold = probe_and_optimize_threshold(
                    model_name,
                    best_params,
                    _model_n_estimators(estimator, model_name, config.n_estimators, profile),
                    task_type,
                    resource_plan,
                    cat_features,
                    X_train,
                    y_train,
                    config.random_state,
                    config.objective,
                    deadline,
                )
                notes.append(
                    f"decision_threshold={decision_threshold:.2f}; optimized for {config.objective}"
                )
        elif config.threshold_optimization and task_type is TaskType.MULTILABEL:
            if not deadline.expired():
                per_label_thresholds = probe_and_optimize_multilabel_thresholds(
                    model_name,
                    best_params,
                    _model_n_estimators(estimator, model_name, config.n_estimators, profile),
                    task_type,
                    resource_plan,
                    cat_features,
                    X_train,
                    y_train,
                    config.random_state,
                    config.objective,
                    deadline,
                )
                notes.append("optimized one decision threshold per multilabel target")
    else:
        if resource_plan.use_gpu:
            resource_plan.use_gpu = all(
                verify_gpu_usable(
                    model_name,
                    task_type,
                    resource_plan.cpu_threads,
                    config.random_state,
                )
                for model_name in ("catboost", "lightgbm")
            )
        search_timeout = _search_timeout(config, deadline, ensemble=True)
        training_started = time.perf_counter()
        with PeakMemoryTracker() as tracker:
            if requested_ensemble == "auto":
                result = run_ensemble_decision_engine(
                    X_train,
                    y_train,
                    task_type,
                    profile,
                    resource_plan,
                    cat_features,
                    validation=config.validation,
                    cv=config.cv,
                    optimizer=config.optimizer,
                    n_trials=config.n_trials,
                    timeout=search_timeout,
                    metrics=config.metrics,
                    optimize=config.optimize,
                    random_state=config.random_state,
                    verbose=config.verbose,
                    threshold_optimization=config.threshold_optimization,
                    objective=config.objective,
                    xgboost_policy=config.xgboost_policy,
                    ensemble_models_limit=config.ensemble_models_limit,
                    ensemble_min_gain=config.ensemble_min_gain,
                    diversity_correlation_limit=config.diversity_correlation_limit,
                    meta_model=config.meta_model,
                    feature_groups=pipeline.feature_groups,
                    fusion=config.fusion,
                    deadline=deadline,
                )
                estimator = result.estimator
                model_name = result.strategy
                best_params = result.best_params
                primary_metric = result.primary_metric
                decision_threshold = result.decision_threshold
                reject_threshold = result.reject_threshold
                ensemble_info = result.ensemble_info
                notes.extend(result.notes)
            else:
                result = train_voting_stacking_ensemble(
                    X_train,
                    y_train,
                    task_type,
                    profile,
                    resource_plan,
                    cat_features,
                    validation=config.validation,
                    cv=config.cv,
                    optimizer=config.optimizer,
                    n_trials=config.n_trials,
                    timeout=search_timeout,
                    metrics=config.metrics,
                    optimize=config.optimize,
                    random_state=config.random_state,
                    verbose=config.verbose,
                    strategy=requested_ensemble,
                    threshold_optimization=config.threshold_optimization,
                    objective=config.objective,
                    xgboost_policy=config.xgboost_policy,
                    ensemble_models_limit=config.ensemble_models_limit,
                    ensemble_min_gain=config.ensemble_min_gain,
                    diversity_correlation_limit=config.diversity_correlation_limit,
                    meta_model=config.meta_model,
                    feature_groups=pipeline.feature_groups,
                    fusion=config.fusion,
                    deadline=deadline,
                )
                estimator = result.estimator
                model_name = result.strategy
                best_params = result.base_params
                primary_metric = result.primary_metric
                decision_threshold = result.decision_threshold
                reject_threshold = result.reject_threshold
                notes.extend(result.notes)
                ensemble_info = {
                    "strategy": result.strategy,
                    "validation_score": result.validation_score,
                    "base_params": result.base_params,
                    "base_n_estimators": result.base_n_estimators,
                    "base_scores": result.base_scores,
                    "selection_scores": result.selection_scores,
                    "selection_vectors": result.selection_vectors,
                    "meta_model_name": result.meta_model_name,
                    "voting_weights": result.voting_weights,
                    "members": result.members,
                    "candidates": result.candidates,
                    "diversity_matrix": result.diversity_matrix,
                    "fusion_strategy": result.fusion_strategy,
                }
        training_seconds = time.perf_counter() - training_started
        memory_info = {
            "peak_training_memory_mb": tracker.peak_mb,
            "training_memory_delta_mb": tracker.delta_mb,
        }

    probability_calibrator = None
    conformal_predictor = None
    if X_calibration is not None and y_calibration is not None:
        probability_calibrator, conformal_predictor, uncertainty_notes = _fit_uncertainty_models(
            estimator, model_name, task_type, X_calibration, y_calibration, config
        )
        notes.extend(uncertainty_notes)
        if probability_calibrator is not None and task_type is TaskType.BINARY and config.threshold_optimization:
            calibrated = probability_calibrator.transform(
                predict_proba(estimator, model_name, X_calibration)
            )
            decision_threshold = optimize_threshold(
                y_calibration, calibrated[:, 1], config.objective
            )[0]
        elif probability_calibrator is not None and task_type is TaskType.MULTILABEL and config.threshold_optimization:
            calibrated = probability_calibrator.transform(
                predict_proba(estimator, model_name, X_calibration)
            )
            per_label_thresholds = optimize_per_label_thresholds(
                y_calibration, calibrated, config.objective
            )

    ood_detector = OODDetector().fit(X_reference) if config.data_science_config.ood_detection else None
    drift_reference = (
        DriftReference.fit(
            raw_reference, X_reference,
            numeric_columns=profile.numeric_columns,
            categorical_columns=profile.categorical_columns,
        )
        if config.data_science_config.drift_monitoring else None
    )

    logger.info("Stage 7/9: held-out evaluation")
    prediction_started = time.perf_counter()
    y_proba = None
    if task_type is TaskType.BINARY:
        y_proba = _calibrated_probabilities(estimator, model_name, X_test, probability_calibrator)
        y_pred = apply_threshold(y_proba[:, 1], decision_threshold)
    elif task_type is TaskType.MULTICLASS:
        y_proba = _calibrated_probabilities(estimator, model_name, X_test, probability_calibrator)
        y_pred = np.argmax(y_proba, axis=1)
    elif task_type is TaskType.MULTILABEL:
        y_proba = _calibrated_probabilities(estimator, model_name, X_test, probability_calibrator)
        thresholds = per_label_thresholds or [DEFAULT_THRESHOLD] * y_proba.shape[1]
        y_pred = np.column_stack(
            [apply_threshold(y_proba[:, index], threshold) for index, threshold in enumerate(thresholds)]
        )
    else:
        y_pred = predict(estimator, model_name, X_test)
    prediction_seconds = time.perf_counter() - prediction_started

    metrics = _evaluate_task(task_type, y_test, y_pred, y_proba, group_test)
    metrics.update(_uncertainty_holdout_metrics(
        task_type, y_test, y_pred, y_proba, conformal_predictor, ood_detector, X_test
    ))
    logger.info(
        "holdout metrics: %s",
        {key: _format_metric_for_log(value) for key, value in metrics.items()},
    )

    logger.info("Stage 8/9: explainability")
    feature_importance = get_feature_importance(estimator, model_name, list(X_train.columns))
    shap_importance = None
    should_compute_shap = config.explain is True or (config.explain == "auto" and config.report)
    if should_compute_shap and _has_optional_time(deadline, minimum=2.0):
        shap_importance = get_shap_importance(estimator, model_name, X_train)
    elif should_compute_shap:
        notes.append("SHAP skipped because the global time budget was exhausted")
    elif config.explain == "auto" and not config.report:
        notes.append("SHAP auto mode skipped because report=False; native importance remains available")

    timings = {
        "fit_total_seconds": deadline.elapsed(),
        "training_seconds": training_seconds,
        "prediction_seconds": prediction_seconds,
        **memory_info,
    }
    model = SmartTabModel(
        model_name=model_name,
        task_type=task_type,
        estimator=estimator,
        cleaning_pipeline=pipeline,
        target_encoder=target_encoder,
        raw_feature_names=list(pipeline.raw_feature_columns_),
        feature_names=list(X_train.columns),
        cat_features=cat_features,
        best_params=best_params,
        primary_metric=primary_metric,
        metrics=metrics,
        feature_importance=feature_importance,
        dataset_profile=profile,
        hardware_profile=hardware,
        resource_plan=resource_plan,
        class_labels=class_labels,
        timings=timings,
        notes=notes,
        shap_importance=shap_importance,
        ensemble_info=ensemble_info,
        decision_threshold=decision_threshold,
        reject_threshold=reject_threshold,
        per_label_thresholds=per_label_thresholds,
        objective=config.objective,
        static_charts=config.static_charts,
        probability_calibrator=probability_calibrator,
        conformal_predictor=conformal_predictor,
        ood_detector=ood_detector,
        drift_reference=drift_reference,
        data_science_config=config.data_science_config.to_dict(),
        data_quality_report=quality_report.to_dict(),
        modality_dropout_info=dropout_info,
    )
    evaluation_quality_report = audit_data_quality(
        test_df, target_columns=target_columns, column_modalities=profile.column_modalities
    ).to_dict()
    evaluation_drift_report = (
        compare_drift(
            drift_reference,
            test_df[profile.feature_columns].reset_index(drop=True),
            X_test.reset_index(drop=True),
        )
        if drift_reference is not None else None
    )
    model._last_eval = {
        "y_true": y_test,
        "y_pred": y_pred,
        "y_proba": y_proba,
        "groups": group_test,
        "metrics": metrics,
        "prediction_seconds": prediction_seconds,
        "data_quality_report": evaluation_quality_report,
        "drift_report": evaluation_drift_report,
    }

    if config.report:
        if _has_optional_time(deadline, minimum=1.0):
            logger.info("Stage 9/9: report generation")
            folder = f"smarttab_reports/{model_name}_{time.strftime('%Y%m%d_%H%M%S')}"
            model.report(folder)
        else:
            model.notes.append("automatic report skipped because time_limit was exhausted")
    model.timings["fit_total_seconds"] = deadline.elapsed()
    if deadline.enabled:
        model.notes.append(
            f"time_limit={config.time_limit:.3f}s; elapsed={model.timings['fit_total_seconds']:.3f}s"
        )
    return model



def audit(
    data,
    target: str | list[str] | None = None,
    *,
    modalities: dict[str, str] | str | None = "auto",
) -> DataQualityReport:
    """Run the SmartTab data-quality audit without fitting a model."""
    frame = load_data(data)
    target_columns = [] if target is None else ([target] if isinstance(target, str) else list(target))
    missing = [column for column in target_columns if column not in frame.columns]
    if missing:
        raise DataValidationError(f"target column(s) not found: {missing}")
    features = [column for column in frame.columns if column not in set(target_columns)]
    column_modalities = resolve_column_modalities(frame, features, modalities)
    return audit_data_quality(
        frame,
        target_columns=target_columns,
        column_modalities=column_modalities,
    )


def fit_text(texts, y, **kwargs) -> SmartTabModel:
    """Fit directly from raw text samples and labels/targets."""
    return fit(texts, y=y, modality="text", **kwargs)


def fit_images(images, y, **kwargs) -> SmartTabModel:
    """Fit directly from image paths, PIL images, bytes, or NumPy arrays."""
    return fit(images, y=y, modality="image", **kwargs)


def fit_audio(audio, y, **kwargs) -> SmartTabModel:
    """Fit directly from audio paths, WAV bytes, arrays, or ``(sample_rate, waveform)`` tuples."""
    return fit(audio, y=y, modality="audio", **kwargs)


def fit_videos(videos, y, **kwargs) -> SmartTabModel:
    """Fit directly from video paths or frame arrays."""
    return fit(videos, y=y, modality="video", **kwargs)


def fit_folder(
    folder: str | Path,
    *,
    modality: str = "image",
    label_from: str = "parent",
    extensions: set[str] | None = None,
    **kwargs,
) -> SmartTabModel:
    """Fit a classification model from ``root/class_name/files`` folders."""
    if label_from != "parent":
        raise ConfigurationError("fit_folder currently supports label_from='parent'")
    root = Path(folder).expanduser()
    if not root.exists() or not root.is_dir():
        raise DataValidationError(f"media folder not found: {root}")
    defaults = {
        "image": {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"},
        "audio": {".wav", ".flac", ".mp3", ".ogg", ".m4a", ".aac"},
        "video": {".mp4", ".avi", ".mov", ".mkv", ".webm", ".mpeg", ".mpg"},
    }
    if modality not in defaults:
        raise ConfigurationError("fit_folder modality must be 'image', 'audio', or 'video'")
    allowed = {suffix.lower() for suffix in (extensions or defaults[modality])}
    paths = sorted(path for path in root.rglob("*") if path.is_file() and path.suffix.lower() in allowed)
    if not paths:
        raise DataValidationError(f"no supported {modality} files found under {root}")
    labels = [path.parent.name for path in paths]
    return fit(paths, y=labels, modality=modality, task_type="multiclass" if len(set(labels)) > 2 else "binary", **kwargs)


def _prepare_fit_input(
    data,
    *,
    target: str | list[str] | None,
    y,
    modality: str,
    modalities: dict[str, str] | str | None,
) -> tuple[pd.DataFrame | str | Path, str | list[str], dict[str, str] | str | None]:
    if modality != "auto" and modalities not in (None, "auto"):
        raise ConfigurationError("use either modality=... for one raw input column or modalities={...}, not both")

    # An explicit raw modality takes precedence over path-like tabular loading.
    # This makes ``fit_images("photo.jpg", [1])`` and equivalent scalar helpers
    # behave as raw samples instead of trying to parse media as a DataFrame.
    explicit_raw = modality != "auto" and y is not None and not isinstance(data, pd.DataFrame)

    if not explicit_raw and isinstance(data, (pd.DataFrame, str, Path)) and not (
        isinstance(data, (str, Path)) and Path(data).is_dir()
    ):
        frame = load_data(data)
        if y is None:
            if target is None:
                raise ConfigurationError("target=... is required for DataFrame/file input")
            return frame, target, modalities if modalities is not None else (modality if modality != "auto" else "auto")
        if target is not None:
            raise ConfigurationError("provide either target=... inside the DataFrame or y=..., not both")
        frame, resolved_target = _append_external_target(frame, y)
        return frame, resolved_target, modalities if modalities is not None else (modality if modality != "auto" else "auto")

    if y is None:
        raise ConfigurationError("raw text/image/audio/video input requires y=...")
    samples = _coerce_raw_samples(data, y, modality)
    frame = pd.DataFrame({"input": samples})
    frame, resolved_target = _append_external_target(frame, y)
    declaration = {"input": modality} if modality != "auto" else (modalities or "auto")
    return frame, resolved_target, declaration


def _coerce_raw_samples(data, y, modality: str) -> list:
    target_length = len(y) if hasattr(y, "__len__") and not isinstance(y, (str, bytes)) else 1
    if isinstance(data, np.ndarray):
        if modality == "image" and data.ndim in (2, 3) and target_length == 1:
            return [data]
        if modality == "video" and data.ndim == 4 and target_length == 1:
            return [data]
        if modality == "audio" and data.ndim in (1, 2) and target_length == 1:
            return [data]
        samples = list(data)
    elif isinstance(data, (list, tuple)):
        if modality == "audio" and len(data) == 2 and isinstance(data[0], (int, float)) and target_length == 1:
            return [data]
        samples = list(data)
    else:
        samples = [data]
    if len(samples) != target_length:
        raise DataValidationError(f"raw input has {len(samples)} samples but y has {target_length} rows")
    return samples


def _append_external_target(frame: pd.DataFrame, y) -> tuple[pd.DataFrame, str | list[str]]:
    result = frame.reset_index(drop=True).copy()
    array = np.asarray(y)
    if array.ndim == 0:
        array = array.reshape(1)
    if len(array) != len(result):
        raise DataValidationError(f"X has {len(result)} rows but y has {len(array)} rows")
    if array.ndim == 1:
        name = "__target__"
        result[name] = array
        return result, name
    if array.ndim == 2:
        names = [f"__target_{index}__" for index in range(array.shape[1])]
        for index, name in enumerate(names):
            result[name] = array[:, index]
        return result, names
    raise DataValidationError("y must be one- or two-dimensional")

def load(path: str | Path, *, trusted: bool = False) -> SmartTabModel:
    bundle = load_bundle(path, trusted=trusted)
    feature_importance = get_feature_importance(
        bundle["estimator"],
        bundle["model_name"],
        bundle["feature_names"],
    )
    return SmartTabModel(
        model_name=bundle["model_name"],
        task_type=bundle["task_type"],
        estimator=bundle["estimator"],
        cleaning_pipeline=bundle["cleaning_pipeline"],
        target_encoder=bundle["target_encoder"],
        raw_feature_names=bundle.get(
            "raw_feature_names",
            list(bundle["cleaning_pipeline"].raw_feature_columns_),
        ),
        feature_names=bundle["feature_names"],
        cat_features=bundle["cat_features"],
        best_params=bundle["best_params"],
        primary_metric=bundle["primary_metric"],
        metrics=bundle["metrics"],
        feature_importance=feature_importance,
        dataset_profile=bundle["dataset_profile"],
        hardware_profile=bundle["hardware_profile"],
        resource_plan=bundle["resource_plan"],
        class_labels=bundle["class_labels"],
        timings=bundle.get("timings", {}),
        notes=bundle.get("notes", []),
        ensemble_info=bundle["ensemble_info"],
        decision_threshold=bundle["decision_threshold"],
        reject_threshold=bundle["reject_threshold"],
        per_label_thresholds=bundle["per_label_thresholds"],
        objective=bundle["objective"],
        static_charts=bundle.get("static_charts", "auto"),
        probability_calibrator=bundle.get("probability_calibrator"),
        conformal_predictor=bundle.get("conformal_predictor"),
        ood_detector=bundle.get("ood_detector"),
        drift_reference=bundle.get("drift_reference"),
        data_science_config=bundle.get("data_science_config", {}),
        data_quality_report=bundle.get("data_quality_report", {}),
        modality_dropout_info=bundle.get("modality_dropout_info", {}),
    )



def _handle_conflicting_labels(
    df: pd.DataFrame,
    target_columns: list[str],
    group_id: str | None,
    policy: str,
) -> tuple[pd.DataFrame, int]:
    if policy == "keep" or len(df) < 2:
        return df.reset_index(drop=True), 0
    excluded = set(target_columns) | ({group_id} if group_id else set())
    features = [column for column in df.columns if column not in excluded]
    if not features:
        return df.reset_index(drop=True), 0
    comparable = df[features].copy()
    for column in comparable.columns:
        comparable[column] = comparable[column].map(
            lambda value: (
                ("ndarray", np.asarray(value).shape, str(np.asarray(value).dtype), hash(np.asarray(value).tobytes()))
                if isinstance(value, np.ndarray)
                else repr(value) if isinstance(value, (list, tuple, dict, bytearray, memoryview))
                else value
            )
        )
    keys = pd.util.hash_pandas_object(comparable, index=False)
    target_view = df[target_columns].copy()
    target_view["__feature_key__"] = keys.to_numpy()
    conflicts = target_view.groupby("__feature_key__", sort=False)[target_columns].nunique(dropna=False)
    conflict_keys = conflicts.index[(conflicts > 1).any(axis=1)]
    if len(conflict_keys) == 0:
        return df.reset_index(drop=True), 0
    mask = keys.isin(conflict_keys)
    count = int(mask.sum())
    if policy == "error":
        raise DataValidationError(
            f"{count} row(s) share identical features but have conflicting target labels; "
            "set data_science={'conflicting_labels': 'drop'} only after confirming the source problem"
        )
    if policy == "drop":
        remaining = df.loc[~mask].reset_index(drop=True)
        if len(remaining) < max(10, int(0.2 * len(df))):
            raise DataValidationError("dropping conflicting labels would leave too little data")
        return remaining, count
    return df.reset_index(drop=True), 0

def _handle_missing_targets(
    df: pd.DataFrame,
    target_columns: list[str],
    policy: str,
) -> tuple[pd.DataFrame, int]:
    missing_columns = [column for column in target_columns if column not in df.columns]
    if missing_columns:
        raise DataValidationError(f"target column(s) not found: {missing_columns}")
    mask = df[target_columns].isna().any(axis=1)
    count = int(mask.sum())
    if count and policy == "error":
        raise DataValidationError(
            f"{count} row(s) contain missing target values; use target_missing='drop' to remove them"
        )
    if count:
        return df.loc[~mask].reset_index(drop=True), count
    return df.reset_index(drop=True), 0


def _handle_duplicate_rows(df: pd.DataFrame, policy: str) -> tuple[pd.DataFrame, int]:
    try:
        duplicate_mask = df.duplicated(keep="first")
    except TypeError:
        comparable = df.copy()
        for column in comparable.columns:
            comparable[column] = comparable[column].map(
                lambda value: repr(value) if isinstance(value, (np.ndarray, list, tuple, dict)) else value
            )
        duplicate_mask = comparable.duplicated(keep="first")
    count = int(duplicate_mask.sum())
    if count and policy == "error":
        raise DataValidationError(f"data contains {count} exact duplicate row(s)")
    if count and policy == "drop":
        return df.loc[~duplicate_mask].reset_index(drop=True), count
    return df.reset_index(drop=True), 0


def _split_train_test(
    df: pd.DataFrame,
    task_type: TaskType,
    target_columns: list[str],
    config: FitConfig,
):
    _validate_target_support_before_split(df, task_type, target_columns, config.test_size)
    strategy = config.split_strategy
    if strategy == "auto":
        if task_type.is_ranking or config.group_id is not None:
            strategy = "group"
        elif config.time_column is not None:
            strategy = "temporal"
        else:
            strategy = "random"

    if strategy == "temporal":
        if config.time_column not in df.columns:
            raise DataValidationError(f"time_column {config.time_column!r} not found")
        parsed = pd.to_datetime(df[config.time_column], errors="coerce", format="mixed")
        if parsed.isna().any():
            raise DataValidationError("time_column contains missing or unparseable values")
        ordered = df.assign(__smarttab_time=parsed).sort_values("__smarttab_time", kind="stable")
        ordered = ordered.drop(columns="__smarttab_time").reset_index(drop=True)
        split_at = int(round(len(ordered) * (1.0 - config.test_size)))
        split_at = min(max(split_at, 1), len(ordered) - 1)
        train_df = ordered.iloc[:split_at].reset_index(drop=True)
        test_df = ordered.iloc[split_at:].reset_index(drop=True)
        return _finalize_split(
            train_df, test_df, task_type, target_columns, config.group_id, strategy
        )

    if strategy in ("group", "stratified_group"):
        if config.group_id is None:
            raise ConfigurationError(f"split_strategy={strategy!r} requires group_id")
        groups = df[config.group_id].to_numpy()
        if len(np.unique(groups)) < 2:
            raise DataValidationError("group split requires at least two distinct groups")
        if strategy == "stratified_group" and task_type in (TaskType.BINARY, TaskType.MULTICLASS):
            n_splits = max(2, int(round(1.0 / config.test_size)))
            splitter = StratifiedGroupKFold(
                n_splits=min(n_splits, len(np.unique(groups))),
                shuffle=True,
                random_state=config.random_state,
            )
            train_index, test_index = next(
                splitter.split(df, df[target_columns[0]], groups=groups)
            )
        else:
            splitter = GroupShuffleSplit(
                n_splits=1,
                test_size=config.test_size,
                random_state=config.random_state,
            )
            train_index, test_index = next(splitter.split(df, groups=groups))
        train_df = df.iloc[train_index].reset_index(drop=True)
        test_df = df.iloc[test_index].reset_index(drop=True)
        return _finalize_split(
            train_df, test_df, task_type, target_columns, config.group_id, strategy
        )

    if strategy != "random":
        raise ConfigurationError(f"unsupported split strategy {strategy!r}")
    stratify = None
    if task_type in (TaskType.BINARY, TaskType.MULTICLASS):
        counts = df[target_columns[0]].value_counts()
        if len(counts) >= 2 and counts.min() >= 2:
            stratify = df[target_columns[0]]
    try:
        train_df, test_df = train_test_split(
            df,
            test_size=config.test_size,
            random_state=config.random_state,
            stratify=stratify,
        )
    except ValueError as exc:
        raise DataValidationError(
            f"failed to create a statistically valid {strategy} holdout split: {exc}"
        ) from exc
    return _finalize_split(
        train_df.reset_index(drop=True),
        test_df.reset_index(drop=True),
        task_type,
        target_columns,
        config.group_id,
        strategy,
    )


def _validate_target_support_before_split(
    df: pd.DataFrame,
    task_type: TaskType,
    target_columns: list[str],
    test_size: float,
) -> None:
    """Reject targets that cannot produce a defensible training/holdout split."""
    if task_type in (TaskType.BINARY, TaskType.MULTICLASS):
        counts = df[target_columns[0]].value_counts(dropna=False)
        rare = counts[counts < 2]
        if not rare.empty:
            raise DataValidationError(
                "classification requires at least two rows per class before splitting; "
                f"insufficient classes: {rare.to_dict()}"
            )
        n_classes = len(counts)
        n_test = int(np.ceil(len(df) * test_size))
        n_train = len(df) - n_test
        if min(n_train, n_test) < n_classes:
            raise DataValidationError(
                f"test_size={test_size} cannot place all {n_classes} classes in both train and holdout; "
                "increase the dataset size or change test_size"
            )
    elif task_type is TaskType.MULTILABEL:
        failures: dict[str, dict] = {}
        for column in target_columns:
            counts = df[column].value_counts(dropna=False)
            if len(counts) != 2 or int(counts.min()) < 2:
                failures[column] = counts.to_dict()
        if failures:
            raise DataValidationError(
                "each multilabel target must contain exactly two classes with at least two rows each; "
                f"invalid targets: {failures}"
            )


def _finalize_split(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    task_type: TaskType,
    target_columns: list[str],
    group_id: str | None,
    strategy: str,
):
    if train_df.empty or test_df.empty:
        raise DataValidationError(f"{strategy} split produced an empty train or holdout partition")

    if task_type in (TaskType.BINARY, TaskType.MULTICLASS):
        source_classes = set(pd.concat([train_df[target_columns[0]], test_df[target_columns[0]]]).unique())
        train_classes = set(train_df[target_columns[0]].unique())
        missing = sorted(source_classes - train_classes, key=str)
        if missing:
            raise DataValidationError(
                f"{strategy} split placed class(es) {missing} only in the holdout. "
                "Use a stratified-compatible split, revise group/time boundaries, or collect more examples."
            )
    elif task_type is TaskType.MULTILABEL:
        invalid = []
        for column in target_columns:
            if train_df[column].nunique(dropna=False) != 2:
                invalid.append(column)
        if invalid:
            raise DataValidationError(
                f"{strategy} split left multilabel target(s) without both classes in training: {invalid}"
            )

    return _with_groups(train_df, test_df, group_id)


def _with_groups(train_df: pd.DataFrame, test_df: pd.DataFrame, group_id: str | None):
    group_train = train_df[group_id].to_numpy() if group_id is not None else None
    group_test = test_df[group_id].to_numpy() if group_id is not None else None
    return train_df, test_df, group_train, group_test


def _extract_targets(train_df: pd.DataFrame, test_df: pd.DataFrame, profile: DatasetProfile):
    if profile.task_type in (TaskType.BINARY, TaskType.MULTICLASS):
        encoder = TargetLabelEncoder().fit(train_df[profile.target_column])
        return (
            encoder,
            encoder.transform(train_df[profile.target_column]),
            encoder.transform(test_df[profile.target_column]),
            encoder.classes_.tolist(),
        )
    if profile.task_type is TaskType.MULTILABEL:
        encoder = MultiLabelTargetEncoder().fit(train_df[profile.target_columns])
        return (
            encoder,
            encoder.transform(train_df[profile.target_columns]),
            encoder.transform(test_df[profile.target_columns]),
            encoder.classes_,
        )
    if profile.task_type is TaskType.MULTIOUTPUT_REGRESSION:
        return (
            None,
            train_df[profile.target_columns].to_numpy(dtype=float),
            test_df[profile.target_columns].to_numpy(dtype=float),
            None,
        )
    return (
        None,
        train_df[profile.target_column].to_numpy(dtype=float),
        test_df[profile.target_column].to_numpy(dtype=float),
        None,
    )


def _train_single_model(
    model_name: str,
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    profile: DatasetProfile,
    resource_plan,
    cat_features: list[str],
    group_train: np.ndarray | None,
    config: FitConfig,
    deadline: FitDeadline,
):
    notes: list[str] = []
    class_weights = resolve_class_weight_params(model_name, profile)
    if config.params is not None:
        params = {**class_weights, **config.params}
        primary_metric = config.metrics if config.metrics != "auto" else _default_metric_name(profile.task_type)
        n_estimators = _resolve_n_estimators(config.n_estimators, profile, model_name)
        notes.append("explicit params supplied; optimization skipped")
    elif config.optimize:
        search_timeout = _search_timeout(config, deadline, ensemble=False)
        if search_timeout is not None and search_timeout <= 0:
            params = {**default_params(model_name), **class_weights}
            primary_metric = _default_metric_name(profile.task_type)
            n_estimators = _resolve_n_estimators(config.n_estimators, profile, model_name)
            notes.append("optimization skipped because no search budget remained")
        else:
            result = run_optimization(
                model_name,
                X_train,
                y_train,
                profile.task_type,
                profile,
                resource_plan,
                cat_features,
                validation=config.validation,
                cv=config.cv,
                optimizer=config.optimizer,
                n_trials=config.n_trials,
                timeout=search_timeout,
                metrics=config.metrics,
                random_state=config.random_state,
                verbose=config.verbose,
                group_ids=group_train if profile.task_type.is_ranking else None,
                deadline=deadline,
            )
            params = result.best_params
            primary_metric = result.primary_metric
            n_estimators = (
                int(config.n_estimators)
                if config.n_estimators != "auto"
                else result.best_n_estimators
            )
            notes.extend(result.notes)
    else:
        params = {**default_params(model_name), **class_weights}
        primary_metric = config.metrics if config.metrics != "auto" else _default_metric_name(profile.task_type)
        n_estimators = _resolve_n_estimators(config.n_estimators, profile, model_name)
        notes.append(
            f"optimize=False: zero search trials; fitting one {model_name} model with {n_estimators} trees"
        )

    estimator = build_estimator(
        model_name,
        params,
        profile.task_type,
        n_estimators,
        resource_plan.cpu_threads,
        resource_plan.use_gpu,
        config.random_state,
        resource_plan=resource_plan,
    )
    started = time.perf_counter()
    with PeakMemoryTracker() as tracker:
        if profile.task_type.is_ranking:
            X_sorted, y_sorted, groups_sorted = sort_by_group(X_train, y_train, group_train)
            fit_estimator(
                estimator,
                model_name,
                X_sorted,
                y_sorted,
                cat_features=cat_features,
                group_ids_train=groups_sorted,
                deadline=deadline,
                resource_plan=resource_plan,
            )
        else:
            fit_estimator(
                estimator,
                model_name,
                X_train,
                y_train,
                cat_features=cat_features,
                deadline=deadline,
                resource_plan=resource_plan,
            )
    training_seconds = time.perf_counter() - started
    return (
        estimator,
        params,
        primary_metric,
        training_seconds,
        {
            "peak_training_memory_mb": tracker.peak_mb,
            "training_memory_delta_mb": tracker.delta_mb,
        },
        notes,
    )




def _drop_low_quality_training_rows(
    train_df: pd.DataFrame,
    target_columns: list[str],
    group_id: str | None,
    group_train: np.ndarray | None,
    threshold: float,
) -> tuple[pd.DataFrame, np.ndarray | None, int]:
    excluded = set(target_columns) | ({group_id} if group_id else set())
    features = [column for column in train_df.columns if column not in excluded]
    if not features or threshold >= 1.0:
        return train_df, group_train, 0
    missing_fraction = train_df[features].isna().mean(axis=1)
    keep = missing_fraction < float(threshold)
    removed = int((~keep).sum())
    if removed == 0:
        return train_df, group_train, 0
    if int(keep.sum()) < max(10, int(0.2 * len(train_df))):
        raise DataValidationError(
            "row_missing_threshold would remove too much training data; repair the source dataset "
            "or increase data_science.row_missing_threshold"
        )
    groups = group_train[keep.to_numpy()] if group_train is not None else None
    return train_df.loc[keep].reset_index(drop=True), groups, removed


def _enforce_quality_policy(report, policy: str) -> None:
    if policy != "strict":
        return
    blocking_codes = {"conflicting_duplicate_labels", "unreadable_media_paths"}
    blocking = [issue for issue in report.issues if issue.code in blocking_codes]
    if blocking:
        messages = "; ".join(issue.message for issue in blocking[:5])
        raise DataValidationError(f"strict data-quality policy rejected the training data: {messages}")


def _split_uncertainty_holdout(
    X: pd.DataFrame,
    y: np.ndarray,
    groups: np.ndarray | None,
    task_type: TaskType,
    config: FitConfig,
):
    ds = config.data_science_config
    wants_calibration = task_type.is_classification and ds.calibration != "none"
    wants_conformal = ds.conformal is True or (
        ds.conformal == "auto" and task_type is not TaskType.RANKING and len(X) >= 300
    )
    if not (wants_calibration or wants_conformal) or len(X) < 80:
        return X, y, groups, None, None
    fraction = float(ds.calibration_fraction)
    stratify = y if task_type in {TaskType.BINARY, TaskType.MULTICLASS} else None
    try:
        indices = np.arange(len(X))
        fit_idx, cal_idx = train_test_split(
            indices,
            test_size=fraction,
            random_state=config.random_state + 97,
            stratify=stratify,
        )
    except ValueError:
        return X, y, groups, None, None
    if len(cal_idx) < 20:
        return X, y, groups, None, None
    X_fit = X.iloc[fit_idx].reset_index(drop=True)
    X_cal = X.iloc[cal_idx].reset_index(drop=True)
    y_array = np.asarray(y)
    group_fit = np.asarray(groups)[fit_idx] if groups is not None else None
    return X_fit, y_array[fit_idx], group_fit, X_cal, y_array[cal_idx]


def _augment_modality_dropout(
    X: pd.DataFrame,
    y: np.ndarray,
    groups: np.ndarray | None,
    feature_groups: dict[str, list[str]],
    config: FitConfig,
):
    modality_groups = {
        name: columns for name, columns in feature_groups.items()
        if name.startswith("modality:") and columns
    }
    if not modality_groups:
        return X, y, groups, {"rate": 0.0, "augmented_rows": 0, "modalities": []}
    setting = config.data_science_config.modality_dropout
    if setting == "auto":
        rate = 0.08 if len(modality_groups) > 1 else 0.04
    else:
        rate = float(setting)
    if rate <= 0:
        return X, y, groups, {"rate": 0.0, "augmented_rows": 0, "modalities": sorted(modality_groups)}
    maximum = int(len(X) * config.data_science_config.modality_dropout_max_expansion)
    n_augmented = min(maximum, int(np.ceil(len(X) * rate * len(modality_groups))))
    if n_augmented <= 0:
        return X, y, groups, {"rate": rate, "augmented_rows": 0, "modalities": sorted(modality_groups)}
    rng = np.random.default_rng(config.random_state + 211)
    source_indices = rng.integers(0, len(X), size=n_augmented)
    modality_names = list(modality_groups)
    selected_modalities = rng.integers(0, len(modality_names), size=n_augmented)
    augmented = X.iloc[source_indices].copy().reset_index(drop=True)
    for row_index, modality_index in enumerate(selected_modalities):
        columns = modality_groups[modality_names[int(modality_index)]]
        augmented.loc[row_index, columns] = 0.0
        missing_features = [column for column in columns if column.endswith("source_missing")]
        if missing_features:
            augmented.loc[row_index, missing_features] = 1.0
    result_X = pd.concat([X.reset_index(drop=True), augmented], ignore_index=True)
    y_array = np.asarray(y)
    result_y = np.concatenate([y_array, y_array[source_indices]], axis=0)
    result_groups = None
    if groups is not None:
        group_array = np.asarray(groups)
        result_groups = np.concatenate([group_array, group_array[source_indices]], axis=0)
    return result_X, result_y, result_groups, {
        "rate": rate,
        "augmented_rows": n_augmented,
        "modalities": sorted(modality_groups),
        "max_expansion": config.data_science_config.modality_dropout_max_expansion,
    }


def _fit_uncertainty_models(
    estimator,
    model_name: str,
    task_type: TaskType,
    X_calibration: pd.DataFrame,
    y_calibration: np.ndarray,
    config: FitConfig,
):
    ds = config.data_science_config
    notes: list[str] = []
    calibrator = None
    conformal = None
    if task_type.is_classification:
        raw_proba = np.asarray(predict_proba(estimator, model_name, X_calibration))
        calibration_method = ds.calibration
        if calibration_method == "auto":
            calibration_method = "isotonic" if len(X_calibration) >= 1200 else "sigmoid"
        if calibration_method != "none":
            task_name = {
                TaskType.BINARY: "binary",
                TaskType.MULTICLASS: "multiclass",
                TaskType.MULTILABEL: "multilabel",
            }[task_type]
            try:
                calibrator = ProbabilityCalibrator(task=task_name).fit(
                    y_calibration, raw_proba, calibration_method
                )
                calibrated_probe = calibrator.transform(raw_proba)
                if task_type in {TaskType.BINARY, TaskType.MULTICLASS}:
                    raw_predictions = raw_proba.argmax(axis=1)
                    calibrated_predictions = calibrated_probe.argmax(axis=1)
                    before = evaluate_classification(y_calibration, raw_predictions, raw_proba)
                    after = evaluate_classification(
                        y_calibration, calibrated_predictions, calibrated_probe
                    )
                    calibrator.diagnostics_.update({
                        "ece_before": before.get("expected_calibration_error"),
                        "ece_after": after.get("expected_calibration_error"),
                        "log_loss_before": before.get("log_loss"),
                        "log_loss_after": after.get("log_loss"),
                        "brier_before": before.get("brier_score"),
                        "brier_after": after.get("brier_score"),
                    })
                else:
                    calibrator.diagnostics_.update({
                        "brier_before": float(np.mean((raw_proba - y_calibration) ** 2)),
                        "brier_after": float(np.mean((calibrated_probe - y_calibration) ** 2)),
                    })
                notes.append(
                    f"probability calibration={calibration_method}; calibration_rows={len(X_calibration)}"
                )
            except Exception as exc:
                calibrator = None
                notes.append(f"probability calibration skipped: {type(exc).__name__}: {exc}")
        calibrated = calibrator.transform(raw_proba) if calibrator is not None else raw_proba
        enabled = ds.conformal is True or (ds.conformal == "auto" and len(X_calibration) >= 50)
        if enabled:
            task_name = {
                TaskType.BINARY: "binary",
                TaskType.MULTICLASS: "multiclass",
                TaskType.MULTILABEL: "multilabel",
            }[task_type]
            try:
                conformal = ConformalPredictor(task_name, ds.conformal_alpha).fit(
                    y_calibration, calibrated
                )
                notes.append(f"conformal alpha={ds.conformal_alpha:.3f}")
            except Exception as exc:
                notes.append(f"conformal calibration skipped: {type(exc).__name__}: {exc}")
    elif task_type in {TaskType.REGRESSION, TaskType.MULTIOUTPUT_REGRESSION}:
        enabled = ds.conformal is True or (ds.conformal == "auto" and len(X_calibration) >= 50)
        if enabled:
            predictions = np.asarray(predict(estimator, model_name, X_calibration))
            task_name = "regression" if task_type is TaskType.REGRESSION else "multioutput_regression"
            conformal = ConformalPredictor(task_name, ds.conformal_alpha).fit(
                y_calibration, predictions
            )
            notes.append(f"conformal regression intervals alpha={ds.conformal_alpha:.3f}")
    return calibrator, conformal, notes


def _calibrated_probabilities(estimator, model_name: str, X: pd.DataFrame, calibrator):
    probabilities = np.asarray(predict_proba(estimator, model_name, X))
    return calibrator.transform(probabilities) if calibrator is not None else probabilities


def _uncertainty_holdout_metrics(
    task_type: TaskType,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray | None,
    conformal,
    ood_detector,
    X_test: pd.DataFrame,
) -> dict[str, float]:
    results: dict[str, float] = {}
    if ood_detector is not None:
        scores = np.asarray(ood_detector.score(X_test), dtype=float)
        results.update({
            "ood_score_mean": float(scores.mean()),
            "ood_score_p95": float(np.quantile(scores, 0.95)),
            "ood_flag_rate": float((scores >= ood_detector.score_threshold_).mean()),
        })
    if conformal is None:
        return results
    y_array = np.asarray(y_true)
    if task_type in {TaskType.BINARY, TaskType.MULTICLASS} and y_proba is not None:
        sets = conformal.prediction_set(y_proba)
        coverage = sets[np.arange(len(y_array)), y_array.astype(int)]
        results["conformal_coverage"] = float(coverage.mean())
        results["conformal_set_size_mean"] = float(sets.sum(axis=1).mean())
    elif task_type is TaskType.MULTILABEL and y_proba is not None:
        sets = conformal.multilabel_sets(y_proba)
        positive = sets["positive_possible"]
        negative = sets["negative_possible"]
        covered = np.where(y_array.astype(int) == 1, positive, negative)
        results["conformal_label_coverage"] = float(covered.mean())
        results["conformal_ambiguous_rate"] = float((positive & negative).mean())
    elif task_type in {TaskType.REGRESSION, TaskType.MULTIOUTPUT_REGRESSION}:
        lower, upper = conformal.interval(np.asarray(y_pred))
        covered = (y_array >= lower) & (y_array <= upper)
        results["conformal_coverage"] = float(np.mean(covered))
        results["conformal_interval_width_mean"] = float(np.mean(upper - lower))
    return results

def _format_metric_for_log(value: Any) -> Any:
    """Return a compact, logging-safe representation for scalar or vector metrics."""
    if isinstance(value, (int, float, np.integer, np.floating)):
        return round(float(value), 5)
    if isinstance(value, np.ndarray):
        return [round(float(item), 5) for item in value.ravel().tolist()]
    if isinstance(value, (list, tuple)):
        return [
            round(float(item), 5) if isinstance(item, (int, float, np.integer, np.floating)) else item
            for item in value
        ]
    return value

def _evaluate_task(task_type, y_true, y_pred, y_proba, groups):
    if task_type in (TaskType.BINARY, TaskType.MULTICLASS):
        return evaluate_classification(y_true, y_pred, y_proba)
    if task_type is TaskType.MULTILABEL:
        return evaluate_multilabel(y_true, y_pred, y_proba)
    if task_type is TaskType.MULTIOUTPUT_REGRESSION:
        return evaluate_multioutput_regression(y_true, y_pred)
    if task_type.is_ranking:
        return evaluate_ranking(y_true, y_pred, groups)
    return evaluate_regression(y_true, y_pred)


def _resolve_n_estimators(value: str | int, profile: DatasetProfile, model_name: str) -> int:
    if value != "auto":
        return int(value)
    if profile.n_samples >= 250_000:
        return 180 if model_name == "lightgbm" else 120
    if profile.n_samples >= 75_000:
        return 250
    if profile.n_samples < 2_000:
        return 350
    return 300


def _model_n_estimators(estimator, model_name: str, configured, profile: DatasetProfile) -> int:
    if configured != "auto":
        return int(configured)
    if model_name == "catboost" and hasattr(estimator, "tree_count_"):
        return int(estimator.tree_count_)
    if model_name == "lightgbm" and hasattr(estimator, "n_estimators_"):
        return int(estimator.n_estimators_)
    return _resolve_n_estimators("auto", profile, model_name)


def _default_metric_name(task_type: TaskType) -> str:
    if task_type in (TaskType.REGRESSION, TaskType.MULTIOUTPUT_REGRESSION):
        return "rmse"
    if task_type.is_ranking:
        return "ndcg"
    if task_type is TaskType.BINARY:
        return "roc_auc"
    return "f1_macro"


def _search_timeout(config: FitConfig, deadline: FitDeadline, *, ensemble: bool) -> float | None:
    if not deadline.enabled:
        return config.timeout
    remaining = deadline.remaining() or 0.0
    reserve_fraction = 0.55 if ensemble else 0.35
    reserve = max(3.0, remaining * reserve_fraction)
    available = max(0.0, remaining - reserve)
    return min(available, config.timeout) if config.timeout is not None else available


def _has_optional_time(deadline: FitDeadline, minimum: float) -> bool:
    remaining = deadline.remaining()
    return remaining is None or remaining >= minimum
