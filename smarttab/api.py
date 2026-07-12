"""``smarttab.fit()`` / ``smarttab.load()`` — the only two top-level entry points.

``fit()`` wires the nine pipeline stages together in order. Every stage's
own module resolves its own "auto" behavior; this function's job is purely
orchestration, not decision-making. Task type is auto-detected from the
shape of ``target`` (and ``group_id``, for ranking) — see
``analysis/dataset_analyzer.py``.
"""

from __future__ import annotations

import time

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit, train_test_split

from smarttab.analysis.dataset_analyzer import TaskType, analyze_dataset
from smarttab.cleaning.encoders import TargetLabelEncoder
from smarttab.cleaning.outliers import compute_outlier_keep_mask
from smarttab.cleaning.pipeline import SmartCleaningPipeline
from smarttab.config import FitConfig, load_data
from smarttab.evaluation.evaluator import (
    evaluate_classification,
    evaluate_multilabel,
    evaluate_multioutput_regression,
    evaluate_ranking,
    evaluate_regression,
)
from smarttab.exceptions import ConfigurationError
from smarttab.explainability.explainer import get_feature_importance, get_shap_importance
from smarttab.hardware.profiler import profile_hardware
from smarttab.hardware.resource_planner import resolve_resource_plan
from smarttab.logging_utils import get_logger, set_verbosity
from smarttab.model import SmartTabModel
from smarttab.optimization.optimizer import resolve_class_weight_params, run_optimization
from smarttab.optimization.search_spaces import default_params
from smarttab.optimization.threshold import (
    DEFAULT_REJECT_THRESHOLD,
    DEFAULT_THRESHOLD,
    apply_threshold,
    probe_and_optimize_multilabel_thresholds,
    probe_and_optimize_threshold,
)
from smarttab.persistence.serializer import load_bundle
from smarttab.selection.model_selector import select_model
from smarttab.training.ensemble import run_ensemble_decision_engine, train_voting_stacking_ensemble
from smarttab.training.trainer import (
    DEFAULT_EARLY_STOPPING_ROUNDS,
    PeakMemoryTracker,
    build_estimator,
    fit_estimator,
    get_best_iteration,
    predict,
    predict_proba,
    sort_by_group,
    verify_gpu_usable,
)

FALLBACK_N_ESTIMATORS = 300
DISCOVERY_N_ESTIMATORS_CAP = 300

# time_limit budgeting (AutoGluon-style global wall-clock budget for the whole fit() call).
TIME_LIMIT_RESERVE_FRACTION = 0.15  # kept back for final training/eval/threshold-tuning/report
TIGHT_BUDGET_SECONDS = 120  # below this, cv is forced down so more trials fit in the budget
TIME_BUDGET_N_TRIALS_CAP = 200  # let the timeout (not trial count) be the real gate when time-boxed

ENSEMBLE_BASE_MODELS = ("catboost", "lightgbm", "xgboost")
ENSEMBLE_SUPPORTED_TASK_TYPES = (TaskType.BINARY, TaskType.MULTICLASS, TaskType.REGRESSION)


def fit(data, target: str | list[str], group_id: str | None = None, **kwargs) -> SmartTabModel:
    fit_start = time.perf_counter()
    config = FitConfig(target=target, group_id=group_id, **kwargs)
    set_verbosity(config.verbose)
    logger = get_logger()

    df = load_data(data)

    logger.info("Stage 1/9: analyzing dataset...")
    profile = analyze_dataset(df, config.target, config.group_id)
    logger.info(
        "Detected task_type=%s, n_samples=%d, n_features=%d, n_targets=%d%s",
        profile.task_type.value, profile.n_samples, profile.n_features, len(profile.target_columns),
        f", n_groups={profile.n_groups}" if profile.task_type.is_ranking else "",
    )
    if profile.task_type.is_classification and profile.is_imbalanced:
        logger.info(
            "Class imbalance detected (minority/majority ratio=%.3f): applying balanced class "
            "weights and preferring %s over raw accuracy so the reported score isn't inflated.",
            profile.class_imbalance_ratio, "ROC AUC/F1" if profile.task_type is TaskType.BINARY else "macro-F1",
        )

    if config.ensemble != "none" and profile.task_type not in ENSEMBLE_SUPPORTED_TASK_TYPES:
        raise ConfigurationError(
            f"ensemble={config.ensemble!r} is only supported for binary/multiclass/regression tasks; "
            f"detected task_type={profile.task_type.value!r}. Use ensemble='none' for this task."
        )

    if config.outlier == "remove":
        keep_mask = compute_outlier_keep_mask(df, profile.numeric_columns)
        removed = int((~keep_mask).sum())
        if removed:
            logger.info("outlier='remove': dropping %d outlier row(s)", removed)
            df = df.loc[keep_mask].reset_index(drop=True)
            profile = analyze_dataset(df, config.target, config.group_id)

    logger.info("Stage 3/9: profiling hardware...")
    hardware = profile_hardware()
    resource_plan = resolve_resource_plan(
        hardware, profile, device=config.device, cpu_threads=config.cpu_threads,
        gpu_memory=config.gpu_memory, ram_limit=config.ram_limit,
    )
    for note in resource_plan.notes:
        logger.debug(note)

    logger.info("Stage 4/9: selecting model...")
    is_ensemble_mode = config.ensemble != "none"
    if is_ensemble_mode:
        model_name = config.ensemble  # placeholder; replaced once training resolves the final strategy
        model_notes = [f"ensemble={config.ensemble!r}: training base learners instead of a single selected model"]
        models_to_verify = ENSEMBLE_BASE_MODELS
    else:
        model_name, model_notes = select_model(profile, model=config.model)
        models_to_verify = (model_name,)
    for note in model_notes:
        logger.info(note)

    if resource_plan.use_gpu:
        resource_plan.use_gpu = all(
            verify_gpu_usable(name, profile.task_type, resource_plan.cpu_threads, config.random_state)
            for name in models_to_verify
        )

    train_df, test_df, group_train, group_test = _split_train_test(df, profile, config)

    logger.info("Stage 2/9: cleaning data...")
    pipeline = SmartCleaningPipeline(
        missing=config.missing, categorical=config.categorical,
        scaling=config.scaling, feature_selection=config.feature_selection,
    )
    X_train = pipeline.fit_transform(train_df, train_df[profile.target_columns], profile)
    X_test = pipeline.transform(test_df.drop(columns=profile.target_columns, errors="ignore"))
    cat_features = pipeline.final_categorical_columns

    target_encoder, y_train, y_test, class_labels = _extract_targets(train_df, test_df, profile)

    effective_timeout, effective_n_trials, effective_cv, time_notes = _resolve_time_budget(config, fit_start)
    notes: list[str] = list(model_notes) + list(resource_plan.notes) + time_notes
    ensemble_info: dict | None = None
    decision_threshold = DEFAULT_THRESHOLD
    reject_threshold = DEFAULT_REJECT_THRESHOLD
    per_label_thresholds: list[float] | None = None
    threshold_ladder: list[dict] | None = None

    if config.ensemble in ("voting", "stacking"):
        logger.info("Stage 5-6/9: training %s ensemble (optimize=%s)...", config.ensemble, config.optimize)
        train_start = time.perf_counter()
        with PeakMemoryTracker() as mem_tracker:
            ensemble_result = train_voting_stacking_ensemble(
                X_train, y_train, profile.task_type, profile, resource_plan, cat_features,
                validation=config.validation, cv=effective_cv, optimizer=config.optimizer,
                n_trials=effective_n_trials, timeout=effective_timeout, metrics=config.metrics,
                optimize=config.optimize, random_state=config.random_state, verbose=config.verbose,
                strategy=config.ensemble, threshold_optimization=config.threshold_optimization,
                objective=config.objective, multi_threshold_ensemble=config.multi_threshold_ensemble,
                threshold_models=config.threshold_models,
            )
        training_seconds = time.perf_counter() - train_start
        estimator = ensemble_result.estimator
        model_name = ensemble_result.strategy
        best_params = ensemble_result.base_params
        primary_metric = ensemble_result.primary_metric
        decision_threshold = ensemble_result.decision_threshold
        reject_threshold = ensemble_result.reject_threshold
        threshold_ladder = ensemble_result.threshold_ladder
        notes.extend(ensemble_result.notes)
        ensemble_info = {
            "strategy": ensemble_result.strategy, "validation_score": ensemble_result.validation_score,
            "base_params": ensemble_result.base_params, "base_n_estimators": ensemble_result.base_n_estimators,
            "base_scores": ensemble_result.base_scores,
        }
    elif config.ensemble == "auto":
        logger.info("Stage 4-6/9: running ensemble decision engine (optimize=%s)...", config.optimize)
        train_start = time.perf_counter()
        with PeakMemoryTracker() as mem_tracker:
            decision = run_ensemble_decision_engine(
                X_train, y_train, profile.task_type, profile, resource_plan, cat_features,
                validation=config.validation, cv=effective_cv, optimizer=config.optimizer,
                n_trials=effective_n_trials, timeout=effective_timeout, metrics=config.metrics,
                optimize=config.optimize, random_state=config.random_state, verbose=config.verbose,
                threshold_optimization=config.threshold_optimization, objective=config.objective,
                multi_threshold_ensemble=config.multi_threshold_ensemble, threshold_models=config.threshold_models,
            )
        training_seconds = time.perf_counter() - train_start
        estimator = decision.estimator
        model_name = decision.strategy
        best_params = decision.best_params
        primary_metric = decision.primary_metric
        decision_threshold = decision.decision_threshold
        reject_threshold = decision.reject_threshold
        threshold_ladder = decision.threshold_ladder
        notes.extend(decision.notes)
        ensemble_info = decision.ensemble_info
    else:
        logger.info("Stage 5/9: hyperparameter optimization (optimize=%s)...", config.optimize)
        class_weight_params = resolve_class_weight_params(model_name, profile)

        if config.params is not None:
            best_params = {**class_weight_params, **dict(config.params)}
            primary_metric = config.metrics if config.metrics != "auto" else _default_metric_name(profile.task_type)
            best_n_estimators = _discover_n_estimators(model_name, best_params, profile, resource_plan, X_train, y_train, cat_features, config.random_state, group_train)
            notes.append("params were explicitly provided; skipping hyperparameter search")
        elif config.optimize:
            opt_result = run_optimization(
                model_name=model_name, X=X_train, y=y_train, task_type=profile.task_type, profile=profile,
                resource_plan=resource_plan, cat_features=cat_features, validation=config.validation,
                cv=effective_cv, optimizer=config.optimizer, n_trials=effective_n_trials, timeout=effective_timeout,
                metrics=config.metrics, random_state=config.random_state, verbose=config.verbose,
                group_ids=group_train,
            )
            best_params = opt_result.best_params
            primary_metric = opt_result.primary_metric
            best_n_estimators = opt_result.best_n_estimators
            notes.extend(opt_result.notes)
        else:
            best_params = {**class_weight_params, **default_params(model_name)}
            primary_metric = config.metrics if config.metrics != "auto" else _default_metric_name(profile.task_type)
            best_n_estimators = _discover_n_estimators(model_name, best_params, profile, resource_plan, X_train, y_train, cat_features, config.random_state, group_train)
            notes.append("optimize=False; using default hyperparameters")

        logger.info("Stage 6/9: training final model (%s, n_estimators=%d)...", model_name, best_n_estimators)
        estimator = build_estimator(
            model_name, best_params, profile.task_type, n_estimators=best_n_estimators,
            cpu_threads=resource_plan.cpu_threads, use_gpu=resource_plan.use_gpu, random_state=config.random_state,
        )
        train_start = time.perf_counter()
        with PeakMemoryTracker() as mem_tracker:
            if profile.task_type.is_ranking:
                X_train_sorted, y_train_sorted, group_train_sorted = sort_by_group(X_train, y_train, group_train)
                fit_estimator(estimator, model_name, X_train_sorted, y_train_sorted, cat_features=cat_features, group_ids_train=group_train_sorted)
            else:
                fit_estimator(estimator, model_name, X_train, y_train, cat_features=cat_features)
        training_seconds = time.perf_counter() - train_start

        if config.threshold_optimization and profile.task_type in (TaskType.BINARY, TaskType.MULTICLASS):
            threshold, threshold_ladder = probe_and_optimize_threshold(
                model_name, best_params, best_n_estimators, profile.task_type, resource_plan,
                cat_features, X_train, y_train, config.random_state, objective=config.objective,
                build_ladder=config.multi_threshold_ensemble, n_ladder_models=config.threshold_models,
            )
            if profile.task_type is TaskType.MULTICLASS:
                reject_threshold = threshold
                notes.append(f"reject_threshold={reject_threshold:.2f} (objective={config.objective})")
            else:
                decision_threshold = threshold
                notes.append(f"decision_threshold={decision_threshold:.2f} (objective={config.objective})")
            if config.multi_threshold_ensemble:
                notes.append(f"multi_threshold_ensemble: built a {config.threshold_models}-point threshold ladder")
        elif config.threshold_optimization and profile.task_type is TaskType.MULTILABEL:
            per_label_thresholds, threshold_ladder = probe_and_optimize_multilabel_thresholds(
                model_name, best_params, best_n_estimators, profile.task_type, resource_plan,
                cat_features, X_train, y_train, config.random_state, objective=config.objective,
                build_ladder=config.multi_threshold_ensemble, n_ladder_models=config.threshold_models,
            )
            notes.append(f"per_label_thresholds={[round(t, 2) for t in per_label_thresholds]} (objective={config.objective})")
            if config.multi_threshold_ensemble:
                notes.append(f"multi_threshold_ensemble: built a {config.threshold_models}-point threshold ladder per label")

    logger.info("Stage 7/9: evaluating on held-out test set...")
    predict_start = time.perf_counter()
    y_proba = None
    if profile.task_type is TaskType.BINARY:
        y_proba = predict_proba(estimator, model_name, X_test)
        y_pred = apply_threshold(y_proba[:, 1], decision_threshold)
    elif profile.task_type in (TaskType.MULTICLASS, TaskType.MULTILABEL):
        y_proba = predict_proba(estimator, model_name, X_test)
        if profile.task_type is TaskType.MULTILABEL and per_label_thresholds is not None:
            # apply the per-label optimized thresholds instead of each library's built-in 0.5
            # cutoff — same philosophy as binary's decision_threshold above.
            y_pred = np.stack(
                [apply_threshold(y_proba[:, i], per_label_thresholds[i]) for i in range(y_proba.shape[1])], axis=1,
            )
        else:
            # multiclass reject_threshold intentionally does NOT affect this internal
            # evaluation — it only kicks in via predict() when multi_threshold_ensemble=True,
            # so held-out metrics stay comparable to a plain-argmax model unless opted into.
            y_pred = predict(estimator, model_name, X_test)
    else:
        y_pred = predict(estimator, model_name, X_test)
    prediction_seconds = time.perf_counter() - predict_start

    if profile.task_type in (TaskType.BINARY, TaskType.MULTICLASS):
        metrics = evaluate_classification(y_test, y_pred, y_proba)
    elif profile.task_type is TaskType.MULTILABEL:
        metrics = evaluate_multilabel(y_test, y_pred)
    elif profile.task_type is TaskType.MULTIOUTPUT_REGRESSION:
        metrics = evaluate_multioutput_regression(y_test, y_pred)
    elif profile.task_type.is_ranking:
        metrics = evaluate_ranking(y_test, y_pred, group_test)
    else:
        metrics = evaluate_regression(y_test, y_pred)
    logger.info("metrics: %s", {k: round(v, 4) for k, v in metrics.items() if isinstance(v, float)})

    if config.explain:
        feature_importance = get_feature_importance(estimator, model_name, list(X_train.columns))
        shap_importance = get_shap_importance(estimator, model_name, X_train)
    else:
        feature_importance = pd.DataFrame(columns=["feature", "importance"])
        shap_importance = None

    timings = {
        "training_seconds": training_seconds,
        "peak_training_memory_mb": mem_tracker.peak_mb,
        "training_memory_delta_mb": mem_tracker.delta_mb,
    }

    model = SmartTabModel(
        model_name=model_name, task_type=profile.task_type, estimator=estimator,
        cleaning_pipeline=pipeline, target_encoder=target_encoder, feature_names=list(X_train.columns),
        cat_features=cat_features, best_params=best_params, primary_metric=primary_metric, metrics=metrics,
        feature_importance=feature_importance, dataset_profile=profile, hardware_profile=hardware,
        resource_plan=resource_plan, class_labels=class_labels, timings=timings, notes=notes,
        shap_importance=shap_importance, ensemble_info=ensemble_info, decision_threshold=decision_threshold,
        reject_threshold=reject_threshold, per_label_thresholds=per_label_thresholds,
        objective=config.objective, multi_threshold_ensemble=config.multi_threshold_ensemble,
        threshold_ladder=threshold_ladder,
    )
    model._last_eval = {
        "y_true": y_test, "y_pred": y_pred, "y_proba": y_proba, "groups": group_test,
        "metrics": metrics, "prediction_seconds": prediction_seconds,
    }

    if config.report:
        logger.info("Stage 9/9: generating report...")
        default_folder = f"smarttab_reports/{model_name}_{time.strftime('%Y%m%d_%H%M%S')}"
        report_dict = model.report(default_folder)
        logger.info("report written to %s", report_dict["_paths"]["folder"])

    return model


def load(path: str) -> SmartTabModel:
    bundle = load_bundle(path)
    feature_importance = get_feature_importance(bundle["estimator"], bundle["model_name"], bundle["feature_names"])
    return SmartTabModel(
        model_name=bundle["model_name"], task_type=bundle["task_type"], estimator=bundle["estimator"],
        cleaning_pipeline=bundle["cleaning_pipeline"], target_encoder=bundle["target_encoder"],
        feature_names=bundle["feature_names"], cat_features=bundle["cat_features"],
        best_params=bundle["best_params"], primary_metric=bundle["primary_metric"], metrics=bundle["metrics"],
        feature_importance=feature_importance, dataset_profile=bundle["dataset_profile"],
        hardware_profile=bundle["hardware_profile"], resource_plan=bundle["resource_plan"],
        class_labels=bundle["class_labels"], timings={}, notes=[], ensemble_info=bundle["ensemble_info"],
        decision_threshold=bundle["decision_threshold"], reject_threshold=bundle["reject_threshold"],
        per_label_thresholds=bundle["per_label_thresholds"], objective=bundle["objective"],
        multi_threshold_ensemble=bundle["multi_threshold_ensemble"], threshold_ladder=bundle["threshold_ladder"],
    )


def _split_train_test(df: pd.DataFrame, profile, config: FitConfig):
    """Returns (train_df, test_df, group_train, group_test). The latter two are None except
    for ranking, where the split must never separate rows of the same group."""
    if profile.task_type.is_ranking:
        splitter = GroupShuffleSplit(n_splits=1, test_size=config.test_size, random_state=config.random_state)
        train_idx, test_idx = next(splitter.split(df, groups=df[profile.group_id_column]))
        train_df = df.iloc[train_idx].reset_index(drop=True)
        test_df = df.iloc[test_idx].reset_index(drop=True)
        group_train = train_df[profile.group_id_column].to_numpy()
        group_test = test_df[profile.group_id_column].to_numpy()
        return train_df, test_df, group_train, group_test

    stratify = df[profile.target_column] if profile.task_type in (TaskType.BINARY, TaskType.MULTICLASS) else None
    train_df, test_df = train_test_split(
        df, test_size=config.test_size, random_state=config.random_state, stratify=stratify,
    )
    return train_df.reset_index(drop=True), test_df.reset_index(drop=True), None, None


def _extract_targets(train_df: pd.DataFrame, test_df: pd.DataFrame, profile):
    """Returns (target_encoder, y_train, y_test, class_labels)."""
    if profile.task_type in (TaskType.BINARY, TaskType.MULTICLASS):
        target_encoder = TargetLabelEncoder().fit(train_df[profile.target_column])
        y_train = target_encoder.transform(train_df[profile.target_column])
        y_test = target_encoder.transform(test_df[profile.target_column])
        return target_encoder, y_train, y_test, target_encoder.classes_.tolist()

    if profile.task_type is TaskType.MULTILABEL:
        y_train = train_df[profile.target_columns].astype(int).to_numpy()
        y_test = test_df[profile.target_columns].astype(int).to_numpy()
        return None, y_train, y_test, list(profile.target_columns)

    if profile.task_type is TaskType.MULTIOUTPUT_REGRESSION:
        y_train = train_df[profile.target_columns].to_numpy(dtype=float)
        y_test = test_df[profile.target_columns].to_numpy(dtype=float)
        return None, y_train, y_test, None

    # REGRESSION, RANKING — single numeric target column
    y_train = train_df[profile.target_column].to_numpy(dtype=float)
    y_test = test_df[profile.target_column].to_numpy(dtype=float)
    return None, y_train, y_test, None


def _default_metric_name(task_type: TaskType) -> str:
    if task_type in (TaskType.REGRESSION, TaskType.MULTIOUTPUT_REGRESSION):
        return "rmse"
    if task_type.is_ranking:
        return "ndcg"
    if task_type is TaskType.BINARY:
        return "roc_auc"
    return "f1_macro"


def _resolve_time_budget(config: FitConfig, fit_start: float) -> tuple[float | None, str | int, str | int, list[str]]:
    """Translate the global ``time_limit`` (if set) into a concrete Optuna timeout, trial-count
    cap, and CV fold count for the optimization/ensemble stage. Returns SmartTab-native
    ``config.timeout``/``config.n_trials``/``config.cv`` unchanged when ``time_limit`` is 0."""
    if config.time_limit <= 0:
        return config.timeout, config.n_trials, config.cv, []

    elapsed = time.perf_counter() - fit_start
    total_remaining = max(1.0, config.time_limit - elapsed)
    optimization_budget = total_remaining * (1 - TIME_LIMIT_RESERVE_FRACTION)

    timeout = min(config.timeout, optimization_budget) if config.timeout is not None else optimization_budget
    n_trials = TIME_BUDGET_N_TRIALS_CAP if config.n_trials == "auto" else config.n_trials
    cv = 2 if (config.cv == "auto" and optimization_budget < TIGHT_BUDGET_SECONDS) else config.cv

    notes = [f"time_limit={config.time_limit:.0f}s: optimization budget ~{optimization_budget:.0f}s (timeout drives trial count)"]
    return timeout, n_trials, cv, notes


def _discover_n_estimators(model_name, params, profile, resource_plan, X_train, y_train, cat_features, random_state, group_ids=None) -> int:
    """A single early-stopped fit on a small holdout carve, used to pick a sane iteration count
    when there was no full hyperparameter search to derive one from."""
    if profile.task_type.is_multi_target:
        # MultiOutputClassifier/Regressor-wrapped fits don't support eval_set/early stopping
        # (see training/trainer.py), so a probe fit here would be pure wasted compute.
        return FALLBACK_N_ESTIMATORS

    if profile.task_type.is_ranking:
        splitter = GroupShuffleSplit(n_splits=1, test_size=0.15, random_state=random_state)
        try:
            sub_idx, val_idx = next(splitter.split(X_train, groups=group_ids))
        except (ValueError, StopIteration):
            return FALLBACK_N_ESTIMATORS
        X_sub, X_val = X_train.iloc[sub_idx], X_train.iloc[val_idx]
        y_sub, y_val = y_train[sub_idx], y_train[val_idx]
        group_sub, group_val = group_ids[sub_idx], group_ids[val_idx]
        X_sub, y_sub, group_sub = sort_by_group(X_sub, y_sub, group_sub)
        X_val, y_val, group_val = sort_by_group(X_val, y_val, group_val)
    else:
        stratify = y_train if profile.task_type in (TaskType.BINARY, TaskType.MULTICLASS) else None
        try:
            X_sub, X_val, y_sub, y_val = train_test_split(
                X_train, y_train, test_size=0.15, random_state=random_state, stratify=stratify,
            )
        except ValueError:
            return FALLBACK_N_ESTIMATORS
        group_sub = group_val = None

    estimator = build_estimator(
        model_name, params, profile.task_type, n_estimators=DISCOVERY_N_ESTIMATORS_CAP,
        cpu_threads=resource_plan.cpu_threads, use_gpu=resource_plan.use_gpu, random_state=random_state,
    )
    fit_estimator(
        estimator, model_name, X_sub, y_sub, X_val, y_val,
        cat_features=cat_features, early_stopping_rounds=DEFAULT_EARLY_STOPPING_ROUNDS,
        group_ids_train=group_sub, group_ids_valid=group_val,
    )
    best_iter = get_best_iteration(estimator, model_name)
    return best_iter if best_iter else FALLBACK_N_ESTIMATORS
