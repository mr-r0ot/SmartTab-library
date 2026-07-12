"""Stage 5 — Hyperparameter Optimization.

Optuna (TPE by default) with cross-validated, early-stopped trials and
median pruning. Grid search is never used; the search spaces themselves
live in ``search_spaces.py``. This module also resolves the "auto" values
for validation strategy, number of trials, and the optimization metric —
across every task type, including group-aware CV for ranking (rows sharing
a group id are never split across train/validation) and 2D-y scoring for
multilabel/multi-output regression (sklearn's metrics already handle those
shapes natively, so no special-casing is needed there).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import optuna
import pandas as pd
from sklearn.model_selection import (
    GroupKFold,
    GroupShuffleSplit,
    KFold,
    ShuffleSplit,
    StratifiedKFold,
    StratifiedShuffleSplit,
)

from smarttab.analysis.dataset_analyzer import DatasetProfile, TaskType
from smarttab.evaluation.evaluator import compute_metric, compute_ndcg
from smarttab.exceptions import ConfigurationError
from smarttab.hardware.resource_planner import ResourcePlan
from smarttab.logging_utils import get_logger
from smarttab.optimization.search_spaces import SEARCH_SPACES
from smarttab.training.trainer import (
    DEFAULT_EARLY_STOPPING_ROUNDS,
    build_estimator,
    fit_estimator,
    get_best_iteration,
    predict,
    predict_proba,
    sort_by_group,
)

logger = get_logger()

optuna.logging.set_verbosity(optuna.logging.WARNING)

METRIC_DIRECTION = {
    "rmse": "minimize",
    "mae": "minimize",
    "r2": "maximize",
    "roc_auc": "maximize",
    "f1_macro": "maximize",
    "f1": "maximize",
    "ndcg": "maximize",
}

LARGE_DATASET_HOLDOUT_THRESHOLD = 200_000
# Cap on boosting rounds *during search*: trials use early stopping, so this only bounds the
# worst case per trial/fold. Kept well below what a final model might use so a 50-100 trial
# search over several CV folds still finishes in a reasonable time.
DEFAULT_N_ESTIMATORS_CAP = 300
FALLBACK_N_ESTIMATORS = 300


@dataclass
class OptimizationResult:
    best_params: dict
    primary_metric: str
    best_score: float
    best_n_estimators: int
    n_trials_run: int
    notes: list[str] = field(default_factory=list)


def resolve_primary_metric(profile: DatasetProfile, metrics: str = "auto") -> str:
    if metrics != "auto":
        if metrics not in METRIC_DIRECTION:
            raise ConfigurationError(f"metrics must be 'auto' or one of {list(METRIC_DIRECTION)}, got {metrics!r}")
        return metrics
    if profile.task_type in (TaskType.REGRESSION, TaskType.MULTIOUTPUT_REGRESSION):
        return "rmse"
    if profile.task_type is TaskType.RANKING:
        return "ndcg"
    if profile.task_type is TaskType.BINARY:
        return "roc_auc"
    return "f1_macro"  # MULTICLASS, MULTILABEL


def resolve_cv_splitter(profile: DatasetProfile, validation: str = "auto", cv: str | int = "auto", random_state: int = 42):
    if validation not in ("auto", "kfold", "holdout"):
        raise ConfigurationError(f"validation must be 'auto', 'kfold', or 'holdout', got {validation!r}")

    use_holdout = validation == "holdout" or (validation == "auto" and profile.n_samples > LARGE_DATASET_HOLDOUT_THRESHOLD)

    if use_holdout:
        if profile.task_type.is_ranking:
            return GroupShuffleSplit(n_splits=1, test_size=0.15, random_state=random_state)
        if profile.task_type is TaskType.MULTILABEL:
            return ShuffleSplit(n_splits=1, test_size=0.15, random_state=random_state)
        if profile.task_type.is_classification:
            return StratifiedShuffleSplit(n_splits=1, test_size=0.15, random_state=random_state)
        return ShuffleSplit(n_splits=1, test_size=0.15, random_state=random_state)

    if cv == "auto":
        # Kept modest by default (search reruns this many times per trial): 5-fold is only
        # worth the extra cost once there's enough data that per-fold noise would otherwise
        # make trials hard to compare.
        folds = 5 if profile.n_samples >= 20_000 else 3
    else:
        try:
            folds = int(cv)
        except (TypeError, ValueError) as exc:
            raise ConfigurationError(f"cv must be 'auto' or an integer, got {cv!r}") from exc
        if folds < 2:
            raise ConfigurationError("cv must be >= 2")

    if profile.task_type.is_ranking:
        return GroupKFold(n_splits=folds)
    if profile.task_type is TaskType.MULTILABEL:
        # StratifiedKFold requires a 1D target; multilabel y is (n_samples, n_labels).
        return KFold(n_splits=folds, shuffle=True, random_state=random_state)
    if profile.task_type.is_classification:
        return StratifiedKFold(n_splits=folds, shuffle=True, random_state=random_state)
    return KFold(n_splits=folds, shuffle=True, random_state=random_state)


def resolve_n_trials(profile: DatasetProfile, resource_plan: ResourcePlan, n_trials: str | int = "auto") -> int:
    if n_trials != "auto":
        return int(n_trials)
    base = 30
    if profile.n_samples > LARGE_DATASET_HOLDOUT_THRESHOLD or profile.n_features > 200:
        base = 15
    elif profile.n_samples < 2000:
        base = 40
    if resource_plan.cpu_threads <= 2:
        base = min(base, 20)
    return base


def resolve_class_weight_params(model_name: str, profile: DatasetProfile) -> dict:
    if not profile.task_type.is_classification or not profile.is_imbalanced:
        return {}
    if model_name == "catboost":
        return {"auto_class_weights": "Balanced"}
    if model_name == "lightgbm":
        return {"class_weight": "balanced"}
    return {}


def run_optimization(
    model_name: str,
    X: pd.DataFrame,
    y: np.ndarray,
    task_type: TaskType,
    profile: DatasetProfile,
    resource_plan: ResourcePlan,
    cat_features: list[str],
    validation: str = "auto",
    cv: str | int = "auto",
    optimizer: str = "auto",
    n_trials: str | int = "auto",
    timeout: float | None = None,
    metrics: str = "auto",
    random_state: int = 42,
    verbose: int = 1,
    group_ids: np.ndarray | None = None,
) -> OptimizationResult:
    primary_metric = resolve_primary_metric(profile, metrics)
    splitter = resolve_cv_splitter(profile, validation, cv, random_state)
    resolved_n_trials = resolve_n_trials(profile, resource_plan, n_trials)
    class_weight_params = resolve_class_weight_params(model_name, profile)
    space_fn = SEARCH_SPACES[model_name]

    if optimizer not in ("auto", "tpe", "random"):
        raise ConfigurationError(f"optimizer must be 'auto', 'tpe', or 'random', got {optimizer!r}")
    sampler = optuna.samplers.RandomSampler(seed=random_state) if optimizer == "random" else optuna.samplers.TPESampler(seed=random_state)

    def objective(trial: optuna.Trial) -> float:
        params = {**space_fn(trial, resource_plan), **class_weight_params}
        scores: list[float] = []
        best_iterations: list[int] = []

        for fold_idx, (train_idx, valid_idx) in enumerate(splitter.split(X, y, groups=group_ids)):
            X_tr, X_val = X.iloc[train_idx], X.iloc[valid_idx]
            y_tr, y_val = y[train_idx], y[valid_idx]
            group_tr = group_val = None

            if task_type.is_ranking:
                group_tr, group_val = group_ids[train_idx], group_ids[valid_idx]
                X_tr, y_tr, group_tr = sort_by_group(X_tr, y_tr, group_tr)
                X_val, y_val, group_val = sort_by_group(X_val, y_val, group_val)

            estimator = build_estimator(
                model_name, params, task_type, n_estimators=DEFAULT_N_ESTIMATORS_CAP,
                cpu_threads=resource_plan.cpu_threads, use_gpu=resource_plan.use_gpu, random_state=random_state,
            )
            fit_estimator(
                estimator, model_name, X_tr, y_tr, X_val, y_val,
                cat_features=cat_features, early_stopping_rounds=DEFAULT_EARLY_STOPPING_ROUNDS,
                group_ids_train=group_tr, group_ids_valid=group_val,
            )

            if task_type.is_ranking:
                score = compute_ndcg(y_val, predict(estimator, model_name, X_val), group_val, k=10)
            elif task_type.is_classification and primary_metric == "roc_auc":
                score = compute_metric(primary_metric, y_val, predict(estimator, model_name, X_val), predict_proba(estimator, model_name, X_val))
            else:
                score = compute_metric(primary_metric, y_val, predict(estimator, model_name, X_val))
            scores.append(score)

            best_iter = get_best_iteration(estimator, model_name)
            if best_iter:
                best_iterations.append(best_iter)

            trial.report(float(np.mean(scores)), step=fold_idx)
            if trial.should_prune():
                raise optuna.TrialPruned()

        trial.set_user_attr("best_iterations", best_iterations)
        return float(np.mean(scores))

    study = optuna.create_study(
        direction=METRIC_DIRECTION[primary_metric], sampler=sampler, pruner=optuna.pruners.MedianPruner()
    )
    study.optimize(objective, n_trials=resolved_n_trials, timeout=timeout, show_progress_bar=verbose > 0)

    best_iterations = study.best_trial.user_attrs.get("best_iterations", [])
    best_n_estimators = int(np.median(best_iterations)) if best_iterations else FALLBACK_N_ESTIMATORS

    notes = [
        f"validation={'holdout' if isinstance(splitter, (ShuffleSplit, StratifiedShuffleSplit, GroupShuffleSplit)) else 'k-fold(' + str(getattr(splitter, 'n_splits', '?')) + ')'}",
        f"optimizing primary metric '{primary_metric}' ({METRIC_DIRECTION[primary_metric]})",
        f"ran {len(study.trials)} trials (requested {resolved_n_trials})",
    ]

    return OptimizationResult(
        best_params={**study.best_trial.params, **class_weight_params},
        primary_metric=primary_metric,
        best_score=study.best_value,
        best_n_estimators=best_n_estimators,
        n_trials_run=len(study.trials),
        notes=notes,
    )
