"""Efficient, baseline-aware hyperparameter optimization."""

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
from smarttab.deadline import FitDeadline
from smarttab.evaluation.evaluator import compute_metric, compute_ndcg
from smarttab.exceptions import ConfigurationError
from smarttab.hardware.resource_planner import ResourcePlan
from smarttab.logging_utils import get_logger
from smarttab.optimization.search_spaces import SEARCH_SPACES, default_params
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

LARGE_DATASET_HOLDOUT_THRESHOLD = 750
VERY_LARGE_DATASET_THRESHOLD = 100_000
MAX_OPTIMIZATION_ROWS = 50_000
DEFAULT_N_ESTIMATORS_CAP = 450
FALLBACK_N_ESTIMATORS = 300
MIN_RELATIVE_IMPROVEMENT = 0.001


@dataclass
class OptimizationResult:
    best_params: dict
    primary_metric: str
    best_score: float
    best_n_estimators: int
    n_trials_run: int
    baseline_score: float | None = None
    relative_improvement: float = 0.0
    used_optimized_params: bool = False
    notes: list[str] = field(default_factory=list)


def resolve_primary_metric(profile: DatasetProfile, metrics: str = "auto") -> str:
    if metrics != "auto":
        if metrics not in METRIC_DIRECTION:
            raise ConfigurationError(
                f"metrics must be 'auto' or one of {tuple(METRIC_DIRECTION)}, got {metrics!r}"
            )
        return metrics
    if profile.task_type in (TaskType.REGRESSION, TaskType.MULTIOUTPUT_REGRESSION):
        return "rmse"
    if profile.task_type is TaskType.RANKING:
        return "ndcg"
    if profile.task_type is TaskType.BINARY:
        return "roc_auc"
    return "f1_macro"


def resolve_cv_splitter(
    profile: DatasetProfile,
    validation: str = "auto",
    cv: str | int = "auto",
    random_state: int = 42,
    y: np.ndarray | None = None,
    group_ids: np.ndarray | None = None,
):
    if validation not in ("auto", "kfold", "holdout"):
        raise ConfigurationError("validation must be 'auto', 'kfold', or 'holdout'")

    use_holdout = validation == "holdout" or (
        validation == "auto" and profile.n_samples >= LARGE_DATASET_HOLDOUT_THRESHOLD
    )
    if use_holdout:
        if profile.task_type.is_ranking or group_ids is not None:
            return GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=random_state)
        if profile.task_type in (TaskType.BINARY, TaskType.MULTICLASS) and _can_stratify(y, 2):
            return StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=random_state)
        return ShuffleSplit(n_splits=1, test_size=0.2, random_state=random_state)

    requested_folds = 3 if cv == "auto" else int(cv)
    if profile.task_type.is_ranking or group_ids is not None:
        n_groups = len(np.unique(group_ids)) if group_ids is not None else int(profile.n_groups or 0)
        folds = min(requested_folds, n_groups)
        if folds < 2:
            raise ConfigurationError("group-aware cross-validation requires at least two groups")
        return GroupKFold(n_splits=folds)
    if profile.task_type in (TaskType.BINARY, TaskType.MULTICLASS):
        minimum_class_count = _minimum_class_count(y)
        folds = min(requested_folds, minimum_class_count)
        if folds >= 2:
            return StratifiedKFold(n_splits=folds, shuffle=True, random_state=random_state)
        return ShuffleSplit(n_splits=1, test_size=0.2, random_state=random_state)
    folds = min(requested_folds, max(2, profile.n_samples // 2))
    return KFold(n_splits=folds, shuffle=True, random_state=random_state)


def resolve_n_trials(
    profile: DatasetProfile,
    resource_plan: ResourcePlan,
    n_trials: str | int = "auto",
) -> int:
    if n_trials != "auto":
        value = int(n_trials)
        if value < 1:
            raise ConfigurationError("n_trials must be >= 1")
        return value
    if profile.n_samples >= VERY_LARGE_DATASET_THRESHOLD or profile.n_features > 250:
        value = 4
    elif profile.n_samples < 2_000:
        value = 8
    else:
        value = 6
    if resource_plan.cpu_threads <= 2:
        value = min(value, 6)
    return value


def resolve_class_weight_params(model_name: str, profile: DatasetProfile) -> dict:
    if not profile.task_type.is_classification or not profile.is_imbalanced:
        return {}
    if model_name == "catboost":
        return {"auto_class_weights": "Balanced"}
    if model_name in ("lightgbm", "xgboost"):
        return {"class_weight": "balanced"} if model_name == "lightgbm" else {}
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
    deadline: FitDeadline | None = None,
) -> OptimizationResult:
    if optimizer not in ("auto", "tpe", "random"):
        raise ConfigurationError("optimizer must be 'auto', 'tpe', or 'random'")
    primary_metric = resolve_primary_metric(profile, metrics)
    direction = METRIC_DIRECTION[primary_metric]
    X_search, y_search, groups_search, sample_note = _optimization_sample(
        X,
        np.asarray(y),
        task_type,
        group_ids,
        random_state,
    )
    splitter = resolve_cv_splitter(
        profile,
        validation,
        cv,
        random_state,
        y_search,
        groups_search,
    )
    resolved_n_trials = resolve_n_trials(profile, resource_plan, n_trials)
    class_weight_params = resolve_class_weight_params(model_name, profile)
    base_params = {**default_params(model_name), **class_weight_params}
    space_fn = SEARCH_SPACES[model_name]

    effective_timeout = timeout
    if deadline is not None:
        effective_timeout = deadline.bounded_timeout(timeout, reserve=2.0)
    if effective_timeout is not None and effective_timeout <= 0:
        return OptimizationResult(
            best_params=base_params,
            primary_metric=primary_metric,
            best_score=float("nan"),
            best_n_estimators=FALLBACK_N_ESTIMATORS,
            n_trials_run=0,
            notes=["optimization skipped because the global time budget was exhausted"],
        )

    baseline_score, baseline_iterations = _evaluate_params(
        model_name,
        base_params,
        X_search,
        y_search,
        task_type,
        primary_metric,
        splitter,
        resource_plan,
        cat_features,
        groups_search,
        random_state,
        deadline,
    )

    sampler = (
        optuna.samplers.RandomSampler(seed=random_state)
        if optimizer == "random"
        else optuna.samplers.TPESampler(
            seed=random_state,
            n_startup_trials=min(3, resolved_n_trials),
        )
    )

    def objective(trial: optuna.Trial) -> float:
        params = {**space_fn(trial, resource_plan), **class_weight_params}
        score, iterations = _evaluate_params(
            model_name,
            params,
            X_search,
            y_search,
            task_type,
            primary_metric,
            splitter,
            resource_plan,
            cat_features,
            groups_search,
            random_state + trial.number,
            deadline,
            trial,
        )
        trial.set_user_attr("best_iterations", iterations)
        return score

    study = optuna.create_study(
        direction=direction,
        sampler=sampler,
        pruner=optuna.pruners.MedianPruner(n_startup_trials=min(3, resolved_n_trials)),
    )
    try:
        study.optimize(
            objective,
            n_trials=resolved_n_trials,
            timeout=effective_timeout,
            show_progress_bar=verbose > 0,
            gc_after_trial=False,
            catch=(RuntimeError,),
        )
    except Exception as exc:
        logger.warning("optimization stopped early for %s: %s", model_name, exc)

    completed = [trial for trial in study.trials if trial.state is optuna.trial.TrialState.COMPLETE]
    if not completed:
        best_score = baseline_score
        chosen_params = base_params
        chosen_iterations = baseline_iterations
        improvement = 0.0
        use_optimized = False
    else:
        best_trial = study.best_trial
        optimized_score = float(best_trial.value)
        improvement = _relative_improvement(baseline_score, optimized_score, direction)
        use_optimized = improvement >= MIN_RELATIVE_IMPROVEMENT
        if use_optimized:
            chosen_params = {**best_trial.params, **class_weight_params}
            chosen_iterations = best_trial.user_attrs.get("best_iterations", [])
            best_score = optimized_score
        else:
            chosen_params = base_params
            chosen_iterations = baseline_iterations
            best_score = baseline_score

    valid_iterations = [int(value) for value in chosen_iterations if value and int(value) > 0]
    best_n_estimators = int(np.median(valid_iterations)) if valid_iterations else FALLBACK_N_ESTIMATORS
    best_n_estimators = max(20, min(best_n_estimators, DEFAULT_N_ESTIMATORS_CAP))

    validation_name = (
        "holdout"
        if isinstance(splitter, (ShuffleSplit, StratifiedShuffleSplit, GroupShuffleSplit))
        else f"k-fold({getattr(splitter, 'n_splits', '?')})"
    )
    notes = [
        f"validation={validation_name}",
        f"optimization metric={primary_metric} ({direction})",
        f"baseline {primary_metric}={baseline_score:.6f}",
        f"completed {len(completed)}/{resolved_n_trials} requested trials",
    ]
    if sample_note:
        notes.append(sample_note)
    if use_optimized:
        notes.append(f"optimized parameters retained; relative improvement={improvement:.3%}")
    else:
        notes.append(
            f"optimized parameters rejected; improvement={improvement:.3%} below {MIN_RELATIVE_IMPROVEMENT:.3%}"
        )

    return OptimizationResult(
        best_params=chosen_params,
        primary_metric=primary_metric,
        best_score=float(best_score),
        best_n_estimators=best_n_estimators,
        n_trials_run=len(completed),
        baseline_score=float(baseline_score),
        relative_improvement=float(improvement),
        used_optimized_params=use_optimized,
        notes=notes,
    )


def _evaluate_params(
    model_name: str,
    params: dict,
    X: pd.DataFrame,
    y: np.ndarray,
    task_type: TaskType,
    primary_metric: str,
    splitter,
    resource_plan: ResourcePlan,
    cat_features: list[str],
    group_ids: np.ndarray | None,
    random_state: int,
    deadline: FitDeadline | None,
    trial: optuna.Trial | None = None,
) -> tuple[float, list[int]]:
    scores: list[float] = []
    iterations: list[int] = []
    for fold_index, (train_index, valid_index) in enumerate(
        splitter.split(X, y, groups=group_ids)
    ):
        if deadline is not None:
            deadline.require(f"optimization fold {fold_index + 1}")
        X_train, X_valid = X.iloc[train_index], X.iloc[valid_index]
        y_train, y_valid = y[train_index], y[valid_index]
        group_train = group_valid = None
        if task_type.is_ranking:
            group_train, group_valid = group_ids[train_index], group_ids[valid_index]
            X_train, y_train, group_train = sort_by_group(X_train, y_train, group_train)
            X_valid, y_valid, group_valid = sort_by_group(X_valid, y_valid, group_valid)

        estimator = build_estimator(
            model_name,
            params,
            task_type,
            DEFAULT_N_ESTIMATORS_CAP,
            resource_plan.cpu_threads,
            resource_plan.use_gpu,
            random_state + fold_index,
            resource_plan=resource_plan,
        )
        fit_estimator(
            estimator,
            model_name,
            X_train,
            y_train,
            X_valid,
            y_valid,
            cat_features=cat_features,
            early_stopping_rounds=DEFAULT_EARLY_STOPPING_ROUNDS,
            group_ids_train=group_train,
            group_ids_valid=group_valid,
            deadline=deadline,
            resource_plan=resource_plan,
        )
        if task_type.is_ranking:
            score = compute_ndcg(y_valid, predict(estimator, model_name, X_valid), group_valid, k=10)
        elif task_type.is_classification and primary_metric == "roc_auc":
            predictions = predict(estimator, model_name, X_valid)
            probabilities = predict_proba(estimator, model_name, X_valid)
            score = compute_metric(primary_metric, y_valid, predictions, probabilities)
        else:
            score = compute_metric(primary_metric, y_valid, predict(estimator, model_name, X_valid))
        scores.append(float(score))
        best_iteration = get_best_iteration(estimator, model_name)
        if best_iteration:
            iterations.append(best_iteration)
        if trial is not None:
            trial.report(float(np.mean(scores)), step=fold_index)
            if trial.should_prune():
                raise optuna.TrialPruned()
    return float(np.mean(scores)), iterations


def _optimization_sample(
    X: pd.DataFrame,
    y: np.ndarray,
    task_type: TaskType,
    group_ids: np.ndarray | None,
    random_state: int,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray | None, str | None]:
    if len(X) <= MAX_OPTIMIZATION_ROWS:
        return X.reset_index(drop=True), y, group_ids, None
    rng = np.random.default_rng(random_state)
    if group_ids is not None:
        unique_groups = np.unique(group_ids)
        rng.shuffle(unique_groups)
        selected_groups: list[object] = []
        selected_rows = 0
        for group in unique_groups:
            selected_groups.append(group)
            selected_rows += int(np.sum(group_ids == group))
            if selected_rows >= MAX_OPTIMIZATION_ROWS:
                break
        mask = np.isin(group_ids, selected_groups)
        indices = np.flatnonzero(mask)
    elif task_type in (TaskType.BINARY, TaskType.MULTICLASS):
        indices = []
        for label in np.unique(y):
            label_indices = np.flatnonzero(y == label)
            take = max(1, round(MAX_OPTIMIZATION_ROWS * len(label_indices) / len(y)))
            indices.extend(rng.choice(label_indices, size=min(take, len(label_indices)), replace=False).tolist())
        indices = np.asarray(sorted(indices[:MAX_OPTIMIZATION_ROWS]))
    else:
        indices = np.sort(rng.choice(len(X), size=MAX_OPTIMIZATION_ROWS, replace=False))
    sampled_groups = group_ids[indices] if group_ids is not None else None
    return (
        X.iloc[indices].reset_index(drop=True),
        y[indices],
        sampled_groups,
        f"optimization used a representative sample of {len(indices):,}/{len(X):,} rows",
    )


def _minimum_class_count(y: np.ndarray | None) -> int:
    if y is None or np.asarray(y).ndim != 1:
        return 0
    _, counts = np.unique(y, return_counts=True)
    return int(counts.min()) if len(counts) else 0


def _can_stratify(y: np.ndarray | None, minimum: int) -> bool:
    return _minimum_class_count(y) >= minimum


def _relative_improvement(baseline: float, candidate: float, direction: str) -> float:
    denominator = max(abs(baseline), 1e-12)
    raw = candidate - baseline if direction == "maximize" else baseline - candidate
    return raw / denominator
