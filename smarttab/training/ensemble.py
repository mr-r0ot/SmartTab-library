"""Voting/Stacking ensembles over CatBoost + LightGBM + XGBoost, plus the
``ensemble="auto"`` decision engine.

Three things live here:

- ``VotingEnsemble`` / ``StackingEnsemble``: thin wrappers exposing the same
  ``predict``/``predict_proba`` interface as a single model, so the rest of
  SmartTab (evaluation, report, persistence) doesn't need to know an
  ensemble is in play. Voting is *weighted* soft voting — each base
  learner's weight comes from its own held-out score, not a plain average.
- ``train_voting_stacking_ensemble``: trains all three base learners and
  builds one or both combiners, controlled by ``strategy``
  (``"voting"``, ``"stacking"``, or ``"compare"`` — build both and keep
  whichever scores better).
- ``run_ensemble_decision_engine``: the ``ensemble="auto"`` policy. Instead
  of always paying for three models, it first tunes just CatBoost and
  LightGBM and compares them. If one is clearly ahead it's returned as-is
  (no ensemble, no XGBoost) — if they're close, it builds the full
  voting/stacking comparison and only keeps the ensemble if it actually
  beats the best single model.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import train_test_split
from tqdm.auto import tqdm

from smarttab.analysis.dataset_analyzer import DatasetProfile, TaskType
from smarttab.evaluation.evaluator import compute_metric
from smarttab.hardware.resource_planner import ResourcePlan
from smarttab.logging_utils import get_logger
from smarttab.optimization.optimizer import METRIC_DIRECTION, OptimizationResult, resolve_n_trials, resolve_primary_metric, run_optimization
from smarttab.optimization.search_spaces import default_params
from smarttab.optimization.threshold import (
    DEFAULT_OBJECTIVE,
    DEFAULT_REJECT_THRESHOLD,
    DEFAULT_THRESHOLD,
    build_multiclass_threshold_ladder,
    build_threshold_ladder,
    optimize_multiclass_reject_threshold,
    optimize_threshold,
    probe_and_optimize_threshold,
)
from smarttab.training.trainer import build_estimator, fit_estimator, predict, predict_proba

logger = get_logger()

BASE_MODEL_NAMES = ("catboost", "lightgbm", "xgboost")
BASE_FIT_FRACTION = 0.6  # rest is split evenly between meta-learner training and strategy selection
MIN_WEIGHT_FLOOR = 0.2  # no base learner drops below this share of an equal-weight baseline

# ensemble="auto" decision thresholds (relative difference in the primary metric).
CLOSE_PERFORMANCE_THRESHOLD = 0.003  # < 0.3%: models are essentially tied -> build an ensemble
CLEAR_WINNER_THRESHOLD = 0.01  # >= 1%: one model is clearly better -> skip the ensemble entirely


class VotingEnsemble:
    """Weighted soft-voting (classification) / weighted averaging (regression).

    ``weights`` (aligned with ``base_models``) default to equal weighting;
    pass per-model weights derived from held-out scores for genuinely
    *weighted* soft voting.
    """

    def __init__(self, base_models: list[tuple[str, object]], task_type: TaskType, weights: list[float] | None = None):
        self.base_models = base_models
        self.task_type = task_type
        self.weights = weights

    def _weights_array(self) -> np.ndarray:
        if self.weights is None:
            return np.full(len(self.base_models), 1.0 / len(self.base_models))
        w = np.asarray(self.weights, dtype=float)
        return w / w.sum()

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        probas = [predict_proba(est, name, X) for name, est in self.base_models]
        averaged = np.average(probas, axis=0, weights=self._weights_array())
        # floating-point summation can leave rows a hair off 1.0 (e.g. 0.9999999999998), which
        # trips sklearn's strict "do the probabilities sum to one" check in log_loss/roc_auc —
        # renormalize so every row sums to exactly 1.0.
        return averaged / averaged.sum(axis=1, keepdims=True)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if self.task_type.is_classification:
            return np.argmax(self.predict_proba(X), axis=1)
        preds = [predict(est, name, X) for name, est in self.base_models]
        return np.average(preds, axis=0, weights=self._weights_array())


class StackingEnsemble:
    """A LogisticRegression/Ridge meta-learner trained on base-model predictions."""

    def __init__(self, base_models: list[tuple[str, object]], task_type: TaskType, meta_model):
        self.base_models = base_models
        self.task_type = task_type
        self.meta_model = meta_model

    def _meta_features(self, X: pd.DataFrame) -> np.ndarray:
        if self.task_type.is_classification:
            probas = [predict_proba(est, name, X) for name, est in self.base_models]
            return np.hstack(probas)
        preds = [predict(est, name, X).reshape(-1, 1) for name, est in self.base_models]
        return np.hstack(preds)

    def fit_meta(self, X_meta: pd.DataFrame, y_meta: np.ndarray) -> "StackingEnsemble":
        self.meta_model.fit(self._meta_features(X_meta), y_meta)
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return np.asarray(self.meta_model.predict_proba(self._meta_features(X)))

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return np.asarray(self.meta_model.predict(self._meta_features(X)))


@dataclass
class EnsembleResult:
    strategy: str  # "voting" | "stacking"
    estimator: object
    base_params: dict = field(default_factory=dict)
    base_n_estimators: dict = field(default_factory=dict)
    base_scores: dict = field(default_factory=dict)
    primary_metric: str = ""
    validation_score: float = 0.0
    decision_threshold: float = DEFAULT_THRESHOLD
    reject_threshold: float = DEFAULT_REJECT_THRESHOLD
    threshold_ladder: list[dict] | None = None
    notes: list[str] = field(default_factory=list)


@dataclass
class AutoDecisionResult:
    used_ensemble: bool
    strategy: str  # "catboost" | "lightgbm" | "voting" | "stacking"
    estimator: object
    best_params: dict
    primary_metric: str
    best_n_estimators: int | None = None
    ensemble_info: dict | None = None
    decision_threshold: float = DEFAULT_THRESHOLD
    reject_threshold: float = DEFAULT_REJECT_THRESHOLD
    threshold_ladder: list[dict] | None = None
    notes: list[str] = field(default_factory=list)


def _is_better(a: float, b: float, direction: str) -> bool:
    return a > b if direction == "maximize" else a < b


def _relative_diff(a: float, b: float) -> float:
    denom = max(abs(a), abs(b), 1e-12)
    return abs(a - b) / denom


def _compute_weights(scores: dict[str, float], direction: str) -> dict[str, float]:
    names = list(scores.keys())
    values = np.array([scores[n] for n in names], dtype=float)
    if direction == "minimize":
        values = -values
    lo, hi = values.min(), values.max()
    goodness = np.ones_like(values) if hi - lo < 1e-12 else MIN_WEIGHT_FLOOR + (1 - MIN_WEIGHT_FLOOR) * (values - lo) / (hi - lo)
    weights = goodness / goodness.sum()
    return dict(zip(names, weights))


def train_voting_stacking_ensemble(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
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
    optimize: bool = True,
    random_state: int = 42,
    verbose: int = 1,
    strategy: str = "compare",
    threshold_optimization: bool = True,
    objective: str = DEFAULT_OBJECTIVE,
    multi_threshold_ensemble: bool = False,
    threshold_models: int = 4,
) -> EnsembleResult:
    if strategy not in ("voting", "stacking", "compare"):
        raise ValueError(f"strategy must be 'voting', 'stacking', or 'compare', got {strategy!r}")

    primary_metric = resolve_primary_metric(profile, metrics)
    notes: list[str] = ["training catboost + lightgbm + xgboost base learners"]

    stratify = y_train if task_type.is_classification else None
    X_fit, X_rest, y_fit, y_rest = train_test_split(
        X_train, y_train, test_size=1 - BASE_FIT_FRACTION, random_state=random_state, stratify=stratify,
    )
    strat_rest = y_rest if task_type.is_classification else None
    X_meta, X_select, y_meta, y_select = train_test_split(
        X_rest, y_rest, test_size=0.5, random_state=random_state, stratify=strat_rest,
    )

    if n_trials == "auto":
        total_trials = resolve_n_trials(profile, resource_plan)
    else:
        total_trials = int(n_trials)
    per_model_trials = max(5, total_trials // len(BASE_MODEL_NAMES))
    per_model_timeout = timeout / len(BASE_MODEL_NAMES) if timeout else None

    base_params: dict[str, dict] = {}
    base_n_estimators: dict[str, int] = {}
    base_estimators: list[tuple[str, object]] = []

    base_model_iter = tqdm(
        BASE_MODEL_NAMES, desc="Training base models", unit="model", disable=verbose == 0,
    )
    for model_name in base_model_iter:
        base_model_iter.set_postfix_str(model_name)
        if optimize:
            opt_result = run_optimization(
                model_name=model_name, X=X_fit, y=y_fit, task_type=task_type, profile=profile,
                resource_plan=resource_plan, cat_features=cat_features, validation=validation, cv=cv,
                optimizer=optimizer, n_trials=per_model_trials, timeout=per_model_timeout, metrics=metrics,
                random_state=random_state, verbose=verbose,
            )
            params, n_estimators = opt_result.best_params, opt_result.best_n_estimators
            notes.append(f"{model_name}: tuned with {per_model_trials} trials -> n_estimators={n_estimators}")
        else:
            params, n_estimators = default_params(model_name), 300
            notes.append(f"{model_name}: optimize=False, using default hyperparameters")

        base_params[model_name] = params
        base_n_estimators[model_name] = n_estimators

        estimator = build_estimator(
            model_name, params, task_type, n_estimators=n_estimators,
            cpu_threads=resource_plan.cpu_threads, use_gpu=resource_plan.use_gpu, random_state=random_state,
        )
        fit_estimator(estimator, model_name, X_fit, y_fit, cat_features=cat_features)
        base_estimators.append((model_name, estimator))

    base_scores = {
        name: _score_base_model(name, est, task_type, primary_metric, X_select, y_select)
        for name, est in base_estimators
    }
    direction = METRIC_DIRECTION[primary_metric]
    weights = _compute_weights(base_scores, direction)
    notes.append(f"base model scores on selection split: {', '.join(f'{k}={v:.4f}' for k, v in base_scores.items())}")

    candidates: dict[str, tuple[object, float]] = {}
    if strategy in ("voting", "compare"):
        voting = VotingEnsemble(base_estimators, task_type, weights=[weights[n] for n, _ in base_estimators])
        candidates["voting"] = (voting, _score(voting, task_type, primary_metric, X_select, y_select))
    if strategy in ("stacking", "compare"):
        meta_model = LogisticRegression(max_iter=1000) if task_type.is_classification else Ridge()
        stacking = StackingEnsemble(base_estimators, task_type, meta_model).fit_meta(X_meta, y_meta)
        candidates["stacking"] = (stacking, _score(stacking, task_type, primary_metric, X_select, y_select))

    if len(candidates) > 1:
        chosen_strategy = max(candidates, key=lambda k: candidates[k][1] if direction == "maximize" else -candidates[k][1])
        notes.append(
            f"voting {primary_metric}={candidates['voting'][1]:.4f} vs stacking {primary_metric}={candidates['stacking'][1]:.4f}"
        )
    else:
        chosen_strategy = next(iter(candidates))
    notes.append(f"selected strategy: {chosen_strategy}")
    chosen_estimator, chosen_score = candidates[chosen_strategy]

    decision_threshold = DEFAULT_THRESHOLD
    reject_threshold = DEFAULT_REJECT_THRESHOLD
    threshold_ladder = None
    if threshold_optimization and task_type in (TaskType.BINARY, TaskType.MULTICLASS):
        proba_full = chosen_estimator.predict_proba(X_select)
        if task_type is TaskType.MULTICLASS:
            reject_threshold, best_score = optimize_multiclass_reject_threshold(y_select, proba_full, objective=objective)
            notes.append(f"reject_threshold={reject_threshold:.2f} ({objective}={best_score:.4f} on selection split)")
            if multi_threshold_ensemble:
                threshold_ladder = build_multiclass_threshold_ladder(y_select, proba_full, threshold_models)
                notes.append(f"multi_threshold_ensemble: built a {threshold_models}-point reject-threshold ladder on the selection split")
        else:
            proba = proba_full[:, 1]
            decision_threshold, best_score = optimize_threshold(y_select, proba, objective=objective)
            notes.append(f"decision_threshold={decision_threshold:.2f} ({objective}={best_score:.4f} on selection split)")
            if multi_threshold_ensemble:
                threshold_ladder = build_threshold_ladder(y_select, proba, threshold_models)
                notes.append(f"multi_threshold_ensemble: built a {threshold_models}-point threshold ladder on the selection split")

    return EnsembleResult(
        strategy=chosen_strategy, estimator=chosen_estimator, base_params=base_params,
        base_n_estimators=base_n_estimators, base_scores=base_scores, primary_metric=primary_metric,
        validation_score=chosen_score, decision_threshold=decision_threshold, reject_threshold=reject_threshold,
        threshold_ladder=threshold_ladder, notes=notes,
    )


def run_ensemble_decision_engine(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
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
    optimize: bool = True,
    random_state: int = 42,
    verbose: int = 1,
    threshold_optimization: bool = True,
    objective: str = DEFAULT_OBJECTIVE,
    multi_threshold_ensemble: bool = False,
    threshold_models: int = 4,
) -> AutoDecisionResult:
    """``ensemble="auto"``: tune CatBoost and LightGBM, compare, and only pay for a full
    3-model ensemble when the two are close enough that combining them plausibly helps."""
    primary_metric = resolve_primary_metric(profile, metrics)
    direction = METRIC_DIRECTION[primary_metric]
    notes: list[str] = ["ensemble=auto: tuning catboost and lightgbm to compare"]

    candidate_trials = "auto" if n_trials == "auto" else max(5, int(n_trials) // 2)
    candidate_timeout = timeout / 2 if timeout else None

    start = time.perf_counter()
    candidates: dict[str, OptimizationResult] = {}
    candidate_iter = tqdm(("catboost", "lightgbm"), desc="Comparing candidates", unit="model", disable=verbose == 0)
    for model_name in candidate_iter:
        candidate_iter.set_postfix_str(model_name)
        candidates[model_name] = run_optimization(
            model_name=model_name, X=X_train, y=y_train, task_type=task_type, profile=profile,
            resource_plan=resource_plan, cat_features=cat_features, validation=validation, cv=cv,
            optimizer=optimizer, n_trials=candidate_trials, timeout=candidate_timeout, metrics=metrics,
            random_state=random_state, verbose=verbose,
        )
    elapsed = time.perf_counter() - start
    remaining_timeout = max(1.0, timeout - elapsed) if timeout else None

    score_cb, score_lgbm = candidates["catboost"].best_score, candidates["lightgbm"].best_score
    rel_diff = _relative_diff(score_cb, score_lgbm)
    winner_name = "catboost" if _is_better(score_cb, score_lgbm, direction) else "lightgbm"
    notes.append(
        f"catboost {primary_metric}={score_cb:.4f} vs lightgbm {primary_metric}={score_lgbm:.4f} "
        f"(relative diff {rel_diff:.2%})"
    )

    def _finalize_single(name: str) -> AutoDecisionResult:
        opt = candidates[name]
        estimator = build_estimator(
            name, opt.best_params, task_type, n_estimators=opt.best_n_estimators,
            cpu_threads=resource_plan.cpu_threads, use_gpu=resource_plan.use_gpu, random_state=random_state,
        )
        fit_estimator(estimator, name, X_train, y_train, cat_features=cat_features)

        decision_threshold = DEFAULT_THRESHOLD
        reject_threshold = DEFAULT_REJECT_THRESHOLD
        threshold_ladder = None
        if threshold_optimization and task_type in (TaskType.BINARY, TaskType.MULTICLASS):
            threshold, threshold_ladder = probe_and_optimize_threshold(
                name, opt.best_params, opt.best_n_estimators, task_type, resource_plan, cat_features,
                X_train, y_train, random_state, objective=objective,
                build_ladder=multi_threshold_ensemble, n_ladder_models=threshold_models,
            )
            if task_type is TaskType.MULTICLASS:
                reject_threshold = threshold
                notes.append(f"reject_threshold={reject_threshold:.2f}")
            else:
                decision_threshold = threshold
                notes.append(f"decision_threshold={decision_threshold:.2f}")
            if multi_threshold_ensemble:
                notes.append(f"multi_threshold_ensemble: built a {threshold_models}-point threshold ladder")

        return AutoDecisionResult(
            used_ensemble=False, strategy=name, estimator=estimator, best_params=opt.best_params,
            primary_metric=primary_metric, best_n_estimators=opt.best_n_estimators, ensemble_info=None,
            decision_threshold=decision_threshold, reject_threshold=reject_threshold,
            threshold_ladder=threshold_ladder, notes=notes,
        )

    if rel_diff >= CLEAR_WINNER_THRESHOLD:
        notes.append(
            f"clear winner: {winner_name} (relative diff {rel_diff:.2%} >= {CLEAR_WINNER_THRESHOLD:.1%}); "
            "skipping ensemble entirely"
        )
        return _finalize_single(winner_name)

    tightness = "very close" if rel_diff < CLOSE_PERFORMANCE_THRESHOLD else "moderately close"
    notes.append(f"{tightness} (relative diff {rel_diff:.2%} < {CLEAR_WINNER_THRESHOLD:.1%}); building full ensemble")

    ensemble_result = train_voting_stacking_ensemble(
        X_train, y_train, task_type, profile, resource_plan, cat_features,
        validation=validation, cv=cv, optimizer=optimizer, n_trials=n_trials, timeout=remaining_timeout,
        metrics=metrics, optimize=optimize, random_state=random_state, verbose=verbose, strategy="compare",
        threshold_optimization=threshold_optimization, objective=objective,
        multi_threshold_ensemble=multi_threshold_ensemble, threshold_models=threshold_models,
    )
    notes.extend(ensemble_result.notes)

    best_single_score = score_cb if winner_name == "catboost" else score_lgbm
    if _is_better(ensemble_result.validation_score, best_single_score, direction):
        notes.append(
            f"ensemble ({ensemble_result.strategy}) score {ensemble_result.validation_score:.4f} beats "
            f"best single model ({winner_name}) score {best_single_score:.4f}"
        )
        return AutoDecisionResult(
            used_ensemble=True, strategy=ensemble_result.strategy, estimator=ensemble_result.estimator,
            best_params=ensemble_result.base_params, primary_metric=primary_metric, best_n_estimators=None,
            ensemble_info={
                "strategy": ensemble_result.strategy, "validation_score": ensemble_result.validation_score,
                "base_params": ensemble_result.base_params, "base_n_estimators": ensemble_result.base_n_estimators,
                "base_scores": ensemble_result.base_scores,
            },
            decision_threshold=ensemble_result.decision_threshold, reject_threshold=ensemble_result.reject_threshold,
            threshold_ladder=ensemble_result.threshold_ladder, notes=notes,
        )

    notes.append(f"ensemble did not beat best single model ({winner_name}); falling back to single model")
    return _finalize_single(winner_name)


def _score(ensemble, task_type: TaskType, primary_metric: str, X: pd.DataFrame, y: np.ndarray) -> float:
    if task_type.is_classification and primary_metric == "roc_auc":
        return compute_metric(primary_metric, y, ensemble.predict(X), ensemble.predict_proba(X))
    return compute_metric(primary_metric, y, ensemble.predict(X))


def _score_base_model(model_name: str, estimator, task_type: TaskType, primary_metric: str, X: pd.DataFrame, y: np.ndarray) -> float:
    if task_type.is_classification and primary_metric == "roc_auc":
        return compute_metric(primary_metric, y, predict(estimator, model_name, X), predict_proba(estimator, model_name, X))
    return compute_metric(primary_metric, y, predict(estimator, model_name, X))
