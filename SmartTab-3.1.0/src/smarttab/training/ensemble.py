"""Diversity-aware, leakage-safe OOF ensembles for CatBoost and LightGBM.

The ensemble engine deliberately keeps CatBoost and LightGBM as the core
learners.  It creates a small pool of materially different variants, generates
out-of-fold (OOF) predictions for every candidate, removes redundant candidates,
optimizes soft-voting weights, evaluates OOF stacking, and refits only retained
members on all training rows.  XGBoost is an optional candidate and is retained
only when OOF evidence demonstrates incremental value.
"""

from __future__ import annotations

import importlib.util
import math
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn import metrics as skm
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import train_test_split

from smarttab.analysis.dataset_analyzer import DatasetProfile, TaskType
from smarttab.deadline import FitDeadline
from smarttab.evaluation.evaluator import compute_metric
from smarttab.exceptions import ConfigurationError
from smarttab.hardware.resource_planner import ResourcePlan
from smarttab.optimization.optimizer import (
    METRIC_DIRECTION,
    resolve_class_weight_params,
    resolve_cv_splitter,
    resolve_n_trials,
    resolve_primary_metric,
    run_optimization,
)
from smarttab.optimization.search_spaces import default_params
from smarttab.optimization.threshold import (
    DEFAULT_OBJECTIVE,
    DEFAULT_REJECT_THRESHOLD,
    DEFAULT_THRESHOLD,
    optimize_threshold,
)
from smarttab.training.trainer import build_estimator, fit_estimator, predict, predict_proba

CORE_MODEL_NAMES = ("catboost", "lightgbm")
AUTO_MIN_RELATIVE_IMPROVEMENT = 0.001
XGB_MIN_RELATIVE_IMPROVEMENT = 0.001
DEFAULT_DIVERSITY_CORRELATION_LIMIT = 0.98
DEFAULT_ENSEMBLE_MODELS_LIMIT = 5
DEFAULT_ENSEMBLE_MIN_GAIN = 0.001
WEIGHT_SEARCH_CANDIDATES = 96
OOF_FOLDS = 3
CLASSIFICATION_TOLERANCES = (0.005, 0.003, 0.003, 1e-6)


@dataclass
class BaseModelSpec:
    alias: str
    model_name: str
    params: dict
    n_estimators: int
    optimize_metric: str = "auto"
    random_seed: int = 42
    train_fraction: float = 1.0
    feature_columns: list[str] | None = None
    feature_group: str = "all"
    score: float | None = None
    diagnostics: dict = field(default_factory=dict)


def _unpack_member(entry) -> tuple[str, str, object, list[str] | None]:
    """Read current four-field members and legacy three-field members."""
    if len(entry) == 4:
        alias, model_name, estimator, feature_columns = entry
        return alias, model_name, estimator, feature_columns
    if len(entry) == 3:
        alias, model_name, estimator = entry
        return alias, model_name, estimator, None
    raise ConfigurationError("invalid ensemble member record")


def _member_frame(X: pd.DataFrame, feature_columns: list[str] | None) -> pd.DataFrame:
    if feature_columns is None:
        return X
    missing = [name for name in feature_columns if name not in X.columns]
    if missing:
        raise ConfigurationError(f"ensemble member features are missing: {missing[:8]}")
    return X.loc[:, feature_columns]


class VotingEnsemble:
    """Weighted soft voting for classification and weighted averaging for regression."""

    def __init__(
        self,
        base_models: list[tuple[str, str, object] | tuple[str, str, object, list[str] | None]],
        task_type: TaskType,
        weights: list[float] | None = None,
    ) -> None:
        if not base_models:
            raise ConfigurationError("VotingEnsemble requires at least one base model")
        self.base_models = base_models
        self.task_type = task_type
        self.weights = weights

    def _weights_array(self) -> np.ndarray:
        if self.weights is None:
            return np.full(len(self.base_models), 1.0 / len(self.base_models))
        weights = np.asarray(self.weights, dtype=float)
        if len(weights) != len(self.base_models):
            raise ConfigurationError("voting weight count does not match base model count")
        weights = np.clip(weights, 0.0, None)
        total = weights.sum()
        if total <= 0:
            return np.full(len(weights), 1.0 / len(weights))
        return weights / total

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        probabilities = []
        for entry in self.base_models:
            _alias, model_name, estimator, feature_columns = _unpack_member(entry)
            probabilities.append(
                predict_proba(estimator, model_name, _member_frame(X, feature_columns))
            )
        averaged = np.average(probabilities, axis=0, weights=self._weights_array())
        row_sums = averaged.sum(axis=1, keepdims=True)
        return np.divide(
            averaged,
            row_sums,
            out=np.zeros_like(averaged),
            where=row_sums != 0,
        )

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if self.task_type.is_classification:
            return np.argmax(self.predict_proba(X), axis=1)
        predictions = []
        for entry in self.base_models:
            _alias, model_name, estimator, feature_columns = _unpack_member(entry)
            predictions.append(predict(estimator, model_name, _member_frame(X, feature_columns)))
        return np.average(predictions, axis=0, weights=self._weights_array())


class StackingEnsemble:
    """OOF-trained meta learner over refitted boosting base models."""

    def __init__(
        self,
        base_models: list[tuple[str, str, object] | tuple[str, str, object, list[str] | None]],
        task_type: TaskType,
        meta_model,
        meta_model_name: str,
    ) -> None:
        if not base_models:
            raise ConfigurationError("StackingEnsemble requires at least one base model")
        self.base_models = base_models
        self.task_type = task_type
        self.meta_model = meta_model
        self.meta_model_name = meta_model_name

    def _meta_features(self, X: pd.DataFrame) -> np.ndarray:
        outputs: list[np.ndarray] = []
        for entry in self.base_models:
            _alias, model_name, estimator, feature_columns = _unpack_member(entry)
            member_X = _member_frame(X, feature_columns)
            if self.task_type.is_classification:
                probabilities = predict_proba(estimator, model_name, member_X)
                outputs.append(_classification_meta_block(probabilities, self.task_type))
            else:
                outputs.append(predict(estimator, model_name, member_X).reshape(-1, 1))
        return np.hstack(outputs)

    def _prepared_meta_features(self, X: pd.DataFrame):
        features = self._meta_features(X)
        module = self.meta_model.__class__.__module__
        if module.startswith("catboost") or module.startswith("lightgbm"):
            return pd.DataFrame(
                features,
                columns=[f"meta_{index}" for index in range(features.shape[1])],
            )
        return features

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return np.asarray(self.meta_model.predict_proba(self._prepared_meta_features(X)))

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return np.asarray(self.meta_model.predict(self._prepared_meta_features(X)))


@dataclass
class EnsembleResult:
    strategy: str
    estimator: object
    base_params: dict = field(default_factory=dict)
    base_n_estimators: dict = field(default_factory=dict)
    base_scores: dict = field(default_factory=dict)
    primary_metric: str = ""
    validation_score: float = 0.0
    decision_threshold: float = DEFAULT_THRESHOLD
    reject_threshold: float = DEFAULT_REJECT_THRESHOLD
    notes: list[str] = field(default_factory=list)
    selection_scores: dict[str, float] = field(default_factory=dict)
    selection_vectors: dict[str, list[float]] = field(default_factory=dict)
    meta_model_name: str | None = None
    voting_weights: dict[str, float] = field(default_factory=dict)
    members: list[dict] = field(default_factory=list)
    candidates: list[dict] = field(default_factory=list)
    diversity_matrix: dict[str, dict[str, float]] = field(default_factory=dict)
    fusion_strategy: str = "early"


@dataclass
class AutoDecisionResult:
    used_ensemble: bool
    strategy: str
    estimator: object
    best_params: dict
    primary_metric: str
    best_n_estimators: int | None = None
    ensemble_info: dict | None = None
    decision_threshold: float = DEFAULT_THRESHOLD
    reject_threshold: float = DEFAULT_REJECT_THRESHOLD
    notes: list[str] = field(default_factory=list)


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
    xgboost_policy: str = "auto",
    ensemble_models_limit: int = DEFAULT_ENSEMBLE_MODELS_LIMIT,
    ensemble_min_gain: float = DEFAULT_ENSEMBLE_MIN_GAIN,
    diversity_correlation_limit: float = DEFAULT_DIVERSITY_CORRELATION_LIMIT,
    meta_model: str = "auto",
    feature_groups: dict[str, list[str]] | None = None,
    fusion: str = "auto",
    deadline: FitDeadline | None = None,
) -> EnsembleResult:
    """Train and select a real OOF voting/stacking ensemble.

    Hyperparameter search is performed only for one anchor per core algorithm.
    Additional candidates are deterministic, materially different variants of
    those anchors.  This keeps ``optimize=True`` useful without multiplying
    Optuna cost by every ensemble member.
    """
    _validate_ensemble_arguments(
        strategy,
        task_type,
        ensemble_models_limit,
        ensemble_min_gain,
        diversity_correlation_limit,
        meta_model,
        fusion,
    )
    y_train = np.asarray(y_train)
    primary_metric = resolve_primary_metric(profile, metrics)
    direction = METRIC_DIRECTION[primary_metric]
    preferred_algorithm = _preferred_algorithm(profile, cat_features)
    effective_limit = _effective_model_limit(profile, ensemble_models_limit, deadline)
    resolved_fusion = _resolve_fusion_strategy(fusion, feature_groups)
    notes = [
        "ensemble candidates use OOF predictions; retained members are refitted on all training rows",
        f"ensemble_models_limit={ensemble_models_limit}; effective_limit={effective_limit}",
        f"preferred core learner={preferred_algorithm}",
        f"fusion={fusion}; resolved_fusion={resolved_fusion}",
    ]

    anchor_specs, anchor_notes = _prepare_anchor_specs(
        X_train,
        y_train,
        task_type,
        profile,
        resource_plan,
        cat_features,
        validation,
        cv,
        optimizer,
        n_trials,
        timeout,
        metrics,
        optimize,
        random_state,
        verbose,
        preferred_algorithm,
        effective_limit,
        deadline,
    )
    notes.extend(anchor_notes)

    candidate_specs = _build_candidate_pool(
        anchor_specs,
        y_train,
        task_type,
        profile,
        preferred_algorithm,
        effective_limit,
        random_state,
        feature_groups=feature_groups,
        fusion=resolved_fusion,
    )
    splitter = resolve_cv_splitter(
        profile,
        validation="kfold",
        cv=OOF_FOLDS if cv == "auto" else cv,
        random_state=random_state,
        y=y_train,
    )

    oof_blocks: dict[str, np.ndarray] = {}
    base_scores: dict[str, float] = {}
    for spec in candidate_specs:
        if deadline is not None and deadline.enabled:
            remaining = deadline.remaining() or 0.0
            if oof_blocks and remaining < _minimum_oof_reserve(profile):
                notes.append(
                    f"candidate generation stopped before {spec.alias}: only {remaining:.1f}s remained"
                )
                break
        block, diagnostics = _generate_oof_predictions(
            spec,
            X_train,
            y_train,
            task_type,
            splitter,
            resource_plan,
            cat_features,
            deadline,
        )
        spec.diagnostics.update(diagnostics)
        oof_blocks[spec.alias] = block
        score = _score_from_output(y_train, block, task_type, primary_metric)
        spec.score = score
        base_scores[spec.alias] = score
        notes.append(
            f"{spec.alias}: OOF {primary_metric}={score:.6f}; "
            f"specialization={spec.optimize_metric}"
        )

    candidate_specs = [spec for spec in candidate_specs if spec.alias in oof_blocks]
    if not candidate_specs:
        raise ConfigurationError("the ensemble engine could not produce any OOF candidate")

    xgb_spec, xgb_block, xgb_notes = _evaluate_optional_xgboost(
        X_train,
        y_train,
        task_type,
        profile,
        resource_plan,
        cat_features,
        splitter,
        primary_metric,
        xgboost_policy,
        random_state,
        deadline,
        effective_limit,
    )
    notes.extend(xgb_notes)
    if xgb_spec is not None and xgb_block is not None:
        candidate_specs.append(xgb_spec)
        oof_blocks[xgb_spec.alias] = xgb_block
        score = _score_from_output(y_train, xgb_block, task_type, primary_metric)
        xgb_spec.score = score
        base_scores[xgb_spec.alias] = score

    meta_train_indices, select_indices = _meta_train_select_split(
        y_train,
        task_type,
        random_state,
    )
    diversity_matrix = _prediction_correlation_matrix(oof_blocks)
    retained_specs, selection_notes = _select_diverse_members(
        candidate_specs,
        oof_blocks,
        y_train,
        task_type,
        meta_train_indices,
        select_indices,
        effective_limit,
        ensemble_min_gain,
        diversity_correlation_limit,
        strategy,
        preferred_algorithm,
        xgboost_policy,
    )
    notes.extend(selection_notes)

    aliases = [spec.alias for spec in retained_specs]
    candidate_scores: dict[str, float] = {}
    candidate_vectors: dict[str, list[float]] = {}
    candidate_payloads: dict[str, object] = {}

    best_single_spec = _best_single_candidate(
        candidate_specs,
        oof_blocks,
        y_train,
        task_type,
        meta_train_indices,
        select_indices,
    )
    best_single_name = f"single:{best_single_spec.alias}"
    best_single_threshold = _fit_binary_threshold(
        y_train[meta_train_indices],
        oof_blocks[best_single_spec.alias][meta_train_indices],
        task_type,
    )
    single_vector = _selection_vector(
        y_train[select_indices],
        oof_blocks[best_single_spec.alias][select_indices],
        task_type,
        best_single_threshold,
    )
    candidate_vectors[best_single_name] = list(single_vector)
    candidate_scores[best_single_name] = _score_from_output(
        y_train[select_indices],
        oof_blocks[best_single_spec.alias][select_indices],
        task_type,
        primary_metric,
    )

    if len(retained_specs) >= 2:
        train_blocks = [oof_blocks[alias][meta_train_indices] for alias in aliases]
        select_blocks = [oof_blocks[alias][select_indices] for alias in aliases]
        voting_weights = _optimize_voting_weights(
            train_blocks,
            y_train[meta_train_indices],
            task_type,
            random_state,
        )
        voting_train_output = _weighted_output(train_blocks, voting_weights)
        voting_threshold = _fit_binary_threshold(
            y_train[meta_train_indices],
            voting_train_output,
            task_type,
        )
        voting_select_output = _weighted_output(select_blocks, voting_weights)
        voting_vector = _selection_vector(
            y_train[select_indices],
            voting_select_output,
            task_type,
            voting_threshold,
        )
        candidate_vectors["voting"] = list(voting_vector)
        candidate_scores["voting"] = _score_from_output(
            y_train[select_indices],
            voting_select_output,
            task_type,
            primary_metric,
        )
        candidate_payloads["voting"] = voting_weights

        stacked_oof = _stack_oof_blocks(
            [oof_blocks[alias] for alias in aliases],
            task_type,
        )
        stacker_candidates = _fit_meta_candidates(
            stacked_oof[meta_train_indices],
            y_train[meta_train_indices],
            task_type,
            profile,
            resource_plan,
            random_state,
            meta_model,
            deadline,
        )
        for meta_name, fitted_meta in stacker_candidates.items():
            train_output = _meta_model_output(
                fitted_meta,
                stacked_oof[meta_train_indices],
                task_type,
            )
            stack_threshold = _fit_binary_threshold(
                y_train[meta_train_indices],
                train_output,
                task_type,
            )
            select_output = _meta_model_output(
                fitted_meta,
                stacked_oof[select_indices],
                task_type,
            )
            candidate_name = f"stacking:{meta_name}"
            candidate_vectors[candidate_name] = list(
                _selection_vector(
                    y_train[select_indices],
                    select_output,
                    task_type,
                    stack_threshold,
                )
            )
            candidate_scores[candidate_name] = _score_from_output(
                y_train[select_indices],
                select_output,
                task_type,
                primary_metric,
            )
            candidate_payloads[candidate_name] = fitted_meta

    chosen_name = _choose_strategy(
        strategy,
        candidate_vectors,
        best_single_name,
        ensemble_min_gain,
    )
    notes.append(
        "selection vectors [MCC/F1/Recall/-LogLoss or -RMSE/-MAE/R2]: "
        + ", ".join(
            f"{name}={tuple(round(value, 6) for value in vector)}"
            for name, vector in sorted(candidate_vectors.items())
        )
    )

    if chosen_name.startswith("single:"):
        chosen_alias = chosen_name.split(":", 1)[1]
        chosen_spec = next(spec for spec in candidate_specs if spec.alias == chosen_alias)
        fitted_bases, final_metadata = _fit_final_base_models(
            [chosen_spec],
            X_train,
            y_train,
            task_type,
            resource_plan,
            cat_features,
            deadline,
        )
        decision_threshold, reject_threshold = _thresholds_from_output(
            y_train,
            oof_blocks[chosen_alias],
            task_type,
            threshold_optimization,
            objective,
        )
        candidates_metadata = _candidate_metadata(
            candidate_specs,
            retained_specs,
            base_scores,
            diversity_matrix,
        )
        return EnsembleResult(
            strategy=chosen_spec.model_name,
            estimator=fitted_bases[0][2],
            base_params={chosen_alias: chosen_spec.params},
            base_n_estimators={chosen_alias: chosen_spec.n_estimators},
            base_scores=base_scores,
            primary_metric=primary_metric,
            validation_score=candidate_scores[chosen_name],
            decision_threshold=decision_threshold,
            reject_threshold=reject_threshold,
            notes=notes + ["auto selection retained a single model; no harmful ensemble was forced"],
            selection_scores=candidate_scores,
            selection_vectors=candidate_vectors,
            members=final_metadata,
            candidates=candidates_metadata,
            diversity_matrix=diversity_matrix,
            fusion_strategy=resolved_fusion,
        )

    fitted_bases, final_metadata = _fit_final_base_models(
        retained_specs,
        X_train,
        y_train,
        task_type,
        resource_plan,
        cat_features,
        deadline,
    )
    final_aliases = [spec.alias for spec in retained_specs]

    if chosen_name == "voting":
        final_weights = _optimize_voting_weights(
            [oof_blocks[alias] for alias in final_aliases],
            y_train,
            task_type,
            random_state + 31,
        )
        estimator = VotingEnsemble(fitted_bases, task_type, final_weights.tolist())
        final_oof_output = _weighted_output(
            [oof_blocks[alias] for alias in final_aliases],
            final_weights,
        )
        chosen_strategy = "voting"
        meta_model_name = None
        voting_weight_map = {
            alias: float(weight)
            for alias, weight in zip(final_aliases, final_weights, strict=True)
        }
    else:
        meta_model_name = chosen_name.split(":", 1)[1]
        stacked_oof = _stack_oof_blocks(
            [oof_blocks[alias] for alias in final_aliases],
            task_type,
        )
        final_meta = _build_meta_model(
            meta_model_name,
            task_type,
            profile,
            resource_plan,
            random_state + 777,
        )
        _fit_meta_model(
            final_meta,
            stacked_oof,
            y_train,
            meta_model_name,
            task_type,
            deadline,
        )
        estimator = StackingEnsemble(
            fitted_bases,
            task_type,
            final_meta,
            meta_model_name,
        )
        final_oof_output = _meta_model_output(final_meta, stacked_oof, task_type)
        chosen_strategy = "stacking"
        voting_weight_map = {}

    decision_threshold, reject_threshold = _thresholds_from_output(
        y_train,
        final_oof_output,
        task_type,
        threshold_optimization,
        objective,
    )
    candidates_metadata = _candidate_metadata(
        candidate_specs,
        retained_specs,
        base_scores,
        diversity_matrix,
    )
    return EnsembleResult(
        strategy=chosen_strategy,
        estimator=estimator,
        base_params={spec.alias: spec.params for spec in retained_specs},
        base_n_estimators={spec.alias: spec.n_estimators for spec in retained_specs},
        base_scores=base_scores,
        primary_metric=primary_metric,
        validation_score=candidate_scores[chosen_name],
        decision_threshold=decision_threshold,
        reject_threshold=reject_threshold,
        notes=notes,
        selection_scores=candidate_scores,
        selection_vectors=candidate_vectors,
        meta_model_name=meta_model_name,
        voting_weights=voting_weight_map,
        members=final_metadata,
        candidates=candidates_metadata,
        diversity_matrix=diversity_matrix,
        fusion_strategy=resolved_fusion,
    )


def run_ensemble_decision_engine(*args, **kwargs) -> AutoDecisionResult:
    """Run ``ensemble='auto'`` and expose a single-model-compatible result."""
    kwargs["strategy"] = "compare"
    result = train_voting_stacking_ensemble(*args, **kwargs)
    used_ensemble = result.strategy in {"voting", "stacking"}
    ensemble_info = _ensemble_info(result) if used_ensemble else None
    if used_ensemble:
        best_params = result.base_params
        best_n_estimators = None
    else:
        alias = next(iter(result.base_params))
        best_params = result.base_params[alias]
        best_n_estimators = result.base_n_estimators[alias]
    return AutoDecisionResult(
        used_ensemble=used_ensemble,
        strategy=result.strategy,
        estimator=result.estimator,
        best_params=best_params,
        primary_metric=result.primary_metric,
        best_n_estimators=best_n_estimators,
        ensemble_info=ensemble_info,
        decision_threshold=result.decision_threshold,
        reject_threshold=result.reject_threshold,
        notes=result.notes,
    )


def _ensemble_info(result: EnsembleResult) -> dict:
    return {
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


def _validate_ensemble_arguments(
    strategy: str,
    task_type: TaskType,
    ensemble_models_limit: int,
    ensemble_min_gain: float,
    diversity_correlation_limit: float,
    meta_model: str,
    fusion: str,
) -> None:
    if strategy not in ("voting", "stacking", "compare"):
        raise ConfigurationError("strategy must be 'voting', 'stacking', or 'compare'")
    if task_type not in (TaskType.BINARY, TaskType.MULTICLASS, TaskType.REGRESSION):
        raise ConfigurationError("ensembles support binary, multiclass, and regression tasks")
    if not isinstance(ensemble_models_limit, int) or not 1 <= ensemble_models_limit <= 10:
        raise ConfigurationError("ensemble_models_limit must be an integer between 1 and 10")
    if float(ensemble_min_gain) < 0:
        raise ConfigurationError("ensemble_min_gain must be >= 0")
    if not 0.0 < float(diversity_correlation_limit) <= 1.0:
        raise ConfigurationError("diversity_correlation_limit must be in (0, 1]")
    if meta_model not in ("auto", "catboost", "lightgbm", "linear"):
        raise ConfigurationError("meta_model must be 'auto', 'catboost', 'lightgbm', or 'linear'")
    if fusion not in ("auto", "early", "late", "hybrid"):
        raise ConfigurationError("fusion must be 'auto', 'early', 'late', or 'hybrid'")


def _preferred_algorithm(profile: DatasetProfile, cat_features: list[str]) -> str:
    if profile.n_samples >= 100_000 or profile.n_features >= 200:
        return "lightgbm"
    if cat_features or profile.categorical_columns:
        return "catboost"
    return "catboost" if profile.n_samples < 25_000 else "lightgbm"


def _effective_model_limit(
    profile: DatasetProfile,
    requested: int,
    deadline: FitDeadline | None,
) -> int:
    limit = requested
    if profile.n_samples >= 500_000:
        limit = min(limit, 2)
    elif profile.n_samples >= 100_000:
        limit = min(limit, 3)
    if deadline is not None and deadline.enabled:
        remaining = deadline.remaining() or 0.0
        if remaining < 40:
            limit = min(limit, 2)
        elif remaining < 90:
            limit = min(limit, 3)
    return max(1, limit)


def _minimum_oof_reserve(profile: DatasetProfile) -> float:
    if profile.n_samples >= 100_000:
        return 12.0
    if profile.n_samples >= 10_000:
        return 7.0
    return 3.0


def _prepare_anchor_specs(
    X: pd.DataFrame,
    y: np.ndarray,
    task_type: TaskType,
    profile: DatasetProfile,
    resource_plan: ResourcePlan,
    cat_features: list[str],
    validation: str,
    cv: str | int,
    optimizer: str,
    n_trials: str | int,
    timeout: float | None,
    metrics: str,
    optimize: bool,
    random_state: int,
    verbose: int,
    preferred_algorithm: str,
    effective_limit: int,
    deadline: FitDeadline | None,
) -> tuple[dict[str, BaseModelSpec], list[str]]:
    notes: list[str] = []
    algorithms = [preferred_algorithm]
    other = "lightgbm" if preferred_algorithm == "catboost" else "catboost"
    # A second core algorithm is valuable for diversity, but CatBoost is not
    # unconditionally forced on massive data where its OOF cost can dominate.
    allow_other = effective_limit >= 2 and not (
        profile.n_samples >= 500_000 and other == "catboost" and not cat_features
    )
    if allow_other:
        algorithms.append(other)
    elif other == "catboost":
        notes.append("CatBoost anchor skipped on massive numeric data; LightGBM is not forced to share the budget")

    total_trials = resolve_n_trials(profile, resource_plan, n_trials) if optimize else 0
    allocations = _allocate_trials(total_trials, len(algorithms)) if optimize else [0] * len(algorithms)
    anchors: dict[str, BaseModelSpec] = {}
    for index, model_name in enumerate(algorithms):
        if deadline is not None:
            deadline.require(f"preparing ensemble anchor {model_name}")
        if allocations[index] > 0:
            result = run_optimization(
                model_name,
                X,
                y,
                task_type,
                profile,
                resource_plan,
                cat_features,
                validation=validation,
                cv=cv,
                optimizer=optimizer,
                n_trials=allocations[index],
                timeout=timeout,
                metrics=metrics,
                random_state=random_state + index * 101,
                verbose=verbose,
                deadline=deadline,
            )
            params = result.best_params
            n_estimators = result.best_n_estimators
            notes.extend(f"{model_name}: {note}" for note in result.notes)
            notes.append(
                f"{model_name}: optimize=True ran {result.n_trials_run} trial(s); "
                f"optimized_params_retained={result.used_optimized_params}"
            )
        else:
            params = {
                **default_params(model_name),
                **resolve_class_weight_params(model_name, profile),
            }
            n_estimators = _default_ensemble_estimators(profile, model_name)
            notes.append(f"{model_name}: optimize=False; no Optuna study was created")
        alias = model_name
        anchors[model_name] = BaseModelSpec(
            alias=alias,
            model_name=model_name,
            params=dict(params),
            n_estimators=int(n_estimators),
            optimize_metric=resolve_primary_metric(profile, metrics),
            random_seed=random_state + index * 101,
        )
    return anchors, notes


def _default_ensemble_estimators(profile: DatasetProfile, model_name: str) -> int:
    if profile.n_samples >= 250_000:
        return 160 if model_name == "lightgbm" else 100
    if profile.n_samples >= 75_000:
        return 220
    return 300


def _resolve_fusion_strategy(
    requested: str,
    feature_groups: dict[str, list[str]] | None,
) -> str:
    if requested != "auto":
        if requested == "late" and not _usable_specialist_groups(feature_groups):
            return "early"
        return requested
    groups = _usable_specialist_groups(feature_groups)
    return "hybrid" if len(groups) >= 2 else "early"


def _usable_specialist_groups(
    feature_groups: dict[str, list[str]] | None,
) -> dict[str, list[str]]:
    if not feature_groups:
        return {}
    all_count = len(feature_groups.get("all", []))
    result: dict[str, list[str]] = {}
    for name, columns in feature_groups.items():
        if name != "tabular" and not name.startswith("modality:"):
            continue
        unique = list(dict.fromkeys(columns))
        if len(unique) < 4:
            continue
        # A group covering the complete matrix is not a late-fusion specialist.
        if all_count and len(unique) >= all_count:
            continue
        result[name] = unique
    return result


def _build_feature_specialists(
    anchors: dict[str, BaseModelSpec],
    feature_groups: dict[str, list[str]] | None,
    y: np.ndarray,
    task_type: TaskType,
    profile: DatasetProfile,
    preferred_algorithm: str,
    random_state: int,
    maximum: int = 4,
) -> list[BaseModelSpec]:
    groups = _usable_specialist_groups(feature_groups)
    specialists: list[BaseModelSpec] = []
    all_count = max(1, len((feature_groups or {}).get("all", [])))
    ordered = sorted(
        groups.items(),
        key=lambda item: (0 if item[0] == "tabular" else 1, -len(item[1]), item[0]),
    )
    for index, (group_name, columns) in enumerate(ordered[:maximum]):
        if group_name == "tabular":
            model_name = preferred_algorithm
        else:
            # Dense generated representations are usually the faster LightGBM
            # path; CatBoost remains available when LightGBM was not prepared.
            model_name = "lightgbm" if "lightgbm" in anchors else preferred_algorithm
        if model_name not in anchors:
            model_name = next(iter(anchors))
        anchor = anchors[model_name]
        specialty = "regularized" if group_name == "tabular" else "diversity"
        params, train_fraction = _variant_params(
            model_name,
            anchor.params,
            specialty,
            y,
            task_type,
            profile,
            index + 1,
        )
        label = group_name.replace(":", "_").replace("/", "_")
        specialists.append(
            BaseModelSpec(
                alias=f"{model_name}_{label}_specialist",
                model_name=model_name,
                params=params,
                n_estimators=max(80, int(round(anchor.n_estimators * 0.9))),
                optimize_metric=f"specialist:{group_name}",
                random_seed=random_state + 5003 + index * 307,
                train_fraction=train_fraction,
                feature_columns=columns,
                feature_group=group_name,
                diagnostics={
                    "specialist_group": group_name,
                    "feature_count": len(columns),
                    "feature_share": len(columns) / all_count,
                },
            )
        )
    return specialists


def _build_candidate_pool(
    anchors: dict[str, BaseModelSpec],
    y: np.ndarray,
    task_type: TaskType,
    profile: DatasetProfile,
    preferred_algorithm: str,
    effective_limit: int,
    random_state: int,
    *,
    feature_groups: dict[str, list[str]] | None = None,
    fusion: str = "early",
) -> list[BaseModelSpec]:
    anchor_order = [
        model_name
        for model_name in (
            preferred_algorithm,
            "lightgbm" if preferred_algorithm == "catboost" else "catboost",
        )
        if model_name in anchors
    ]
    full_specs = [anchors[name] for name in anchor_order]
    specialists = _build_feature_specialists(
        anchors,
        feature_groups,
        y,
        task_type,
        profile,
        preferred_algorithm,
        random_state,
        maximum=min(4, max(1, effective_limit)),
    )

    if fusion == "late" and specialists:
        # Keep one complete-view safety anchor so auto-selection can prove that
        # late fusion is useful instead of forcing it blindly.
        pool = list(full_specs[:1]) + list(specialists)
    elif fusion == "hybrid":
        pool = list(full_specs) + list(specialists)
    else:
        pool = list(full_specs)

    candidate_budget = min(12, max(effective_limit + 3, len(pool)))
    specialties = _specialties(task_type)
    preferred_target = max(1, math.ceil(candidate_budget * 0.7))
    preferred_count = sum(spec.model_name == preferred_algorithm for spec in pool)
    other_algorithm = "lightgbm" if preferred_algorithm == "catboost" else "catboost"
    index = 0
    while len(pool) < candidate_budget:
        specialty = specialties[index % len(specialties)]
        if fusion == "late" and specialists:
            base = specialists[index % len(specialists)]
            model_name = base.model_name
            anchor = anchors[model_name]
            feature_columns = base.feature_columns
            feature_group = base.feature_group
        else:
            if preferred_count < preferred_target or other_algorithm not in anchors:
                model_name = preferred_algorithm
            else:
                model_name = other_algorithm
            if model_name not in anchors:
                model_name = preferred_algorithm
            anchor = anchors[model_name]
            feature_columns = None
            feature_group = "all"
        variant_number = 1 + sum(spec.model_name == model_name for spec in pool)
        params, train_fraction = _variant_params(
            model_name,
            anchor.params,
            specialty,
            y,
            task_type,
            profile,
            variant_number,
        )
        estimator_scale = 0.85 + 0.08 * (variant_number % 4)
        n_estimators = max(80, min(600, int(round(anchor.n_estimators * estimator_scale))))
        group_label = "" if feature_group == "all" else f"_{feature_group.replace(':', '_')}"
        alias = f"{model_name}{group_label}_{specialty}_{variant_number}"
        pool.append(
            BaseModelSpec(
                alias=alias,
                model_name=model_name,
                params=params,
                n_estimators=n_estimators,
                optimize_metric=specialty,
                random_seed=random_state + len(pool) * 997,
                train_fraction=train_fraction,
                feature_columns=feature_columns,
                feature_group=feature_group,
                diagnostics={
                    "specialist_group": feature_group,
                    "feature_count": len(feature_columns) if feature_columns else None,
                },
            )
        )
        preferred_count += int(model_name == preferred_algorithm)
        index += 1
    return pool


def _specialties(task_type: TaskType) -> tuple[str, ...]:
    if task_type is TaskType.BINARY:
        return ("mcc", "f1", "recall", "precision", "calibration", "diversity")
    if task_type is TaskType.MULTICLASS:
        return ("mcc", "f1_macro", "recall_macro", "calibration", "diversity")
    return ("rmse", "mae", "regularized", "diversity")


def _variant_params(
    model_name: str,
    anchor_params: dict,
    specialty: str,
    y: np.ndarray,
    task_type: TaskType,
    profile: DatasetProfile,
    variant_number: int,
) -> tuple[dict, float]:
    params = dict(anchor_params)
    train_fraction = 1.0
    ratio = _negative_positive_ratio(y) if task_type is TaskType.BINARY else 1.0

    if model_name == "catboost":
        params.pop("class_weights", None)
        params.pop("auto_class_weights", None)
        params.pop("bootstrap_type", None)
        params.pop("subsample", None)
        params.pop("bagging_temperature", None)
        if task_type.is_classification and profile.is_imbalanced:
            params["auto_class_weights"] = "Balanced"
        if specialty == "recall" and task_type is TaskType.BINARY:
            params.pop("auto_class_weights", None)
            params["class_weights"] = [1.0, max(1.0, ratio * 1.35)]
            params["depth"] = min(9, int(params.get("depth", 6)) + 1)
        elif specialty == "precision" and task_type is TaskType.BINARY:
            params.pop("auto_class_weights", None)
            params["class_weights"] = [1.0, max(1.0, ratio * 0.7)]
            params["l2_leaf_reg"] = max(4.0, float(params.get("l2_leaf_reg", 3.0)) * 1.5)
        elif specialty in {"f1", "f1_macro", "mcc"}:
            params["random_strength"] = max(0.2, float(params.get("random_strength", 0.5)) * 1.4)
        elif specialty == "calibration":
            params["depth"] = max(3, int(params.get("depth", 6)) - 2)
            params["learning_rate"] = max(0.02, float(params.get("learning_rate", 0.06)) * 0.75)
            params["l2_leaf_reg"] = max(6.0, float(params.get("l2_leaf_reg", 3.0)) * 2.0)
        elif specialty == "mae" and task_type is TaskType.REGRESSION:
            params["loss_function"] = "MAE"
        elif specialty == "regularized":
            params["depth"] = max(3, int(params.get("depth", 6)) - 1)
            params["l2_leaf_reg"] = max(8.0, float(params.get("l2_leaf_reg", 3.0)) * 2.0)
        elif specialty == "diversity":
            params["bootstrap_type"] = "Bernoulli"
            params["subsample"] = 0.72 + 0.04 * (variant_number % 3)
            params["random_strength"] = max(1.2, float(params.get("random_strength", 0.5)) * 2.2)
            params["depth"] = max(4, int(params.get("depth", 6)) - 1)
            train_fraction = 0.9
    else:
        params.pop("class_weight", None)
        if task_type.is_classification and profile.is_imbalanced:
            params["class_weight"] = "balanced"
        if specialty == "recall" and task_type is TaskType.BINARY:
            params["class_weight"] = {0: 1.0, 1: max(1.0, ratio * 1.35)}
            params["num_leaves"] = min(128, int(params.get("num_leaves", 31) * 1.35))
        elif specialty == "precision" and task_type is TaskType.BINARY:
            params["class_weight"] = {0: 1.0, 1: max(1.0, ratio * 0.7)}
            params["min_child_samples"] = max(25, int(params.get("min_child_samples", 20) * 1.5))
        elif specialty in {"f1", "f1_macro", "mcc"}:
            params["subsample"] = 0.9
            params["colsample_bytree"] = 0.9
        elif specialty == "calibration":
            params["num_leaves"] = min(31, int(params.get("num_leaves", 31)))
            params["learning_rate"] = max(0.02, float(params.get("learning_rate", 0.06)) * 0.75)
            params["reg_lambda"] = max(2.0, float(params.get("reg_lambda", 0.1)) * 10.0)
        elif specialty == "mae" and task_type is TaskType.REGRESSION:
            params["objective"] = "regression_l1"
        elif specialty == "regularized":
            params["num_leaves"] = max(12, int(params.get("num_leaves", 31) * 0.65))
            params["min_child_samples"] = max(35, int(params.get("min_child_samples", 20) * 1.8))
            params["reg_lambda"] = max(2.0, float(params.get("reg_lambda", 0.1)) * 10.0)
        elif specialty == "diversity":
            params["subsample"] = 0.72 + 0.04 * (variant_number % 3)
            params["colsample_bytree"] = 0.68 + 0.05 * (variant_number % 3)
            params["extra_trees"] = True
            params["min_child_samples"] = max(30, int(params.get("min_child_samples", 20) * 1.5))
            train_fraction = 0.9
    return params, train_fraction


def _negative_positive_ratio(y: np.ndarray) -> float:
    values, counts = np.unique(y, return_counts=True)
    if len(values) != 2:
        return 1.0
    positive_count = counts[int(np.argmax(values))]
    negative_count = counts.sum() - positive_count
    return float(max(1.0, negative_count / max(positive_count, 1)))


def _generate_oof_predictions(
    spec: BaseModelSpec,
    X: pd.DataFrame,
    y: np.ndarray,
    task_type: TaskType,
    splitter,
    resource_plan: ResourcePlan,
    cat_features: list[str],
    deadline: FitDeadline | None,
) -> tuple[np.ndarray, dict]:
    if task_type is TaskType.REGRESSION:
        oof = np.zeros(len(X), dtype=float)
    else:
        n_classes = len(np.unique(y))
        oof = np.zeros((len(X), n_classes), dtype=float)
    training_seconds = 0.0
    inference_seconds = 0.0
    train_sizes: list[int] = []
    model_X = _member_frame(X, spec.feature_columns)
    model_cat_features = [name for name in cat_features if name in model_X.columns]

    for fold_index, (train_index, valid_index) in enumerate(splitter.split(model_X, y)):
        if deadline is not None:
            deadline.require(f"OOF {spec.alias} fold {fold_index + 1}")
        sampled_train = _sample_training_indices(
            np.asarray(train_index),
            y,
            task_type,
            spec.train_fraction,
            spec.random_seed + fold_index,
        )
        train_sizes.append(len(sampled_train))
        estimator = build_estimator(
            spec.model_name,
            spec.params,
            task_type,
            spec.n_estimators,
            resource_plan.cpu_threads,
            resource_plan.use_gpu,
            spec.random_seed + fold_index,
            resource_plan=resource_plan,
        )
        started = time.perf_counter()
        fit_estimator(
            estimator,
            spec.model_name,
            model_X.iloc[sampled_train],
            y[sampled_train],
            model_X.iloc[valid_index],
            y[valid_index],
            cat_features=model_cat_features if spec.model_name != "xgboost" else [],
            early_stopping_rounds=30,
            deadline=deadline,
            resource_plan=resource_plan,
        )
        training_seconds += time.perf_counter() - started
        started = time.perf_counter()
        if task_type.is_classification:
            oof[valid_index] = predict_proba(
                estimator, spec.model_name, model_X.iloc[valid_index]
            )
        else:
            oof[valid_index] = predict(estimator, spec.model_name, model_X.iloc[valid_index])
        inference_seconds += time.perf_counter() - started

    diagnostics = {
        "algorithm": spec.model_name,
        "optimize_metric": spec.optimize_metric,
        "random_seed": spec.random_seed,
        "train_fraction": spec.train_fraction,
        "mean_fold_train_size": float(np.mean(train_sizes)) if train_sizes else 0.0,
        "oof_training_seconds": training_seconds,
        "oof_inference_seconds": inference_seconds,
        "bootstrap": _bootstrap_description(spec.model_name, spec.params),
        "feature_subsampling": _feature_subsampling(spec.model_name, spec.params),
        "hyperparameters": dict(spec.params),
        "n_estimators": spec.n_estimators,
        "feature_group": spec.feature_group,
        "feature_count": model_X.shape[1],
        "feature_columns": list(spec.feature_columns) if spec.feature_columns is not None else None,
    }
    return oof, diagnostics


def _sample_training_indices(
    train_index: np.ndarray,
    y: np.ndarray,
    task_type: TaskType,
    fraction: float,
    random_state: int,
) -> np.ndarray:
    if fraction >= 0.999 or len(train_index) < 50:
        return train_index
    stratify = y[train_index] if task_type.is_classification else None
    try:
        sampled, _ = train_test_split(
            train_index,
            train_size=fraction,
            random_state=random_state,
            stratify=stratify,
        )
        return np.asarray(sampled)
    except ValueError:
        return train_index


def _bootstrap_description(model_name: str, params: dict) -> dict:
    if model_name == "catboost":
        return {
            "type": params.get("bootstrap_type", "Bayesian/default"),
            "subsample": params.get("subsample"),
            "bagging_temperature": params.get("bagging_temperature"),
        }
    return {
        "type": "row_subsample" if float(params.get("subsample", 1.0)) < 1.0 else "none/default",
        "subsample": float(params.get("subsample", 1.0)),
    }


def _feature_subsampling(model_name: str, params: dict) -> float:
    if model_name == "catboost":
        return float(params.get("rsm", 1.0))
    return float(params.get("colsample_bytree", 1.0))


def _evaluate_optional_xgboost(
    X: pd.DataFrame,
    y: np.ndarray,
    task_type: TaskType,
    profile: DatasetProfile,
    resource_plan: ResourcePlan,
    cat_features: list[str],
    splitter,
    primary_metric: str,
    policy: str,
    random_state: int,
    deadline: FitDeadline | None,
    effective_limit: int,
) -> tuple[BaseModelSpec | None, np.ndarray | None, list[str]]:
    if policy == "never":
        return None, None, ["XGBoost policy=never; skipped"]
    if effective_limit < 3 and policy == "auto":
        return None, None, ["XGBoost skipped: effective ensemble limit is below three models"]
    if importlib.util.find_spec("xgboost") is None:
        if policy == "always":
            raise ConfigurationError("xgboost_policy='always' requires the 'ensemble-xgb' extra")
        return None, None, ["XGBoost not installed; core CatBoost/LightGBM path remains complete"]
    if cat_features:
        if policy == "always":
            raise ConfigurationError(
                "xgboost_policy='always' is incompatible with native categorical features"
            )
        return None, None, ["XGBoost skipped: native categorical semantics would be lost"]
    if policy == "auto" and not (5_000 <= len(X) <= 500_000):
        return None, None, ["XGBoost skipped: dataset size is outside its cost-effective candidate range"]
    if deadline is not None and deadline.enabled and (deadline.remaining() or 0.0) < 12:
        if policy == "always":
            deadline.require("XGBoost OOF evaluation")
        return None, None, ["XGBoost skipped: insufficient global time budget for OOF proof"]

    spec = BaseModelSpec(
        alias="xgboost_diversity",
        model_name="xgboost",
        params={
            **default_params("xgboost"),
            "subsample": 0.8,
            "colsample_bytree": 0.75,
        },
        n_estimators=_default_ensemble_estimators(profile, "lightgbm"),
        optimize_metric=primary_metric,
        random_seed=random_state + 90_000,
    )
    block, diagnostics = _generate_oof_predictions(
        spec,
        X,
        y,
        task_type,
        splitter,
        resource_plan,
        [],
        deadline,
    )
    spec.diagnostics.update(diagnostics)
    return spec, block, ["XGBoost evaluated as an optional OOF diversity candidate"]


def _prediction_correlation_matrix(blocks: dict[str, np.ndarray]) -> dict[str, dict[str, float]]:
    aliases = list(blocks)
    matrix: dict[str, dict[str, float]] = {alias: {} for alias in aliases}
    for left in aliases:
        for right in aliases:
            if left == right:
                value = 1.0
            else:
                value = _prediction_correlation(blocks[left], blocks[right])
            matrix[left][right] = float(value)
    return matrix


def _prediction_correlation(left: np.ndarray, right: np.ndarray) -> float:
    left_vector = _correlation_vector(left)
    right_vector = _correlation_vector(right)
    if np.allclose(left_vector, left_vector[0]) or np.allclose(right_vector, right_vector[0]):
        return 1.0 if np.allclose(left_vector, right_vector) else 0.0
    value = np.corrcoef(left_vector, right_vector)[0, 1]
    return float(abs(value)) if np.isfinite(value) else 0.0


def _correlation_vector(output: np.ndarray) -> np.ndarray:
    array = np.asarray(output, dtype=float)
    if array.ndim == 2 and array.shape[1] == 2:
        return array[:, 1]
    return array.ravel()


def _select_diverse_members(
    specs: list[BaseModelSpec],
    blocks: dict[str, np.ndarray],
    y: np.ndarray,
    task_type: TaskType,
    train_indices: np.ndarray,
    select_indices: np.ndarray,
    limit: int,
    min_gain: float,
    correlation_limit: float,
    strategy: str,
    preferred_algorithm: str,
    xgboost_policy: str,
) -> tuple[list[BaseModelSpec], list[str]]:
    notes: list[str] = []
    best_single = _best_single_candidate(
        specs,
        blocks,
        y,
        task_type,
        train_indices,
        select_indices,
    )
    selected = [best_single]
    remaining = [spec for spec in specs if spec.alias != best_single.alias]
    current_train = blocks[best_single.alias][train_indices]
    current_select = blocks[best_single.alias][select_indices]
    current_threshold = _fit_binary_threshold(y[train_indices], current_train, task_type)
    current_vector = _selection_vector(
        y[select_indices],
        current_select,
        task_type,
        current_threshold,
    )

    while remaining and len(selected) < limit:
        best_trial = None
        best_vector = None
        best_weights = None
        best_corr = None
        candidate_pool = remaining
        if strategy in {"voting", "stacking"} and len(selected) == 1:
            other_core = [
                candidate for candidate in remaining
                if candidate.model_name in {"catboost", "lightgbm"}
                and candidate.model_name != selected[0].model_name
                and candidate.feature_group == "all"
            ]
            if other_core:
                candidate_pool = other_core
        for candidate in candidate_pool:
            aliases = [spec.alias for spec in selected] + [candidate.alias]
            train_outputs = [blocks[alias][train_indices] for alias in aliases]
            select_outputs = [blocks[alias][select_indices] for alias in aliases]
            weights = _optimize_voting_weights(
                train_outputs,
                y[train_indices],
                task_type,
                candidate.random_seed,
            )
            train_output = _weighted_output(train_outputs, weights)
            threshold = _fit_binary_threshold(y[train_indices], train_output, task_type)
            select_output = _weighted_output(select_outputs, weights)
            vector = _selection_vector(y[select_indices], select_output, task_type, threshold)
            max_corr = max(
                _prediction_correlation(blocks[candidate.alias], blocks[item.alias])
                for item in selected
            )
            improves = _materially_better(vector, current_vector, min_gain, task_type)
            force_second_candidate = strategy in {"voting", "stacking"} and len(selected) == 1
            if max_corr > correlation_limit and not improves and not force_second_candidate:
                continue
            if best_vector is None or _compare_vectors(vector, best_vector, task_type) > 0:
                best_trial = candidate
                best_vector = vector
                best_weights = weights
                best_corr = max_corr

        if best_trial is None:
            notes.append("candidate growth stopped: remaining predictions were redundant or harmful")
            break
        improves = _materially_better(best_vector, current_vector, min_gain, task_type)
        force_second = strategy in {"voting", "stacking"} and len(selected) == 1
        if not improves and not force_second:
            notes.append(
                f"candidate growth stopped: {best_trial.alias} improved less than ensemble_min_gain={min_gain}"
            )
            break
        selected.append(best_trial)
        remaining.remove(best_trial)
        current_vector = best_vector
        aliases = [spec.alias for spec in selected]
        current_train = _weighted_output(
            [blocks[alias][train_indices] for alias in aliases],
            best_weights,
        )
        current_select = _weighted_output(
            [blocks[alias][select_indices] for alias in aliases],
            best_weights,
        )
        notes.append(
            f"retained {best_trial.alias}: max prediction correlation={best_corr:.4f}; "
            f"selection_vector={tuple(round(value, 6) for value in best_vector)}"
        )

    if xgboost_policy == "always":
        xgb = next((spec for spec in specs if spec.model_name == "xgboost"), None)
        if xgb is not None and all(spec.alias != xgb.alias for spec in selected):
            if len(selected) >= limit:
                selected[-1] = xgb
            else:
                selected.append(xgb)
            notes.append("xgboost_policy='always': XGBoost retained explicitly")

    if len(selected) >= 3:
        preferred_share = sum(spec.model_name == preferred_algorithm for spec in selected) / len(selected)
        notes.append(
            f"retained preferred-algorithm share={preferred_share:.1%}; accuracy/diversity evidence takes precedence over a rigid quota"
        )
    return selected, notes


def _best_single_candidate(
    specs: list[BaseModelSpec],
    blocks: dict[str, np.ndarray],
    y: np.ndarray,
    task_type: TaskType,
    train_indices: np.ndarray,
    select_indices: np.ndarray,
) -> BaseModelSpec:
    eligible = [spec for spec in specs if spec.feature_group == "all"] or specs
    best_spec = eligible[0]
    best_vector = None
    for spec in eligible:
        train_output = blocks[spec.alias][train_indices]
        threshold = _fit_binary_threshold(y[train_indices], train_output, task_type)
        vector = _selection_vector(
            y[select_indices],
            blocks[spec.alias][select_indices],
            task_type,
            threshold,
        )
        if best_vector is None or _compare_vectors(vector, best_vector, task_type) > 0:
            best_spec = spec
            best_vector = vector
    return best_spec


def _choose_strategy(
    requested: str,
    vectors: dict[str, list[float]],
    best_single_name: str,
    min_gain: float,
) -> str:
    if requested == "voting":
        return "voting" if "voting" in vectors else best_single_name
    if requested == "stacking":
        stackers = [name for name in vectors if name.startswith("stacking:")]
        if not stackers:
            return best_single_name
        return _best_vector_name({name: vectors[name] for name in stackers})

    chosen = _best_vector_name(vectors)
    if chosen == best_single_name:
        return chosen
    task_type = TaskType.REGRESSION if len(vectors[best_single_name]) == 3 else TaskType.BINARY
    if not _materially_better(
        tuple(vectors[chosen]),
        tuple(vectors[best_single_name]),
        min_gain,
        task_type,
    ):
        return best_single_name
    return chosen


def _best_vector_name(vectors: dict[str, list[float]]) -> str:
    names = list(vectors)
    best = names[0]
    # Vector length disambiguates regression (3) from classification (4).
    task_type = TaskType.REGRESSION if len(vectors[best]) == 3 else TaskType.BINARY
    for name in names[1:]:
        if _compare_vectors(tuple(vectors[name]), tuple(vectors[best]), task_type) > 0:
            best = name
    return best


def _selection_vector(
    y_true: np.ndarray,
    output: np.ndarray,
    task_type: TaskType,
    threshold: float = DEFAULT_THRESHOLD,
) -> tuple[float, ...]:
    y_true = np.asarray(y_true)
    array = np.asarray(output)
    if task_type is TaskType.REGRESSION:
        prediction = array.ravel()
        rmse = float(np.sqrt(skm.mean_squared_error(y_true, prediction)))
        mae = float(skm.mean_absolute_error(y_true, prediction))
        r2 = float(skm.r2_score(y_true, prediction))
        return (-rmse, -mae, r2)

    if task_type is TaskType.BINARY:
        probability = np.clip(array[:, 1].astype(float), 1e-12, 1.0 - 1e-12)
        prediction = probability >= threshold
        positive = y_true == 1
        tp = float(np.sum(prediction & positive))
        tn = float(np.sum(~prediction & ~positive))
        fp = float(np.sum(prediction & ~positive))
        fn = float(np.sum(~prediction & positive))
        denominator = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
        mcc = ((tp * tn) - (fp * fn)) / denominator if denominator else 0.0
        f1_denominator = 2.0 * tp + fp + fn
        f1 = 2.0 * tp / f1_denominator if f1_denominator else 0.0
        recall_denominator = tp + fn
        recall = tp / recall_denominator if recall_denominator else 0.0
        negative_log_loss = float(
            np.mean(y_true * np.log(probability) + (1 - y_true) * np.log(1.0 - probability))
        )
        return (float(mcc), float(f1), float(recall), negative_log_loss)

    prediction = np.argmax(array, axis=1)
    mcc = float(skm.matthews_corrcoef(y_true, prediction))
    f1 = float(skm.f1_score(y_true, prediction, average="macro", zero_division=0))
    recall = float(skm.recall_score(y_true, prediction, average="macro", zero_division=0))
    try:
        negative_log_loss = -float(
            skm.log_loss(y_true, array, labels=np.arange(array.shape[1]))
        )
    except ValueError:
        negative_log_loss = -float("inf")
    return (mcc, f1, recall, negative_log_loss)


def _fit_binary_threshold(
    y_true: np.ndarray,
    output: np.ndarray,
    task_type: TaskType,
) -> float:
    if task_type is not TaskType.BINARY:
        return DEFAULT_THRESHOLD
    probabilities = np.asarray(output, dtype=float)[:, 1]
    thresholds = np.linspace(0.05, 0.95, 37)
    predictions = probabilities[:, None] >= thresholds[None, :]
    positive = np.asarray(y_true)[:, None] == 1
    tp = np.sum(predictions & positive, axis=0).astype(float)
    tn = np.sum(~predictions & ~positive, axis=0).astype(float)
    fp = np.sum(predictions & ~positive, axis=0).astype(float)
    fn = np.sum(~predictions & positive, axis=0).astype(float)
    denominator = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    scores = np.divide(
        tp * tn - fp * fn,
        denominator,
        out=np.zeros_like(denominator),
        where=denominator > 0,
    )
    return float(thresholds[int(np.argmax(scores))])


def _compare_vectors(
    left: tuple[float, ...],
    right: tuple[float, ...],
    task_type: TaskType,
) -> int:
    tolerances = (0.0, 0.0, 0.0) if task_type is TaskType.REGRESSION else CLASSIFICATION_TOLERANCES
    for left_value, right_value, tolerance in zip(left, right, tolerances, strict=True):
        difference = left_value - right_value
        if abs(difference) > tolerance:
            return 1 if difference > 0 else -1
    return 0


def _materially_better(
    candidate: tuple[float, ...],
    baseline: tuple[float, ...],
    minimum_gain: float,
    task_type: TaskType,
) -> bool:
    if _compare_vectors(candidate, baseline, task_type) <= 0:
        return False
    if task_type is TaskType.REGRESSION:
        return candidate[0] - baseline[0] >= minimum_gain
    if candidate[0] - baseline[0] >= minimum_gain:
        return True
    if abs(candidate[0] - baseline[0]) <= CLASSIFICATION_TOLERANCES[0]:
        if candidate[1] - baseline[1] >= minimum_gain:
            return True
        if abs(candidate[1] - baseline[1]) <= CLASSIFICATION_TOLERANCES[1]:
            if candidate[2] - baseline[2] >= minimum_gain:
                return True
            if abs(candidate[2] - baseline[2]) <= CLASSIFICATION_TOLERANCES[2]:
                return candidate[3] > baseline[3]
    return False


def _fit_final_base_models(
    specs: list[BaseModelSpec],
    X: pd.DataFrame,
    y: np.ndarray,
    task_type: TaskType,
    resource_plan: ResourcePlan,
    cat_features: list[str],
    deadline: FitDeadline | None,
) -> tuple[list[tuple[str, str, object, list[str] | None]], list[dict]]:
    fitted: list[tuple[str, str, object, list[str] | None]] = []
    metadata: list[dict] = []
    for spec in specs:
        if deadline is not None:
            deadline.require(f"final fit for {spec.alias}")
        estimator = build_estimator(
            spec.model_name,
            spec.params,
            task_type,
            spec.n_estimators,
            resource_plan.cpu_threads,
            resource_plan.use_gpu,
            spec.random_seed,
            resource_plan=resource_plan,
        )
        model_X = _member_frame(X, spec.feature_columns)
        model_cat_features = [name for name in cat_features if name in model_X.columns]
        inference_sample = model_X.iloc[: min(len(model_X), 512)]
        started = time.perf_counter()
        fit_estimator(
            estimator,
            spec.model_name,
            model_X,
            y,
            cat_features=model_cat_features if spec.model_name != "xgboost" else [],
            deadline=deadline,
            resource_plan=resource_plan,
        )
        training_seconds = time.perf_counter() - started
        started = time.perf_counter()
        if task_type.is_classification:
            predict_proba(estimator, spec.model_name, inference_sample)
        else:
            predict(estimator, spec.model_name, inference_sample)
        inference_seconds = time.perf_counter() - started
        model_size_mb = _estimate_model_size_mb(estimator, spec.model_name)
        fitted.append(
            (
                spec.alias,
                spec.model_name,
                estimator,
                list(spec.feature_columns) if spec.feature_columns is not None else None,
            )
        )
        member = {
            **spec.diagnostics,
            "alias": spec.alias,
            "algorithm": spec.model_name,
            "optimize_metric": spec.optimize_metric,
            "hyperparameters": dict(spec.params),
            "random_seed": spec.random_seed,
            "train_sample_size": len(model_X),
            "feature_group": spec.feature_group,
            "feature_count": model_X.shape[1],
            "feature_columns": list(spec.feature_columns) if spec.feature_columns is not None else None,
            "training_seconds": training_seconds,
            "inference_seconds_per_512_rows": inference_seconds,
            "model_size_mb": model_size_mb,
            "n_estimators": spec.n_estimators,
            "oof_score": spec.score,
        }
        metadata.append(member)
    return fitted, metadata


def _estimate_model_size_mb(estimator, model_name: str) -> float | None:
    try:
        with tempfile.TemporaryDirectory(prefix="smarttab_size_") as folder:
            path = Path(folder) / "model"
            if model_name == "catboost" and hasattr(estimator, "save_model"):
                path = path.with_suffix(".cbm")
                estimator.save_model(str(path))
            else:
                path = path.with_suffix(".joblib")
                joblib.dump(estimator, path, compress=0)
            return float(path.stat().st_size / (1024**2))
    except Exception:
        return None


def _candidate_metadata(
    specs: list[BaseModelSpec],
    retained: list[BaseModelSpec],
    scores: dict[str, float],
    diversity_matrix: dict[str, dict[str, float]],
) -> list[dict]:
    retained_aliases = {spec.alias for spec in retained}
    result: list[dict] = []
    for spec in specs:
        correlations = [
            value
            for alias, value in diversity_matrix.get(spec.alias, {}).items()
            if alias != spec.alias
        ]
        result.append(
            {
                **spec.diagnostics,
                "alias": spec.alias,
                "algorithm": spec.model_name,
                "optimize_metric": spec.optimize_metric,
                "random_seed": spec.random_seed,
                "hyperparameters": dict(spec.params),
                "n_estimators": spec.n_estimators,
                "feature_group": spec.feature_group,
                "feature_count": len(spec.feature_columns) if spec.feature_columns is not None else None,
                "feature_columns": list(spec.feature_columns) if spec.feature_columns is not None else None,
                "oof_score": scores.get(spec.alias),
                "max_prediction_correlation": max(correlations) if correlations else 0.0,
                "retained": spec.alias in retained_aliases,
            }
        )
    return result


def _fit_meta_candidates(
    X_meta: np.ndarray,
    y_meta: np.ndarray,
    task_type: TaskType,
    profile: DatasetProfile,
    resource_plan: ResourcePlan,
    random_state: int,
    requested: str,
    deadline: FitDeadline | None,
) -> dict[str, object]:
    if requested == "auto":
        preferred = "lightgbm" if profile.n_samples >= 100_000 else "catboost"
        alternative = "catboost" if preferred == "lightgbm" else "lightgbm"
        names = (preferred, alternative, "linear")
    else:
        names = (requested,)
    candidates: dict[str, object] = {}
    for name in names:
        if deadline is not None and deadline.enabled and (deadline.remaining() or 0.0) < 2:
            break
        model = _build_meta_model(name, task_type, profile, resource_plan, random_state)
        try:
            _fit_meta_model(model, X_meta, y_meta, name, task_type, deadline)
            candidates[name] = model
        except Exception:
            continue
    if not candidates:
        model = _build_meta_model("linear", task_type, profile, resource_plan, random_state)
        _fit_meta_model(model, X_meta, y_meta, "linear", task_type, deadline)
        candidates["linear"] = model
    return candidates


def _build_meta_model(
    name: str,
    task_type: TaskType,
    profile: DatasetProfile,
    resource_plan: ResourcePlan,
    random_state: int,
):
    if name == "linear":
        if task_type.is_classification:
            return LogisticRegression(
                max_iter=1000,
                class_weight="balanced" if profile.is_imbalanced else None,
                random_state=random_state,
            )
        return Ridge(alpha=1.0)
    if name == "lightgbm":
        params = {
            "num_leaves": 7,
            "max_depth": 3,
            "learning_rate": 0.05,
            "min_child_samples": 20,
            "reg_lambda": 1.0,
        }
        return build_estimator(
            "lightgbm",
            params,
            task_type,
            120,
            max(1, min(resource_plan.cpu_threads, 4)),
            False,
            random_state,
            resource_plan=resource_plan,
        )
    if name == "catboost":
        params = {"depth": 3, "learning_rate": 0.05, "l2_leaf_reg": 5.0}
        return build_estimator(
            "catboost",
            params,
            task_type,
            120,
            max(1, min(resource_plan.cpu_threads, 4)),
            False,
            random_state,
            resource_plan=resource_plan,
        )
    raise ConfigurationError(f"unknown meta model {name!r}")


def _fit_meta_model(
    model,
    X: np.ndarray,
    y: np.ndarray,
    name: str,
    task_type: TaskType,
    deadline: FitDeadline | None,
) -> None:
    if name == "linear":
        model.fit(X, y)
        return
    frame = pd.DataFrame(X, columns=[f"meta_{index}" for index in range(X.shape[1])])
    fit_estimator(model, name, frame, y, deadline=deadline)


def _meta_model_output(model, X: np.ndarray, task_type: TaskType) -> np.ndarray:
    module = model.__class__.__module__
    prepared = X
    if module.startswith("catboost") or module.startswith("lightgbm"):
        prepared = pd.DataFrame(X, columns=[f"meta_{index}" for index in range(X.shape[1])])
    if task_type.is_classification:
        return np.asarray(model.predict_proba(prepared))
    return np.asarray(model.predict(prepared)).ravel()


def _classification_meta_block(probabilities: np.ndarray, task_type: TaskType) -> np.ndarray:
    if task_type is TaskType.BINARY:
        return probabilities[:, 1:2]
    return probabilities


def _stack_oof_blocks(blocks: list[np.ndarray], task_type: TaskType) -> np.ndarray:
    if task_type.is_classification:
        return np.hstack([_classification_meta_block(block, task_type) for block in blocks])
    return np.column_stack([np.asarray(block).ravel() for block in blocks])


def _weighted_output(blocks: list[np.ndarray], weights: np.ndarray) -> np.ndarray:
    return np.average(np.stack(blocks, axis=0), axis=0, weights=weights)


def _optimize_voting_weights(
    blocks: list[np.ndarray],
    y: np.ndarray,
    task_type: TaskType,
    random_state: int,
) -> np.ndarray:
    if len(blocks) == 1:
        return np.ones(1, dtype=float)
    rng = np.random.default_rng(random_state)
    candidates = [np.full(len(blocks), 1.0 / len(blocks))]
    candidates.extend(np.eye(len(blocks)))
    candidates.extend(rng.dirichlet(np.ones(len(blocks)) * 0.75, size=WEIGHT_SEARCH_CANDIDATES))
    best_weights = candidates[0]
    best_vector = None
    for weights in candidates:
        output = _weighted_output(blocks, np.asarray(weights, dtype=float))
        threshold = _fit_binary_threshold(y, output, task_type)
        vector = _selection_vector(y, output, task_type, threshold)
        if best_vector is None or _compare_vectors(vector, best_vector, task_type) > 0:
            best_vector = vector
            best_weights = np.asarray(weights, dtype=float)
    best_weights = np.clip(best_weights, 0.0, None)
    return best_weights / best_weights.sum()


def _meta_train_select_split(
    y: np.ndarray,
    task_type: TaskType,
    random_state: int,
) -> tuple[np.ndarray, np.ndarray]:
    indices = np.arange(len(y))
    stratify = y if task_type in (TaskType.BINARY, TaskType.MULTICLASS) else None
    try:
        return train_test_split(
            indices,
            test_size=0.25,
            random_state=random_state,
            stratify=stratify,
        )
    except ValueError:
        return train_test_split(
            indices,
            test_size=0.25,
            random_state=random_state,
        )


def _score_from_output(
    y_true: np.ndarray,
    output: np.ndarray,
    task_type: TaskType,
    metric: str,
) -> float:
    if task_type.is_classification:
        predictions = np.argmax(output, axis=1)
        return compute_metric(metric, y_true, predictions, output)
    predictions = np.asarray(output).ravel()
    return compute_metric(metric, y_true, predictions)


def _thresholds_from_output(
    y_true: np.ndarray,
    output: np.ndarray,
    task_type: TaskType,
    enabled: bool,
    objective: str,
) -> tuple[float, float]:
    if not enabled:
        return DEFAULT_THRESHOLD, DEFAULT_REJECT_THRESHOLD
    if task_type is TaskType.BINARY:
        threshold, _ = optimize_threshold(y_true, np.asarray(output)[:, 1], objective)
        return float(threshold), DEFAULT_REJECT_THRESHOLD
    return DEFAULT_THRESHOLD, DEFAULT_REJECT_THRESHOLD


def _allocate_trials(total: int, n_models: int) -> list[int]:
    if n_models <= 0:
        return []
    base, remainder = divmod(total, n_models)
    return [base + int(index < remainder) for index in range(n_models)]


def _best_name(scores: dict[str, float], direction: str) -> str:
    return min(scores, key=scores.get) if direction == "minimize" else max(scores, key=scores.get)


def _relative_improvement(baseline: float, candidate: float, direction: str) -> float:
    denominator = max(abs(baseline), 1e-12)
    difference = baseline - candidate if direction == "minimize" else candidate - baseline
    return float(difference / denominator)
