"""Estimator construction, fitting, inference, and resource tracking."""

from __future__ import annotations

import threading
import time
from functools import lru_cache
from dataclasses import dataclass

import numpy as np
import pandas as pd
import psutil
from threadpoolctl import threadpool_limits

from smarttab.analysis.dataset_analyzer import TaskType
from smarttab.deadline import FitDeadline
from smarttab.exceptions import ConfigurationError, UnsupportedModelError
from smarttab.hardware.resource_planner import ResourcePlan
from smarttab.logging_utils import get_logger

logger = get_logger()

DEFAULT_EARLY_STOPPING_ROUNDS = 30
TRAINABLE_MODEL_NAMES = ("catboost", "lightgbm", "xgboost")
RANKING_MODEL_NAMES = ("catboost", "lightgbm")
DEFAULT_RANKING_LOSS = {"catboost": "YetiRank", "lightgbm": "lambdarank"}


def sort_by_group(
    X: pd.DataFrame,
    y: np.ndarray,
    group_ids: np.ndarray,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    order = np.argsort(np.asarray(group_ids), kind="stable")
    return (
        X.iloc[order].reset_index(drop=True),
        np.asarray(y)[order],
        np.asarray(group_ids)[order],
    )


def _group_sizes(sorted_group_ids: np.ndarray) -> np.ndarray:
    if sorted_group_ids is None:
        raise ValueError("group ids are required")
    changes = np.r_[True, sorted_group_ids[1:] != sorted_group_ids[:-1]]
    starts = np.flatnonzero(changes)
    return np.diff(np.r_[starts, len(sorted_group_ids)])


@dataclass
class MultiOutputBoostingEstimator:
    model_name: str
    output_task: TaskType
    params: dict
    n_estimators: int
    cpu_threads: int
    use_gpu: bool
    random_state: int
    resource_plan: ResourcePlan | None = None

    def __post_init__(self) -> None:
        self.estimators_: list[object] = []

    def fit(
        self,
        X: pd.DataFrame,
        y: np.ndarray,
        *,
        cat_features: list[str] | None = None,
        deadline: FitDeadline | None = None,
    ) -> "MultiOutputBoostingEstimator":
        matrix = np.asarray(y)
        if matrix.ndim != 2:
            raise ValueError("multi-output target must be a 2D array")
        self.estimators_ = []
        for output_index in range(matrix.shape[1]):
            if deadline is not None:
                deadline.require(f"training output {output_index + 1}")
            estimator = build_estimator(
                self.model_name,
                self.params,
                self.output_task,
                self.n_estimators,
                self.cpu_threads,
                self.use_gpu,
                self.random_state + output_index,
                resource_plan=self.resource_plan,
            )
            fit_estimator(
                estimator,
                self.model_name,
                X,
                matrix[:, output_index],
                cat_features=cat_features,
                deadline=deadline,
            )
            self.estimators_.append(estimator)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return np.column_stack([np.asarray(estimator.predict(X)).ravel() for estimator in self.estimators_])

    def predict_proba(self, X: pd.DataFrame) -> list[np.ndarray]:
        return [np.asarray(estimator.predict_proba(X)) for estimator in self.estimators_]


def build_estimator(
    model_name: str,
    params: dict,
    task_type: TaskType,
    n_estimators: int,
    cpu_threads: int,
    use_gpu: bool,
    random_state: int = 42,
    resource_plan: ResourcePlan | None = None,
):
    if model_name not in TRAINABLE_MODEL_NAMES:
        raise UnsupportedModelError(f"unknown model {model_name!r}")
    if n_estimators < 1:
        raise ConfigurationError("n_estimators must be >= 1")

    params = _normalize_params(model_name, params)
    if task_type in (TaskType.MULTILABEL, TaskType.MULTIOUTPUT_REGRESSION):
        output_task = TaskType.BINARY if task_type is TaskType.MULTILABEL else TaskType.REGRESSION
        return MultiOutputBoostingEstimator(
            model_name,
            output_task,
            params,
            n_estimators,
            cpu_threads,
            use_gpu,
            random_state,
            resource_plan,
        )

    params = _apply_resource_parameters(model_name, params, use_gpu, resource_plan)

    try:
        if task_type is TaskType.RANKING:
            if model_name not in RANKING_MODEL_NAMES:
                raise UnsupportedModelError(f"ranking supports only {RANKING_MODEL_NAMES}")
            if model_name == "catboost":
                from catboost import CatBoostRanker

                kwargs = {
                    "iterations": n_estimators,
                    "thread_count": cpu_threads,
                    "random_seed": random_state,
                    "verbose": False,
                    "allow_writing_files": False,
                    "loss_function": DEFAULT_RANKING_LOSS["catboost"],
                    **params,
                }
                if use_gpu:
                    kwargs.update(task_type="GPU", devices="0")
                return CatBoostRanker(**kwargs)

            from lightgbm import LGBMRanker

            kwargs = {
                "n_estimators": n_estimators,
                "n_jobs": cpu_threads,
                "random_state": random_state,
                "verbose": -1,
                "objective": DEFAULT_RANKING_LOSS["lightgbm"],
                **params,
            }
            if use_gpu:
                kwargs["device"] = "gpu"
            return LGBMRanker(**kwargs)

        if model_name == "catboost":
            from catboost import CatBoostClassifier, CatBoostRegressor

            estimator_class = CatBoostRegressor if task_type is TaskType.REGRESSION else CatBoostClassifier
            kwargs = {
                "iterations": n_estimators,
                "thread_count": cpu_threads,
                "random_seed": random_state,
                "verbose": False,
                "allow_writing_files": False,
                **params,
            }
            if use_gpu:
                kwargs.update(task_type="GPU", devices="0")
            return estimator_class(**kwargs)

        if model_name == "lightgbm":
            from lightgbm import LGBMClassifier, LGBMRegressor

            estimator_class = LGBMRegressor if task_type is TaskType.REGRESSION else LGBMClassifier
            kwargs = {
                "n_estimators": n_estimators,
                "n_jobs": cpu_threads,
                "random_state": random_state,
                "verbose": -1,
                **params,
            }
            if use_gpu:
                kwargs["device"] = "gpu"
            return estimator_class(**kwargs)

        from xgboost import XGBClassifier, XGBRegressor

        estimator_class = XGBRegressor if task_type is TaskType.REGRESSION else XGBClassifier
        kwargs = {
            "n_estimators": n_estimators,
            "n_jobs": cpu_threads,
            "random_state": random_state,
            "verbosity": 0,
            "tree_method": "hist",
            "device": "cuda" if use_gpu else "cpu",
            **params,
        }
        return estimator_class(**kwargs)
    except TypeError as exc:
        raise ConfigurationError(f"invalid parameters for {model_name}: {exc}") from exc


@lru_cache(maxsize=1)
def _lightgbm_parameter_names() -> frozenset[str]:
    """Return the native and sklearn parameter aliases accepted by LightGBM."""
    from lightgbm import LGBMClassifier
    from lightgbm.basic import _ConfigAliases

    names = set(LGBMClassifier().get_params(deep=False))
    aliases = _ConfigAliases._get_all_param_aliases()
    names.update(aliases)
    for values in aliases.values():
        names.update(values)
    return frozenset(names)


def _normalize_params(model_name: str, params: dict) -> dict:
    normalized = dict(params)
    forbidden = {
        "catboost": {
            "iterations", "n_estimators", "thread_count", "random_seed", "task_type",
            "devices", "used_ram_limit", "gpu_ram_part",
        },
        "lightgbm": {
            "n_estimators", "n_jobs", "random_state", "device", "histogram_pool_size",
        },
        "xgboost": {"n_estimators", "n_jobs", "random_state", "device", "tree_method"},
    }[model_name]
    collisions = sorted(forbidden.intersection(normalized))
    if collisions:
        raise ConfigurationError(
            f"params cannot override SmartTab-managed parameters for {model_name}: {collisions}"
        )
    if model_name == "lightgbm":
        unknown = sorted(set(normalized) - _lightgbm_parameter_names())
        if unknown:
            raise ConfigurationError(
                f"unknown LightGBM params: {unknown}; check spelling and the installed LightGBM version"
            )
        if float(normalized.get("subsample", 1.0)) < 1.0:
            normalized.setdefault("subsample_freq", 1)
    return normalized


def _apply_resource_parameters(
    model_name: str,
    params: dict,
    use_gpu: bool,
    resource_plan: ResourcePlan | None,
) -> dict:
    if resource_plan is None:
        return params
    managed = dict(params)
    budget_mb = max(128, int(resource_plan.memory_budget_mb))
    if model_name == "catboost":
        managed["used_ram_limit"] = f"{budget_mb}mb"
        if use_gpu and resource_plan.gpu_ram_part > 0:
            managed["gpu_ram_part"] = float(resource_plan.gpu_ram_part)
    elif model_name == "lightgbm":
        # LightGBM exposes a histogram-cache budget rather than a total-process cap.
        # The pre-fit admission check below protects the raw matrix; this parameter
        # bounds the dominant learner-controlled cache.
        managed["histogram_pool_size"] = max(64, int(budget_mb * 0.5))
    elif model_name == "xgboost" and budget_mb < 2048:
        managed.setdefault("max_bin", 128)
    return managed


def _validate_matrix_memory_budget(
    X: pd.DataFrame,
    resource_plan: ResourcePlan | None,
    model_name: str,
) -> None:
    if resource_plan is None:
        return
    raw_mb = float(X.memory_usage(index=True, deep=True).sum()) / (1024 * 1024)
    # Native learners keep at least one converted copy plus working histograms.
    estimated_floor_mb = max(raw_mb * 2.0, raw_mb + 32.0)
    if estimated_floor_mb > resource_plan.memory_budget_mb:
        raise ConfigurationError(
            f"{model_name} training matrix requires at least about {estimated_floor_mb:.1f}MB, "
            f"exceeding ram_limit={resource_plan.memory_budget_mb:.1f}MB"
        )


class _CatBoostDeadlineCallback:
    def __init__(self, deadline: FitDeadline) -> None:
        self.deadline = deadline

    def after_iteration(self, info) -> bool:  # CatBoost callback protocol
        return not self.deadline.expired()


class _LightGBMDeadlineCallback:
    order = 5
    before_iteration = False

    def __init__(self, deadline: FitDeadline) -> None:
        self.deadline = deadline

    def __call__(self, env) -> None:
        if self.deadline.expired():
            import lightgbm as lgb

            scores = env.evaluation_result_list or []
            raise lgb.callback.EarlyStopException(max(env.iteration, 0), scores)


def fit_estimator(
    estimator,
    model_name: str,
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_valid: pd.DataFrame | None = None,
    y_valid: np.ndarray | None = None,
    cat_features: list[str] | None = None,
    early_stopping_rounds: int | None = None,
    group_ids_train: np.ndarray | None = None,
    group_ids_valid: np.ndarray | None = None,
    deadline: FitDeadline | None = None,
    resource_plan: ResourcePlan | None = None,
):
    _validate_matrix_memory_budget(X_train, resource_plan, model_name)
    if deadline is not None:
        deadline.require(f"training {model_name}")
    has_eval_set = X_valid is not None and y_valid is not None

    native_threads = _resolve_native_thread_limit(estimator, model_name, resource_plan)
    with threadpool_limits(limits=native_threads):
        return _fit_estimator_impl(
            estimator, model_name, X_train, y_train, X_valid, y_valid,
            cat_features, early_stopping_rounds, group_ids_train, group_ids_valid,
            deadline, resource_plan, has_eval_set,
        )


def _resolve_native_thread_limit(estimator, model_name: str, resource_plan: ResourcePlan | None) -> int:
    if resource_plan is not None:
        return max(1, int(resource_plan.cpu_threads))
    try:
        params = estimator.get_params()
    except Exception:
        params = {}
    if model_name == "catboost":
        value = params.get("thread_count")
    else:
        value = params.get("n_jobs")
    try:
        return max(1, int(value)) if value not in (None, -1) else 1
    except (TypeError, ValueError):
        return 1


def _fit_estimator_impl(
    estimator,
    model_name: str,
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_valid: pd.DataFrame | None,
    y_valid: np.ndarray | None,
    cat_features: list[str] | None,
    early_stopping_rounds: int | None,
    group_ids_train: np.ndarray | None,
    group_ids_valid: np.ndarray | None,
    deadline: FitDeadline | None,
    resource_plan: ResourcePlan | None,
    has_eval_set: bool,
):
    if isinstance(estimator, MultiOutputBoostingEstimator):
        return estimator.fit(X_train, y_train, cat_features=cat_features, deadline=deadline)

    if group_ids_train is not None:
        if model_name == "catboost":
            from catboost import Pool

            train_pool = Pool(X_train, y_train, group_id=group_ids_train, cat_features=cat_features or None)
            eval_pool = (
                Pool(X_valid, y_valid, group_id=group_ids_valid, cat_features=cat_features or None)
                if has_eval_set
                else None
            )
            callbacks = [_CatBoostDeadlineCallback(deadline)] if deadline and deadline.enabled else None
            estimator.fit(
                train_pool,
                eval_set=eval_pool,
                early_stopping_rounds=early_stopping_rounds if has_eval_set else None,
                use_best_model=has_eval_set,
                callbacks=callbacks,
                verbose=False,
            )
            return estimator

        if model_name == "lightgbm":
            import lightgbm as lgb

            fit_kwargs: dict = {"group": _group_sizes(group_ids_train)}
            callbacks = []
            if has_eval_set:
                fit_kwargs["eval_set"] = [(X_valid, y_valid)]
                fit_kwargs["eval_group"] = [_group_sizes(group_ids_valid)]
                if early_stopping_rounds:
                    callbacks.append(lgb.early_stopping(early_stopping_rounds, verbose=False))
            if deadline and deadline.enabled:
                callbacks.append(_LightGBMDeadlineCallback(deadline))
            estimator.fit(X_train, y_train, callbacks=callbacks or None, **fit_kwargs)
            return estimator
        raise UnsupportedModelError(f"ranking supports only {RANKING_MODEL_NAMES}")

    if model_name == "catboost":
        callbacks = [_CatBoostDeadlineCallback(deadline)] if deadline and deadline.enabled else None
        estimator.fit(
            X_train,
            y_train,
            cat_features=cat_features or None,
            eval_set=(X_valid, y_valid) if has_eval_set else None,
            early_stopping_rounds=early_stopping_rounds if has_eval_set else None,
            use_best_model=has_eval_set,
            callbacks=callbacks,
            verbose=False,
        )
        return estimator

    if model_name == "lightgbm":
        import lightgbm as lgb

        callbacks = []
        if has_eval_set and early_stopping_rounds:
            callbacks.append(lgb.early_stopping(early_stopping_rounds, verbose=False))
        if deadline and deadline.enabled:
            callbacks.append(_LightGBMDeadlineCallback(deadline))
        estimator.fit(
            X_train,
            y_train,
            categorical_feature=cat_features or "auto",
            eval_set=[(X_valid, y_valid)] if has_eval_set else None,
            callbacks=callbacks or None,
        )
        return estimator

    if model_name == "xgboost":
        if has_eval_set and early_stopping_rounds:
            estimator.set_params(early_stopping_rounds=early_stopping_rounds)
        estimator.fit(
            X_train,
            y_train,
            eval_set=[(X_valid, y_valid)] if has_eval_set else None,
            verbose=False,
        )
        return estimator
    raise UnsupportedModelError(f"unknown model {model_name!r}")


def get_best_iteration(estimator, model_name: str) -> int | None:
    if isinstance(estimator, MultiOutputBoostingEstimator):
        values = [get_best_iteration(item, model_name) for item in estimator.estimators_]
        valid = [value for value in values if value]
        return int(np.median(valid)) if valid else None
    if model_name == "catboost":
        best = estimator.get_best_iteration()
        return int(best) + 1 if best is not None and int(best) >= 0 else None
    if model_name == "lightgbm":
        best = getattr(estimator, "best_iteration_", None)
        return int(best) if best else None
    if model_name == "xgboost":
        best = getattr(estimator, "best_iteration", None)
        return int(best) + 1 if best is not None else None
    raise UnsupportedModelError(f"unknown model {model_name!r}")


def predict(estimator, model_name: str, X: pd.DataFrame) -> np.ndarray:
    predictions = np.asarray(estimator.predict(X))
    if predictions.ndim == 2 and predictions.shape[1] == 1:
        return predictions.ravel()
    return predictions


def predict_proba(estimator, model_name: str, X: pd.DataFrame) -> np.ndarray:
    probabilities = estimator.predict_proba(X)
    if isinstance(probabilities, list):
        return np.stack([np.asarray(item)[:, 1] for item in probabilities], axis=1)
    return np.asarray(probabilities)


def verify_gpu_usable(
    model_name: str,
    task_type: TaskType,
    cpu_threads: int,
    random_state: int = 42,
) -> bool:
    rng = np.random.default_rng(random_state)
    X = pd.DataFrame(rng.normal(size=(40, 3)), columns=["a", "b", "c"])
    group_ids = None
    if task_type is TaskType.REGRESSION:
        y = rng.normal(size=40)
    elif task_type is TaskType.MULTIOUTPUT_REGRESSION:
        y = rng.normal(size=(40, 2))
    elif task_type is TaskType.MULTILABEL:
        y = rng.integers(0, 2, size=(40, 2))
    elif task_type is TaskType.RANKING:
        y = rng.integers(0, 4, size=40)
        group_ids = np.repeat(np.arange(8), 5)
    else:
        y = rng.integers(0, 2, size=40)
    try:
        estimator = build_estimator(
            model_name,
            {},
            task_type,
            10,
            cpu_threads,
            True,
            random_state,
        )
        fit_estimator(estimator, model_name, X, y, group_ids_train=group_ids)
        return True
    except Exception as exc:
        logger.warning("GPU training unavailable for %s (%s); using CPU", model_name, exc)
        return False


class PeakMemoryTracker:
    """Sample process RSS while a training block runs."""

    def __init__(self, interval_seconds: float = 0.05) -> None:
        self.interval_seconds = interval_seconds
        self._process = psutil.Process()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self.start_mb = 0.0
        self.peak_mb = 0.0

    def __enter__(self) -> "PeakMemoryTracker":
        self.start_mb = self._process.memory_info().rss / (1024 * 1024)
        self.peak_mb = self.start_mb
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._poll, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc_info) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)

    def _poll(self) -> None:
        while not self._stop_event.is_set():
            try:
                self.peak_mb = max(
                    self.peak_mb,
                    self._process.memory_info().rss / (1024 * 1024),
                )
            except (psutil.Error, OSError):
                pass
            self._stop_event.wait(self.interval_seconds)

    @property
    def delta_mb(self) -> float:
        return max(0.0, self.peak_mb - self.start_mb)
