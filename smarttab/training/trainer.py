"""Stage 6 — Training.

Owns the only code in SmartTab that knows how CatBoost's, LightGBM's, and
XGBoost's APIs differ (constructor kwargs, fit() signature, early stopping,
categorical feature handling, prediction shapes) across every supported task
type — binary/multiclass/regression, multilabel, multi-output regression,
and ranking. Every other stage talks to models only through the functions in
this module. XGBoost is only ever used as a base learner inside a
``voting``/``stacking`` ensemble (see ``training/ensemble.py``), never as a
standalone selectable model; ranking is only ever CatBoost/LightGBM.

Multilabel and multi-output regression are handled by wrapping a normal
single-output estimator (built exactly as for binary/regression) in
sklearn's ``MultiOutputClassifier``/``MultiOutputRegressor`` — one
independent copy of the model per label/output. This is simpler and more
robust than chasing each library's native multi-output loss functions, at
the cost of ``n_labels``x the training calls; ``n_jobs=1`` is used on the
wrapper (rather than sklearn's default parallel-across-labels) since each
individual CatBoost/LightGBM model already parallelizes internally via
``cpu_threads``, and stacking both levels of parallelism risks oversubscribing
the CPU.

Ranking is fundamentally different: CatBoost/LightGBM rankers need every row
tagged with a group (query) id, and rows belonging to the same group must be
*consecutive* in the training data. ``sort_by_group`` below is the one
utility that guarantees that; every ranking call site (trainer, optimizer,
api) is expected to sort through it before calling ``fit_estimator``.
"""

from __future__ import annotations

import threading

import numpy as np
import pandas as pd
import psutil
from sklearn.multioutput import MultiOutputClassifier, MultiOutputRegressor

from smarttab.analysis.dataset_analyzer import TaskType
from smarttab.exceptions import UnsupportedModelError
from smarttab.logging_utils import get_logger

logger = get_logger()

DEFAULT_EARLY_STOPPING_ROUNDS = 20
TRAINABLE_MODEL_NAMES = ("catboost", "lightgbm", "xgboost")
RANKING_MODEL_NAMES = ("catboost", "lightgbm")
DEFAULT_RANKING_LOSS = {"catboost": "YetiRank", "lightgbm": "lambdarank"}


def sort_by_group(X: pd.DataFrame, y: np.ndarray, group_ids: np.ndarray) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """Reorder (X, y, group_ids) so rows sharing a group are consecutive — required by both
    CatBoostRanker (``group_id=``) and LGBMRanker (``group=`` as consecutive run lengths)."""
    order = np.argsort(np.asarray(group_ids), kind="stable")
    X_sorted = X.iloc[order].reset_index(drop=True)
    y_sorted = np.asarray(y)[order]
    group_ids_sorted = np.asarray(group_ids)[order]
    return X_sorted, y_sorted, group_ids_sorted


def _group_sizes(sorted_group_ids: np.ndarray) -> np.ndarray:
    _, sizes = np.unique(sorted_group_ids, return_counts=True)
    return sizes


def build_estimator(
    model_name: str,
    params: dict,
    task_type: TaskType,
    n_estimators: int,
    cpu_threads: int,
    use_gpu: bool,
    random_state: int = 42,
):
    if task_type in (TaskType.MULTILABEL, TaskType.MULTIOUTPUT_REGRESSION):
        base_task = TaskType.BINARY if task_type is TaskType.MULTILABEL else TaskType.REGRESSION
        base = build_estimator(model_name, params, base_task, n_estimators, cpu_threads, use_gpu, random_state)
        wrapper_cls = MultiOutputClassifier if task_type is TaskType.MULTILABEL else MultiOutputRegressor
        return wrapper_cls(base, n_jobs=1)

    if task_type is TaskType.RANKING:
        if model_name not in RANKING_MODEL_NAMES:
            raise UnsupportedModelError(f"Ranking only supports {RANKING_MODEL_NAMES}, got {model_name!r}")
        if model_name == "catboost":
            from catboost import CatBoostRanker

            kwargs = dict(
                iterations=n_estimators, thread_count=cpu_threads, random_seed=random_state,
                verbose=False, allow_writing_files=False, loss_function=DEFAULT_RANKING_LOSS["catboost"],
            )
            if use_gpu:
                kwargs.update(task_type="GPU", devices="0")
            kwargs.update(params)
            return CatBoostRanker(**kwargs)

        from lightgbm import LGBMRanker

        kwargs = dict(
            n_estimators=n_estimators, n_jobs=cpu_threads, random_state=random_state,
            verbose=-1, objective=DEFAULT_RANKING_LOSS["lightgbm"],
        )
        if use_gpu:
            kwargs.update(device="gpu")
        kwargs.update(params)
        return LGBMRanker(**kwargs)

    if model_name == "catboost":
        from catboost import CatBoostClassifier, CatBoostRegressor

        cls = CatBoostRegressor if task_type is TaskType.REGRESSION else CatBoostClassifier
        kwargs = dict(
            iterations=n_estimators,
            thread_count=cpu_threads,
            random_seed=random_state,
            verbose=False,
            allow_writing_files=False,
        )
        if use_gpu:
            kwargs.update(task_type="GPU", devices="0")
        kwargs.update(params)
        return cls(**kwargs)

    if model_name == "lightgbm":
        from lightgbm import LGBMClassifier, LGBMRegressor

        cls = LGBMRegressor if task_type is TaskType.REGRESSION else LGBMClassifier
        kwargs = dict(n_estimators=n_estimators, n_jobs=cpu_threads, random_state=random_state, verbose=-1)
        if use_gpu:
            kwargs.update(device="gpu")
        kwargs.update(params)
        return cls(**kwargs)

    if model_name == "xgboost":
        from xgboost import XGBClassifier, XGBRegressor

        cls = XGBRegressor if task_type is TaskType.REGRESSION else XGBClassifier
        kwargs = dict(
            n_estimators=n_estimators,
            n_jobs=cpu_threads,
            random_state=random_state,
            verbosity=0,
            tree_method="hist",
            device="cuda" if use_gpu else "cpu",
        )
        kwargs.update(params)
        return cls(**kwargs)

    raise UnsupportedModelError(f"Unknown model_name {model_name!r}; expected one of {TRAINABLE_MODEL_NAMES}")


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
):
    has_eval_set = X_valid is not None and y_valid is not None

    if isinstance(estimator, (MultiOutputClassifier, MultiOutputRegressor)):
        # sklearn's multi-output wrappers don't support eval_set/early stopping uniformly
        # across arbitrary base estimators, so each label/output is fit to completion.
        estimator.fit(X_train, y_train)
        return estimator

    if group_ids_train is not None:
        if model_name == "catboost":
            from catboost import Pool

            train_pool = Pool(X_train, y_train, group_id=group_ids_train, cat_features=cat_features or None)
            eval_pool = (
                Pool(X_valid, y_valid, group_id=group_ids_valid, cat_features=cat_features or None)
                if has_eval_set else None
            )
            estimator.fit(
                train_pool, eval_set=eval_pool,
                early_stopping_rounds=early_stopping_rounds if has_eval_set else None,
                use_best_model=has_eval_set, verbose=False,
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
            estimator.fit(X_train, y_train, callbacks=callbacks or None, **fit_kwargs)
            return estimator

        raise UnsupportedModelError(f"Ranking only supports {RANKING_MODEL_NAMES}, got {model_name!r}")

    if model_name == "catboost":
        estimator.fit(
            X_train,
            y_train,
            cat_features=cat_features or None,
            eval_set=(X_valid, y_valid) if has_eval_set else None,
            early_stopping_rounds=early_stopping_rounds if has_eval_set else None,
            use_best_model=has_eval_set,
            verbose=False,
        )
        return estimator

    if model_name == "lightgbm":
        import lightgbm as lgb

        callbacks = []
        if has_eval_set and early_stopping_rounds:
            callbacks.append(lgb.early_stopping(early_stopping_rounds, verbose=False))
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

    raise UnsupportedModelError(f"Unknown model_name {model_name!r}; expected one of {TRAINABLE_MODEL_NAMES}")


def get_best_iteration(estimator, model_name: str) -> int | None:
    if isinstance(estimator, (MultiOutputClassifier, MultiOutputRegressor)):
        return None
    if model_name == "catboost":
        best = estimator.get_best_iteration()
        return int(best) if best is not None else None
    if model_name == "lightgbm":
        best = getattr(estimator, "best_iteration_", None)
        return int(best) if best else None
    if model_name == "xgboost":
        best = getattr(estimator, "best_iteration", None)
        return int(best) if best is not None else None
    raise UnsupportedModelError(f"Unknown model_name {model_name!r}")


def predict(estimator, model_name: str, X: pd.DataFrame) -> np.ndarray:
    preds = np.asarray(estimator.predict(X))
    if preds.ndim == 2 and preds.shape[1] == 1:
        return preds.ravel()
    return preds


def predict_proba(estimator, model_name: str, X: pd.DataFrame) -> np.ndarray:
    proba = estimator.predict_proba(X)
    if isinstance(proba, list):
        # MultiOutputClassifier: list of (n_samples, 2) arrays, one per label -> (n_samples, n_labels)
        return np.stack([p[:, 1] for p in proba], axis=1)
    return np.asarray(proba)


def verify_gpu_usable(model_name: str, task_type: TaskType, cpu_threads: int, random_state: int = 42) -> bool:
    """Smoke-test GPU training on a tiny synthetic sample; returns False (and logs) on any failure."""
    rng = np.random.default_rng(random_state)
    X = pd.DataFrame(rng.normal(size=(30, 3)), columns=["a", "b", "c"])
    group_ids = None

    if task_type is TaskType.REGRESSION:
        y = rng.normal(size=30)
    elif task_type is TaskType.MULTIOUTPUT_REGRESSION:
        y = pd.DataFrame(rng.normal(size=(30, 2)), columns=["y1", "y2"])
    elif task_type is TaskType.MULTILABEL:
        y = pd.DataFrame(rng.integers(0, 2, size=(30, 2)), columns=["y1", "y2"])
    elif task_type is TaskType.RANKING:
        y = rng.integers(0, 4, size=30)
        group_ids = np.sort(rng.integers(0, 6, size=30))
    else:
        y = rng.integers(0, 2, size=30)

    try:
        estimator = build_estimator(
            model_name, params={}, task_type=task_type, n_estimators=10,
            cpu_threads=cpu_threads, use_gpu=True, random_state=random_state,
        )
        fit_estimator(estimator, model_name, X, y, group_ids_train=group_ids)
        return True
    except Exception as exc:
        logger.warning("GPU training unavailable for %s (%s); falling back to CPU.", model_name, exc)
        return False


class PeakMemoryTracker:
    """Context manager that samples this process's RSS in a background thread to
    capture peak memory usage during a block (e.g. a training call)."""

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
                rss_mb = self._process.memory_info().rss / (1024 * 1024)
                self.peak_mb = max(self.peak_mb, rss_mb)
            except Exception:
                pass
            self._stop_event.wait(self.interval_seconds)

    @property
    def delta_mb(self) -> float:
        return max(0.0, self.peak_mb - self.start_mb)
