"""Decision-threshold optimization.

CatBoost/LightGBM/XGBoost (and our ensembles) all output probabilities;
converting those into hard predictions with a fixed 0.5 cutoff is arbitrary
and, on imbalanced data especially, often far from the threshold that
actually maximizes the metric the user actually cares about. This module
sweeps a grid of thresholds on held-out probabilities and picks the one that
maximizes a configurable ``objective`` metric.

Three task types get threshold support, each with different semantics:

- **Binary**: the classic case — one cutoff on P(positive) turns probability
  into a 0/1 label. ``optimize_threshold`` / ``build_threshold_ladder``.
- **Multi-class**: there's no single "positive class" to threshold, so this
  is a *reject option* — a cutoff on the top predicted class's probability.
  Below the cutoff, a row is treated as "too uncertain to call" (scored as
  wrong for objective purposes) rather than forced through argmax.
  ``threshold=0.0`` means "never reject", i.e. identical to plain argmax —
  this is the default, so nothing changes unless a caller explicitly opts
  in. ``optimize_multiclass_reject_threshold`` / `build_multiclass_threshold_ladder``.
- **Multi-label**: each label is independently binary, so this is just the
  binary sweep run once per label column. ``optimize_per_label_thresholds`` /
  ``build_per_label_threshold_ladders``.

Regression and ranking have no threshold concept at all and are not touched
by this module.

``build_threshold_ladder`` (and its multiclass/multilabel counterparts) is
the second piece: instead of one threshold, it picks several (``n_models``)
spanning the recall/precision trade-off curve (coverage/accuracy for
multiclass), from most lenient (catches the most positives, more false
alarms) to strictest (fewer false alarms, may miss borderline cases). A
prediction that clears even the strictest threshold is a high-confidence
call; one that only clears the most lenient threshold is a low-confidence,
borderline call — this is what powers ``multi_threshold_ensemble``'s
per-prediction confidence score.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    matthews_corrcoef,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

from smarttab.analysis.dataset_analyzer import TaskType
from smarttab.hardware.resource_planner import ResourcePlan

DEFAULT_THRESHOLD = 0.5
DEFAULT_REJECT_THRESHOLD = 0.0  # multiclass: 0.0 = never reject = identical to plain argmax
DEFAULT_OBJECTIVE = "mcc"
_THRESHOLD_GRID = np.linspace(0.01, 0.99, 99)
_MULTICLASS_THRESHOLD_GRID = np.concatenate(([0.0], _THRESHOLD_GRID))
_PROBE_HOLDOUT_FRACTION = 0.15
_REJECT_SENTINEL = -1  # target-encoded classes are always 0..K-1, so -1 never collides

# A ladder level whose accepted/predicted-positive rate falls below this is not a useful
# "confidence tier" — it's effectively "predicts almost nobody" (or, unbounded, "predicts
# literally everyone") and would only clutter multi_threshold_ensemble's output. See
# _sane_ladder_candidates.
_LADDER_MIN_COVERAGE = 0.01

# Metrics a threshold sweep can actually optimize (they change with the cutoff).
SWEEPABLE_OBJECTIVES = ("f1", "precision", "recall", "accuracy", "balanced_accuracy", "mcc")
# roc_auc is threshold-*invariant* (it's computed from the raw scores, not a hard cutoff) — it's
# still a valid `objective=` choice, it just means "don't bother tuning a threshold at all".
VALID_OBJECTIVES = SWEEPABLE_OBJECTIVES + ("roc_auc",)


def _score_at_threshold(objective: str, y_true: np.ndarray, y_proba_positive: np.ndarray, threshold: float) -> float:
    preds = apply_threshold(y_proba_positive, threshold)
    if objective == "f1":
        return f1_score(y_true, preds, zero_division=0)
    if objective == "precision":
        return precision_score(y_true, preds, zero_division=0)
    if objective == "recall":
        return recall_score(y_true, preds, zero_division=0)
    if objective == "accuracy":
        return accuracy_score(y_true, preds)
    if objective == "balanced_accuracy":
        return balanced_accuracy_score(y_true, preds)
    if objective == "mcc":
        return matthews_corrcoef(y_true, preds)
    raise ValueError(f"objective must be one of {SWEEPABLE_OBJECTIVES}, got {objective!r}")


def optimize_threshold(
    y_true: np.ndarray, y_proba_positive: np.ndarray, objective: str = DEFAULT_OBJECTIVE
) -> tuple[float, float]:
    """Return (best_threshold, best_score) found by sweeping ``_THRESHOLD_GRID`` to maximize
    ``objective``.

    ``objective="roc_auc"`` is a deliberate no-op for the sweep itself — ROC AUC doesn't depend
    on the cutoff, so there's nothing to tune; the default threshold is kept and the (constant)
    ROC AUC is returned as the score, purely informationally.

    Falls back to ``DEFAULT_THRESHOLD`` if ``y_true`` has fewer than two classes present (every
    objective is meaningless in that case).
    """
    if objective not in VALID_OBJECTIVES:
        raise ValueError(f"objective must be one of {VALID_OBJECTIVES}, got {objective!r}")
    if len(np.unique(y_true)) < 2:
        return DEFAULT_THRESHOLD, 0.0

    if objective == "roc_auc":
        return DEFAULT_THRESHOLD, float(roc_auc_score(y_true, y_proba_positive))

    best_threshold, best_score = DEFAULT_THRESHOLD, -1.0
    for threshold in _THRESHOLD_GRID:
        score = _score_at_threshold(objective, y_true, y_proba_positive, threshold)
        if score > best_score:
            best_score, best_threshold = score, float(threshold)

    return best_threshold, best_score


def apply_threshold(y_proba_positive: np.ndarray, threshold: float) -> np.ndarray:
    return (y_proba_positive >= threshold).astype(int)


def _sane_ladder_candidates(
    thresholds: np.ndarray, recalls: np.ndarray, precisions: np.ndarray, n_positives: int, n: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Restrict ``precision_recall_curve``'s raw candidate thresholds to a sane, usable range
    before building a ladder from them.

    Two problems otherwise show up on a highly separable and/or heavily imbalanced dataset:
    the curve's raw thresholds can sit at whatever the most extreme observed probabilities
    happen to be (e.g. ``1e-5`` or ``0.9999``), well outside the ``[0.01, 0.99]`` band
    ``optimize_threshold`` itself sweeps — so a ladder built from the unrestricted curve can
    include a "threshold ≈ 0" level (predicts literally everyone) that no one would ever
    actually use. And even within bounds, the strictest few candidates can predict positive
    for a vanishingly small handful of rows — not a meaningfully "high confidence" tier, just
    noise. Both are filtered out here: candidates must fall within ``_THRESHOLD_GRID`` and
    predict positive for at least ``_LADDER_MIN_COVERAGE`` of rows (recomputed from
    precision/recall without a second pass over the data: predicted positive count is
    ``recall * n_positives / precision``).
    """
    in_bounds = (thresholds >= _THRESHOLD_GRID[0]) & (thresholds <= _THRESHOLD_GRID[-1])
    with np.errstate(divide="ignore", invalid="ignore"):
        predicted_positive_rate = np.where(precisions > 0, recalls * n_positives / precisions, 0.0) / n
    has_min_coverage = predicted_positive_rate >= _LADDER_MIN_COVERAGE

    both = in_bounds & has_min_coverage
    if both.sum() >= 2:
        return thresholds[both], recalls[both]
    if in_bounds.sum() >= 2:
        return thresholds[in_bounds], recalls[in_bounds]
    return thresholds, recalls


def build_threshold_ladder(y_true: np.ndarray, y_proba_positive: np.ndarray, n_models: int = 4) -> list[dict]:
    """Pick ``n_models`` thresholds spanning the recall/precision trade-off curve, sorted
    ascending (most lenient/highest-recall first, strictest/highest-precision last). Each entry
    reports the threshold and the precision/recall/f1 it actually achieves on ``y_true``.

    Falls back to ``n_models`` copies of the default threshold if there's no usable
    precision-recall curve (e.g. a single class present).
    """
    if n_models < 2:
        raise ValueError(f"n_models must be >= 2, got {n_models}")
    if len(np.unique(y_true)) < 2:
        return [_ladder_point(DEFAULT_THRESHOLD, y_true, y_proba_positive) for _ in range(n_models)]

    precisions, recalls, thresholds = precision_recall_curve(y_true, y_proba_positive)
    # precision_recall_curve returns one more precision/recall point than thresholds (the
    # threshold=+inf endpoint); drop it so the arrays line up.
    precisions, recalls = precisions[:-1], recalls[:-1]
    if len(thresholds) == 0:
        return [_ladder_point(DEFAULT_THRESHOLD, y_true, y_proba_positive) for _ in range(n_models)]

    order = np.argsort(thresholds)
    thresholds, recalls, precisions = thresholds[order], recalls[order], precisions[order]
    thresholds, recalls = _sane_ladder_candidates(
        thresholds, recalls, precisions, n_positives=int(np.sum(y_true == 1)), n=len(y_true),
    )

    # Evenly spaced *recall* targets (not threshold values) give operating points that are
    # meaningfully different from each other, from most lenient (max recall) to strictest
    # (min recall) — spacing evenly in raw threshold value tends to bunch most points in the
    # low-recall tail since recall drops off steeply.
    target_recalls = np.linspace(recalls.max(), recalls.min(), n_models)
    chosen_thresholds = sorted({float(thresholds[np.argmin(np.abs(recalls - target))]) for target in target_recalls})

    # De-duplication can leave fewer than n_models points on coarse/small datasets; pad with
    # linearly interpolated values between neighbors so the caller always gets n_models back.
    while len(chosen_thresholds) < n_models:
        gaps = np.diff(chosen_thresholds)
        if len(gaps) == 0:
            chosen_thresholds.append(min(1.0, chosen_thresholds[-1] + 0.01))
            continue
        widest = int(np.argmax(gaps))
        midpoint = (chosen_thresholds[widest] + chosen_thresholds[widest + 1]) / 2
        chosen_thresholds.insert(widest + 1, midpoint)

    chosen_thresholds = sorted(chosen_thresholds)[:n_models]
    return [_ladder_point(t, y_true, y_proba_positive) for t in chosen_thresholds]


def _ladder_point(threshold: float, y_true: np.ndarray, y_proba_positive: np.ndarray) -> dict:
    preds = apply_threshold(y_proba_positive, threshold)
    return {
        "threshold": float(threshold),
        "precision": float(precision_score(y_true, preds, zero_division=0)),
        "recall": float(recall_score(y_true, preds, zero_division=0)),
        "f1": float(f1_score(y_true, preds, zero_division=0)),
        "accuracy": float(accuracy_score(y_true, preds)),
        "predicted_positive_rate": float(np.mean(preds)),
    }


def _multiclass_score_with_rejection(
    objective: str, y_true: np.ndarray, predicted_class: np.ndarray, max_proba: np.ndarray,
    threshold: float, class_labels: list,
) -> float:
    """Score ``objective`` treating any row with ``max_proba < threshold`` as rejected — a
    rejected row is never counted as correct, so raising the threshold trades coverage for
    accuracy-on-accepted instead of trivially "winning" by rejecting everything."""
    accepted = max_proba >= threshold
    pred_or_reject = np.where(accepted, predicted_class, _REJECT_SENTINEL)
    if objective == "accuracy":
        return accuracy_score(y_true, pred_or_reject)
    if objective == "balanced_accuracy":
        return balanced_accuracy_score(y_true, pred_or_reject)
    if objective == "mcc":
        return matthews_corrcoef(y_true, pred_or_reject)
    if objective in ("f1", "precision", "recall"):
        fn = {"f1": f1_score, "precision": precision_score, "recall": recall_score}[objective]
        with warnings.catch_warnings():
            # pred_or_reject deliberately contains a sentinel class (rejected rows) that's never
            # in y_true; labels=class_labels already restricts the score to the real classes, so
            # sklearn's "y_pred contains classes not in y_true" warning here is expected noise.
            warnings.filterwarnings("ignore", message=".*y_pred contains classes not in y_true.*")
            return fn(y_true, pred_or_reject, labels=class_labels, average="macro", zero_division=0)
    raise ValueError(f"objective must be one of {SWEEPABLE_OBJECTIVES}, got {objective!r}")


def optimize_multiclass_reject_threshold(
    y_true: np.ndarray, y_proba: np.ndarray, objective: str = DEFAULT_OBJECTIVE
) -> tuple[float, float]:
    """Multi-class counterpart of :func:`optimize_threshold`: sweeps a *reject* cutoff on the
    top predicted-class probability rather than a decision cutoff on a single positive class.
    Returns ``(best_threshold, best_score)``; ``threshold=0.0`` (returned whenever nothing beats
    "accept everyone", or when there are fewer than 2 classes present) means plain argmax with
    no rejection at all — the same behavior as before this feature existed.
    """
    if objective not in VALID_OBJECTIVES:
        raise ValueError(f"objective must be one of {VALID_OBJECTIVES}, got {objective!r}")
    class_labels = sorted(np.unique(y_true).tolist())
    if len(class_labels) < 2:
        return DEFAULT_REJECT_THRESHOLD, 0.0

    if objective == "roc_auc":
        try:
            score = float(roc_auc_score(y_true, y_proba, multi_class="ovr", average="macro"))
        except ValueError:
            score = 0.0
        return DEFAULT_REJECT_THRESHOLD, score

    predicted_class = np.argmax(y_proba, axis=1)
    max_proba = np.max(y_proba, axis=1)

    best_threshold, best_score = DEFAULT_REJECT_THRESHOLD, -1.0
    for threshold in _MULTICLASS_THRESHOLD_GRID:
        score = _multiclass_score_with_rejection(objective, y_true, predicted_class, max_proba, threshold, class_labels)
        if score > best_score:
            best_score, best_threshold = score, float(threshold)

    return best_threshold, best_score


def _multiclass_ladder_point(threshold: float, y_true: np.ndarray, predicted_class: np.ndarray, max_proba: np.ndarray) -> dict:
    accepted = max_proba >= threshold
    n_accepted = int(accepted.sum())
    pred_or_reject = np.where(accepted, predicted_class, _REJECT_SENTINEL)
    return {
        "threshold": float(threshold),
        "coverage": float(np.mean(accepted)),
        "accuracy_on_accepted": float(accuracy_score(y_true[accepted], predicted_class[accepted])) if n_accepted else 0.0,
        "accuracy_overall": float(accuracy_score(y_true, pred_or_reject)),
        "n_accepted": n_accepted,
    }


def build_multiclass_threshold_ladder(y_true: np.ndarray, y_proba: np.ndarray, n_models: int = 4) -> list[dict]:
    """Pick ``n_models`` reject-thresholds spanning the coverage/accuracy trade-off, sorted
    ascending (most lenient/highest-coverage first, strictest/highest-accuracy-on-accepted
    last). Mirrors :func:`build_threshold_ladder` but for the multiclass reject-option case —
    see module docstring."""
    if n_models < 2:
        raise ValueError(f"n_models must be >= 2, got {n_models}")

    predicted_class = np.argmax(y_proba, axis=1)
    max_proba = np.max(y_proba, axis=1)

    if len(np.unique(y_true)) < 2:
        return [_multiclass_ladder_point(DEFAULT_REJECT_THRESHOLD, y_true, predicted_class, max_proba) for _ in range(n_models)]

    coverages = np.array([float(np.mean(max_proba >= t)) for t in _MULTICLASS_THRESHOLD_GRID])
    # Same reasoning as _sane_ladder_candidates: a reject threshold that accepts almost no rows
    # isn't a meaningfully "strict" confidence tier, just noise — drop candidates below the floor
    # when enough remain above it.
    has_min_coverage = coverages >= _LADDER_MIN_COVERAGE
    if has_min_coverage.sum() >= 2:
        grid, coverages = _MULTICLASS_THRESHOLD_GRID[has_min_coverage], coverages[has_min_coverage]
    else:
        grid, coverages = _MULTICLASS_THRESHOLD_GRID, coverages
    target_coverages = np.linspace(coverages.max(), coverages.min(), n_models)
    chosen_thresholds = sorted({float(grid[np.argmin(np.abs(coverages - target))]) for target in target_coverages})

    while len(chosen_thresholds) < n_models:
        gaps = np.diff(chosen_thresholds)
        if len(gaps) == 0:
            chosen_thresholds.append(min(1.0, chosen_thresholds[-1] + 0.01))
            continue
        widest = int(np.argmax(gaps))
        midpoint = (chosen_thresholds[widest] + chosen_thresholds[widest + 1]) / 2
        chosen_thresholds.insert(widest + 1, midpoint)

    chosen_thresholds = sorted(chosen_thresholds)[:n_models]
    return [_multiclass_ladder_point(t, y_true, predicted_class, max_proba) for t in chosen_thresholds]


def optimize_per_label_thresholds(y_true: np.ndarray, y_proba: np.ndarray, objective: str = DEFAULT_OBJECTIVE) -> list[float]:
    """Multi-label counterpart of :func:`optimize_threshold`: each label is independently
    binary (0/1), so this just runs the ordinary binary sweep once per label column.
    ``y_true``/``y_proba`` are (n_samples, n_labels)."""
    n_labels = y_true.shape[1]
    return [optimize_threshold(y_true[:, i], y_proba[:, i], objective=objective)[0] for i in range(n_labels)]


def build_per_label_threshold_ladders(y_true: np.ndarray, y_proba: np.ndarray, n_models: int = 4) -> list[list[dict]]:
    """Multi-label counterpart of :func:`build_threshold_ladder`: one ladder per label column."""
    n_labels = y_true.shape[1]
    return [build_threshold_ladder(y_true[:, i], y_proba[:, i], n_models) for i in range(n_labels)]


def probe_and_optimize_threshold(
    model_name: str,
    params: dict,
    n_estimators: int,
    task_type: TaskType,
    resource_plan: ResourcePlan,
    cat_features: list[str],
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    random_state: int = 42,
    objective: str = DEFAULT_OBJECTIVE,
    build_ladder: bool = False,
    n_ladder_models: int = 4,
) -> tuple[float, list[dict] | None]:
    """Threshold tuning (and, optionally, threshold-ladder building) for a single
    (non-ensemble) final model. Handles ``task_type`` BINARY (decision threshold on the
    positive class) or MULTICLASS (reject threshold on the top predicted class — see module
    docstring); multi-label goes through :func:`probe_and_optimize_multilabel_thresholds`
    instead, since its return shape is a list of thresholds, not one.

    The final model is trained on *all* of X_train for best quality, so there's no held-out
    slice left to tune a threshold on without leaking. This trains a cheap disposable "probe"
    copy (same hyperparameters, same n_estimators, no further search) on 85% of X_train and
    tunes the threshold (and ladder, if requested) on the remaining 15% — the real production
    model is untouched by this. Returns ``(threshold, ladder_or_None)``.
    """
    from smarttab.training.trainer import build_estimator, fit_estimator, predict_proba

    is_multiclass = task_type is TaskType.MULTICLASS
    fallback_threshold = DEFAULT_REJECT_THRESHOLD if is_multiclass else DEFAULT_THRESHOLD

    try:
        X_probe_train, X_probe_val, y_probe_train, y_probe_val = train_test_split(
            X_train, y_train, test_size=_PROBE_HOLDOUT_FRACTION, random_state=random_state, stratify=y_train,
        )
    except ValueError:
        if not build_ladder:
            return fallback_threshold, None
        if is_multiclass:
            zeros = np.zeros_like(y_train)
            fallback_ladder = [_multiclass_ladder_point(fallback_threshold, y_train, zeros, np.zeros_like(y_train, dtype=float))] * n_ladder_models
        else:
            fallback_ladder = [_ladder_point(fallback_threshold, y_train, np.zeros_like(y_train, dtype=float))] * n_ladder_models
        return fallback_threshold, fallback_ladder

    probe = build_estimator(
        model_name, params, task_type, n_estimators=n_estimators,
        cpu_threads=resource_plan.cpu_threads, use_gpu=resource_plan.use_gpu, random_state=random_state,
    )
    fit_estimator(probe, model_name, X_probe_train, y_probe_train, cat_features=cat_features)
    proba_full = predict_proba(probe, model_name, X_probe_val)

    if is_multiclass:
        threshold, _ = optimize_multiclass_reject_threshold(y_probe_val, proba_full, objective=objective)
        ladder = build_multiclass_threshold_ladder(y_probe_val, proba_full, n_ladder_models) if build_ladder else None
        return threshold, ladder

    proba = proba_full[:, 1]
    threshold, _ = optimize_threshold(y_probe_val, proba, objective=objective)
    ladder = build_threshold_ladder(y_probe_val, proba, n_ladder_models) if build_ladder else None
    return threshold, ladder


def probe_and_optimize_multilabel_thresholds(
    model_name: str,
    params: dict,
    n_estimators: int,
    task_type: TaskType,
    resource_plan: ResourcePlan,
    cat_features: list[str],
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    random_state: int = 42,
    objective: str = DEFAULT_OBJECTIVE,
    build_ladder: bool = False,
    n_ladder_models: int = 4,
) -> tuple[list[float], list[list[dict]] | None]:
    """Multi-label counterpart of :func:`probe_and_optimize_threshold`: same disposable-probe
    methodology, but returns one threshold (and, optionally, one ladder) per label column
    instead of a single scalar. ``y_train`` is (n_samples, n_labels).
    """
    from smarttab.training.trainer import build_estimator, fit_estimator, predict_proba

    n_labels = y_train.shape[1]

    try:
        X_probe_train, X_probe_val, y_probe_train, y_probe_val = train_test_split(
            X_train, y_train, test_size=_PROBE_HOLDOUT_FRACTION, random_state=random_state,
        )
    except ValueError:
        thresholds = [DEFAULT_THRESHOLD] * n_labels
        if not build_ladder:
            return thresholds, None
        zeros = np.zeros(len(y_train), dtype=float)
        ladders = [[_ladder_point(DEFAULT_THRESHOLD, y_train[:, i], zeros) for _ in range(n_ladder_models)] for i in range(n_labels)]
        return thresholds, ladders

    probe = build_estimator(
        model_name, params, task_type, n_estimators=n_estimators,
        cpu_threads=resource_plan.cpu_threads, use_gpu=resource_plan.use_gpu, random_state=random_state,
    )
    fit_estimator(probe, model_name, X_probe_train, y_probe_train, cat_features=cat_features)
    proba = predict_proba(probe, model_name, X_probe_val)

    thresholds = optimize_per_label_thresholds(y_probe_val, proba, objective=objective)
    ladders = build_per_label_threshold_ladders(y_probe_val, proba, n_ladder_models) if build_ladder else None
    return thresholds, ladders
