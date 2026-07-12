# How SmartTab Works

This document explains *what happens inside `fit()` and why* — the algorithms,
heuristics, and design decisions. If you want to know *how to call* a
parameter, see [documents.md](documents.md) instead; this is the internals
tour.

## Philosophy

Every stage below has an `"auto"` mode that makes a real decision based on
your dataset and hardware — not a fixed default disguised as one. Every one
of those decisions can also be pinned to an explicit value. The goal is that
`smarttab.fit(df, target="y")` alone is a reasonable, defensible choice for
most tabular problems, and every knob is there for the 20% of cases where it
isn't.

## The Pipeline

`fit()` runs nine stages in order. Each stage's output feeds the next; none
of them re-derive facts another stage already established.

```
1. Dataset Analyzer
2. Smart Cleaning
3. Hardware Analyzer
4. Model Selection (or Ensemble Decision Engine)
5. Hyperparameter Optimization
6. Training (+ Threshold Optimization)
7. Evaluation
8. Explainability
9. Report
```

### 1. Dataset Analyzer

Produces one read-only `DatasetProfile` that every later stage consumes.
This is where **task type** is detected — purely from the shape of `target`
(and whether `group_id` was given):

| Signal | Task type |
|---|---|
| One target column, ≤ 2 unique values | Binary classification |
| One target column, 3–20 unique discrete values, or string/categorical | Multi-class |
| One target column, continuous | Regression |
| Multiple target columns, all 0/1 or bool | Multi-label |
| Multiple target columns, all continuous | Multi-output regression |
| `group_id` given | Ranking (single relevance/label column) |

Alongside task type, this stage also computes: missing-value rates per
column, duplicate rows and duplicate column groups, constant columns,
high-correlation pairs, per-column cardinality, outlier counts (IQR method,
detection only), ID-like columns (all-unique values), datetime and free-text
columns, potential target leakage (near-perfect correlation with the
target, or a suspiciously matching column name), and class imbalance ratio.

### 2. Smart Cleaning

Acts on the profile from stage 1, fit only on the training split (never on
test data, and never in a way that would leak information from test into
train):

- Drops constant columns, ID-like columns, redundant duplicate columns, and
  columns that correlate ≥ 0.999 with the target (near-certain leakage).
- Datetime columns become `_year`, `_month`, `_day`, `_dayofweek` (and
  `_hour` if there's a real time component); the original column is dropped.
- Free-text columns become `_len` (character count) and `_word_count`; the
  raw text is dropped — SmartTab does not do NLP.
- Missing values: median for numeric columns, a reserved `__missing__` token
  for categorical columns.
- Categorical encoding is ordinal-integer, not one-hot — both CatBoost and
  LightGBM consume integer-encoded categoricals natively and efficiently;
  one-hot would just inflate dimensionality for no benefit. Unseen
  categories at prediction time map to a reserved code instead of crashing.
- Scaling is a no-op by default — tree-based models don't need it.
- Feature selection only removes near-duplicate numeric columns
  (correlation ≥ 0.999); tree models aren't hurt by ordinary correlation,
  so nothing more aggressive is applied by default.

### 3. Hardware Analyzer

Profiles physical/logical CPU cores, CPU brand and AVX flags, total and
available RAM, and GPU presence/name/free VRAM (via `nvidia-ml-py3`, with a
clean no-GPU fallback). From that it decides:

- **Thread count**: physical cores − 1 (always leaves one core for the
  system).
- **GPU usage**: only when a GPU is present with ≥ 1 GB free VRAM *and* the
  dataset is large enough (≥ 5,000 rows) that GPU overhead pays off. Before
  committing, it runs one tiny fit as a smoke test — if the installed
  library build can't actually use the GPU (e.g. a CPU-only LightGBM
  wheel), it silently falls back to CPU instead of failing the whole run.
- **RAM budget**: caps usage so at least 15% of RAM stays free, regardless
  of what you ask for.

### 4. Model Selection (or Ensemble Decision Engine)

Without `ensemble=`, the choice is a simple rule:

```
n_samples ≤ 100,000                          → CatBoost
n_samples > 100,000 and a categorical column
  has cardinality ≥ 50                       → CatBoost (native categorical handling wins)
otherwise                                    → LightGBM
```

With `ensemble="voting"` / `"stacking"` / `"auto"`, this stage hands off to
the ensemble decision engine — see [Ensembles](#ensembles) below.

### 5. Hyperparameter Optimization

Optuna with a `TPESampler` (or `RandomSampler`) and a `MedianPruner`. Grid
search is never used — for continuous, correlated hyperparameters like
learning rate and depth, TPE finds better configurations in far fewer
trials. Each trial is scored via cross-validation; within each fold, the
model trains with early stopping rather than a fixed tree count, so the
search implicitly discovers a good iteration count too, and a bad trial's
partial CV results can prune it before the remaining folds even run.

Trial count, fold count, and validation strategy all have `"auto"` rules
that scale with dataset size and available time budget (see `n_trials=`,
`cv=`, `validation=`, `time_limit=` in documents.md).

### 6. Training (+ Threshold Optimization)

The final model retrains on the *entire* training split (not just the
portion used during CV) with the best hyperparameters and iteration count
found. Training time and peak memory (sampled by a background thread every
50ms) are recorded here.

If the task has a meaningful decision threshold (binary, multi-class, or
multi-label — see the deep dive below), it's tuned immediately after this
step.

### 7. Evaluation

Metrics are computed on a **held-out test split**, never on cross-validation
folds — CV scores are for hyperparameter selection, not for reporting, since
they tend to be optimistic. `model.evaluate(X, y)` runs the exact same code
path on any data you give it later.

### 8. Explainability

Native feature importance always runs (CatBoost's `PredictionValuesChange`,
LightGBM's built-in importances; for ensembles, an importance-weighted
average across base learners). SHAP (`TreeExplainer`, mean |SHAP value| per
feature on up to 200 sampled rows) runs for single models on
binary/multiclass/regression/ranking; it's skipped — not failed — for
ensembles and for multi-label/multi-output-regression's per-target
sub-estimators, and any SHAP failure degrades to "no SHAP chart" rather than
crashing the run.

### 9. Report

Covered in full in documents.md's report section — this stage assembles
everything the previous eight produced into `report.html`, `report.json`,
and one PNG per chart.

## Ensembles

Four modes, `ensemble=`:

- **`"none"`** (default): the single model chosen in stage 4.
- **`"voting"`**: CatBoost + LightGBM + XGBoost are each tuned on 60% of the
  training data, scored on a held-out 20% slice, and combined via
  **weighted** soft voting — a model that scores better gets a
  proportionally larger vote, but no model's weight collapses to near
  zero. Voting probabilities are renormalized so every row sums to exactly
  1.0 (floating-point averaging can otherwise drift a hair off, which trips
  strict downstream checks like log-loss).
- **`"stacking"`**: the same three base models, but a
  `LogisticRegression`/`Ridge` meta-learner is trained on their outputs
  (using a separate 20% slice reserved for the meta-learner, so it never
  sees data the base models trained on either).
- **`"auto"`**: the decision engine. Rather than always paying for three
  models, it first tunes only CatBoost and LightGBM (half the trial budget
  each) and compares them:
  - If one is clearly ahead (≥ 1% relative difference), that model is
    retrained on the full training set and returned as-is — XGBoost is
    never trained and no ensemble is built.
  - If they're close (< 1% relative difference), the full voting/stacking
    comparison from above runs, and the better of the two combiners is
    built.
  - The resulting ensemble is then compared against the best single model
    from the first round — if it doesn't actually win, SmartTab falls back
    to the single model. Complexity is never added just because it was
    already computed.

  `ensemble="auto"` currently only supports binary, multi-class, and
  regression tasks — multi-label, multi-output regression, and ranking
  always use `ensemble="none"`.

## Threshold Optimization, Objective, and Confidence Scores

This is the piece worth explaining carefully, because the same two knobs
(`threshold_optimization=`, `objective=`) mean something different depending
on the task type.

### Why a threshold matters at all

CatBoost/LightGBM/XGBoost output *probabilities*. Turning a probability into
a hard prediction requires a cutoff, and the textbook cutoff — 0.5, or
argmax — is not usually the cutoff that maximizes the metric you actually
care about, especially on imbalanced data. SmartTab searches for a better
cutoff on held-out data instead of assuming the textbook one.

### The search mechanism

A model trained on 100% of the training data has no held-out slice left to
tune a threshold on without leaking test-like information back into the
tuning process. So:

- **Single model**: a disposable "probe" copy — same hyperparameters, same
  iteration count, no further search — is trained on 85% of the training
  data, and the threshold is tuned on the remaining 15%. The real
  production model is untouched by this; only the threshold value it
  learns is kept.
- **Ensembles** (`"voting"`/`"stacking"`/`"auto"`-ensemble): the 20% slice
  already reserved for scoring base models is reused — no extra fit needed.

The threshold itself comes from sweeping a grid (0.01 to 0.99, plus 0.0 for
the multi-class case below) and keeping whichever value maximizes
`objective` on that held-out slice.

### Binary classification

The classic case: one cutoff on P(positive class). `objective=` picks which
metric that cutoff maximizes — `"mcc"` by default (robust on imbalanced
data, unlike accuracy), or `"f1"`, `"precision"`, `"recall"`,
`"accuracy"`, `"balanced_accuracy"`, `"roc_auc"` (which is threshold-invariant
by definition, so this choice is a no-op on the threshold itself and just
reports ROC AUC). The learned threshold applies to *every* prediction
afterward (`predict()`, `evaluate()`, the report) — not just the number
shown once at fit time.

### Multi-class: a reject option, not a threshold

Multi-class predictions come from argmax over class probabilities — there's
no single "positive class" to put a cutoff on. So for multi-class, the
threshold is a **reject option**: a cutoff on the *top* predicted class's
probability. Below the cutoff, a row is "too uncertain to call."

This is deliberately **informational by default**. `threshold_optimization`
still runs the search and stores the result in `model.reject_threshold`, but
`predict()` keeps returning plain argmax labels unless you explicitly opt in
with `multi_threshold_ensemble=True` — otherwise, turning on threshold
tuning for a multi-class model could silently start returning `None` for
some predictions, which is too large a behavior change to make by default.

When scoring a candidate reject-threshold during the sweep, a rejected row
is counted as **wrong**, not excluded from the calculation — this is what
stops the search from "cheating" by rejecting almost everything to
inflate accuracy-on-the-remainder. Raising the threshold is a genuine
trade-off between coverage and correctness, and the objective reflects that
honestly.

### Multi-label: one threshold per label

Each label in a multi-label problem is independently binary — "does this
row have this label or not" — so this reduces to running the exact same
binary threshold search once per label column. Unlike multi-class, this
*is* applied by default (`threshold_optimization=True`'s default), because
it's a direct drop-in replacement for each label's 0.5 cutoff, not a new
"maybe" state.

### The confidence ladder (`multi_threshold_ensemble=`)

A single threshold, however well-tuned, only ever says "yes" or "no" — it
doesn't say how close the call was. `multi_threshold_ensemble=True` builds
`threshold_models` thresholds instead of one, spanning lenient to strict, all
computed from the *same* trained model's probabilities (no extra models are
trained). A prediction that still clears the strictest threshold is a
high-confidence call; one that only clears the most lenient threshold is a
borderline call.

- **Binary**: thresholds are spaced evenly across the recall/precision
  trade-off curve (derived from `precision_recall_curve`). Confidence for a
  row is the fraction of the ladder's levels that agree with its final
  label.
- **Multi-class**: thresholds are spaced evenly across the coverage curve
  (what fraction of rows would be accepted at each cutoff). Confidence is
  the fraction of levels that still accept the row; rows below the
  optimized reject threshold come back as `None` instead of a class label.
- **Multi-label**: one full ladder per label, and a confidence value per
  label per row.

This is the mechanism behind routing "maybe" predictions to a human —
medical screening, fraud review, content moderation — see documents.md for
worked examples.

## Class Imbalance

Detected once, in stage 1, as minority/majority ratio < 0.1. When
imbalanced, every training path (auto-search, `optimize=False`, or explicit
`params=`) applies balanced class weighting (`auto_class_weights="Balanced"`
for CatBoost, `class_weight="balanced"` for LightGBM), the default
optimization metric shifts away from raw accuracy (to ROC AUC for binary,
macro-F1 for multi-class) so a model can't win by only predicting the
majority class, and train/test splits are always stratified.

## Persistence

`save()`/`load()` use a zip bundle, not a raw pickle, specifically so the
native model format (CatBoost's `.cbm`, stable across CatBoost versions)
doesn't depend on Python pickle compatibility across library versions.
Everything else — the fitted cleaning pipeline, the target encoder, dataset
and hardware profiles, metrics, and every threshold/objective/ladder value —
is stored as plain JSON/joblib alongside it. See documents.md for the exact
file layout.

## Report Generation

Charts are built once with Plotly, exported as static PNGs (for embedding
elsewhere), and also embedded as interactive HTML — the plotly.js library
itself is injected once in the report's `<head>`, so chart rendering doesn't
depend on which section happens to appear first in the page.
