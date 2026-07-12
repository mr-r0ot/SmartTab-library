# SmartTab Documentation

This is a working guide, not a pitch — it teaches you `smarttab` from
`pip install` to production tuning, in the order you'll actually need it. For
the "why is it built this way" internals, see [HowItWorks.md](HowItWorks.md).

**Contents**

1. [Installation](#1-installation)
2. [Your First Model](#2-your-first-model)
3. [Loading Data](#3-loading-data)
4. [Making Predictions](#4-making-predictions)
5. [Reports: What's Inside](#5-reports-whats-inside)
6. [Saving and Loading Models](#6-saving-and-loading-models)
7. [Real Project: Customer Churn](#7-real-project-customer-churn)
8. [Parameters You'll Actually Use](#8-parameters-youll-actually-use)
9. [Medical and Other Sensitive Domains](#9-medical-and-other-sensitive-domains)
10. [Financial Domain](#10-financial-domain)
11. [Task Types Beyond Binary Classification](#11-task-types-beyond-binary-classification)
12. [Ensembles](#12-ensembles)
13. [Threshold Optimization and Confidence Scores](#13-threshold-optimization-and-confidence-scores)
14. [Full Parameter Reference](#14-full-parameter-reference)
15. [Persistence Format](#15-persistence-format)
16. [Errors You Might See](#16-errors-you-might-see)
17. [What's Not Here Yet](#17-whats-not-here-yet)

---

## 1. Installation

```bash
pip install smarttab                  # core: CatBoost + LightGBM + Optuna + reporting
pip install "smarttab[voting]"        # + XGBoost, needed for ensemble="voting"/"stacking"/"auto"
pip install "smarttab[gpu]"           # + GPU detection
pip install "smarttab[all]"           # everything
```

`smarttab[voting]` is worth calling out: XGBoost is never selectable on its
own — it only ever appears as one of three base models inside an ensemble.
If you don't plan to use `ensemble=`, you don't need it.

## 2. Your First Model

```python
import smarttab

model = smarttab.fit(df, target="churned")

model.predict(new_df)
model.evaluate(X_test, y_test)
model.report("my_report")
model.save("model.smarttab")
```

That one call to `fit()`:

1. Looks at your `target` column and figures out what kind of problem this
   is (binary classification, in this example).
2. Cleans the data — imputes missing values, encodes categoricals, drops
   useless columns.
3. Checks your CPU/RAM/GPU and tunes resource usage accordingly.
4. Picks CatBoost or LightGBM based on your dataset's size and shape.
5. Runs a hyperparameter search with Optuna.
6. Trains the final model and finds a better decision threshold than the
   textbook 0.5 cutoff.
7. Evaluates it on data the model never trained on.
8. Computes feature importance and SHAP values.
9. Writes an HTML report, if you don't turn it off.

You don't have to think about any of that to get a usable model. You *can*
override every single step — that's what the rest of this document is for.

## 3. Loading Data

`data` accepts either a `pandas.DataFrame` you already have in memory, or a
path to a file:

```python
import pandas as pd
import smarttab

# from a DataFrame you built or already loaded
df = pd.read_csv("customers.csv")
model = smarttab.fit(df, target="churned")

# or hand smarttab the path directly — it reads it for you
model = smarttab.fit("customers.csv", target="churned")
```

Supported file extensions: `.csv`, `.tsv`, `.xlsx`/`.xls`, `.parquet`,
`.json`, `.feather`, `.pkl`/`.pickle`. The extension decides how the file is
read; there's nothing else to configure. If the file doesn't exist or the
extension isn't supported, you get a `DataValidationError` telling you
exactly what went wrong.

There's no separate "load my data" step to learn — `fit()` is the loading
step.

## 4. Making Predictions

```python
model.predict(new_df)          # predicted labels / values
model.predict_proba(new_df)    # class probabilities (classification only)
model.evaluate(X_test, y_test) # full metric dict on labeled data
```

`predict()` returns predictions in your original label space — if your
target column had `"yes"`/`"no"` strings, that's what comes back, not `0`/`1`
codes. The input needs the same feature columns the model was trained on;
extra columns are ignored, and SmartTab's cleaning pipeline (encoders,
imputers) is reapplied automatically — you never call it yourself.

### Whatever shape your data is in

`predict()` (and `predict_proba()`) don't just take a `DataFrame` — they take
whatever's actually sitting in front of you: a file on disk, a single record
someone just sent you as JSON, a batch of them, or a raw array. There's
nothing to convert first:

```python
model.predict(df)                       # a DataFrame
model.predict("new_customers.csv")      # a path — CSV, TSV, Excel, Parquet, JSON, Feather, Pickle
model.predict(sample)                   # one record, as a plain dict
model.predict([sample_1, sample_2])     # a batch of records, as a list of dicts
model.predict(numpy_array)              # a raw array — 1-D for one row, 2-D for a batch
```

A single dict is the common case for "predict on the one thing a user just
submitted" — a form, an API request body, a row someone typed by hand:

```python
sample = {
    "gender": "Male",
    "age": 45,
    "hypertension": 1,
    "heart_disease": 0,
    "smoking_history": "former",
    "bmi": 28.5,
    "HbA1c_level": 6.2,
    "blood_glucose_level": 145,
}

result = model.predict(sample)
```

### The result is still a plain array — with extras

`predict()` on a single dict or a batch always returns something you can use
exactly like before: index it, slice it, compare it, loop over it. Nothing
about existing code that does `preds = model.predict(X)` and then treats
`preds` as a NumPy array needs to change. On top of that, the same object
also exposes a few named shortcuts, so a one-off prediction doesn't need to
be pulled out of an array just to read it:

```python
result = model.predict(sample)

print(result)              # [1]  — still a real array
print(result.prediction)   # 1    — a bare value, since the input was a single record
print(result.label)        # 1    — same thing, whichever name reads better in your code
print(result.csv)          # prediction\n1\n
print(result.json)         # {"prediction": 1}
```

For a batch, `.prediction`/`.label` are the full array and `.csv`/`.json` are
the full table — one row per input record:

```python
results = model.predict(new_customers_df)
print(results.csv)
# prediction
# 1
# 0
# 1
```

```python
print(results.json)
# [{"prediction": 1}, {"prediction": 0}, {"prediction": 1}]
```

That `.json` output is a plain string — ready to hand straight to a web
framework's response, write to a file, or log — no `json.dumps` needed on
your end, no manual `DataFrame.to_csv` either. Loading the input and reading
the result are both a one-liner, from JSON, CSV, or nothing more exotic than
Python's own dict and list.

When `multi_threshold_ensemble=True`, `predict()` still returns
`(labels, confidence)` exactly as documented in [section 13](#13-threshold-optimization-and-confidence-scores)
— unpacking it works precisely like before — but that same value also has
`.prediction` (or `.label`) and `.probability`, plus `.csv`/`.json` that
include both columns:

```python
labels, confidence = model.predict(sample)   # still works, unchanged

result = model.predict(sample)
print(result.label, result.probability)   # 1 0.75
print(result.json)                        # {"prediction": 1, "probability": 0.75}
```

```python
churn_risk = model.predict(new_customers)
probabilities = model.predict_proba(new_customers)   # shape (n, 2) for binary
```

`evaluate()` runs the same prediction path and returns every metric
SmartTab computes for that task type (see [section 13](#13-threshold-optimization-and-confidence-scores)
for how classification thresholds factor in). It also caches its result, so
a `report()` call right after doesn't need labeled data passed again.

## 5. Reports: What's Inside

```python
model.report("my_report")
```

This creates `my_report/` (if it doesn't exist yet) with three things:

```
my_report/
  report.html            # open this in a browser — fully interactive, no internet needed
  report.json            # the exact same data as plain JSON
  charts/
    metrics.png              # bar chart of every scalar metric
    timing.png                # training/prediction speed
    memory.png                 # peak training memory vs. total system RAM
    feature_importance.png
    shap_importance.png        # only present if SHAP ran (single models only)
    diagnostic.png              # ROC curve or confusion matrix (classification),
                                 # per-label accuracy (multi-label),
                                 # per-output RMSE (multi-output regression),
                                 # or predicted-vs-actual (regression/ranking)
    threshold_ladder.png        # only if multi_threshold_ensemble=True — section 13
```

`report.html` includes everything above plus a dataset summary, a hardware
summary, the model's hyperparameters, and a running log of every decision
SmartTab made (`model.notes`). It's meant to be something you can actually
hand to a teammate or a stakeholder, not just a debug dump.

`report.json` (and the dict `report()` returns) has the same content as
structured data: `metrics`, `best_params`, `feature_importance`,
`dataset_profile`, `hardware_profile`, `ensemble_info`, `decision_threshold`,
and more — the full key list is in [section 14](#14-full-parameter-reference).

If you don't pass evaluation data to `report()`, it reuses whatever
`evaluate()` (or `fit()`'s own internal evaluation) last computed:

```python
model.report("my_report")                      # reuses fit()'s own held-out evaluation
model.report("my_report", X_test, y_test)       # or evaluate explicitly first
```

A freshly `load()`-ed model has no cached evaluation from a previous
session, so you must pass `X, y` the first time you call `report()` on it.

## 6. Saving and Loading Models

```python
model.save("model.smarttab")

loaded = smarttab.load("model.smarttab")
loaded.predict(new_df)   # works immediately, no extra setup
```

The file extension doesn't matter — `.smarttab` is a convention, not a
requirement; it's really just a zip archive. Everything needed to reproduce
predictions is inside it: the trained model(s), the fitted cleaning
pipeline, the target label encoder, and every decision SmartTab made
(decision threshold, chosen hyperparameters, dataset/hardware profile). See
[section 15](#15-persistence-format) for the exact file layout.

## 7. Real Project: Customer Churn

Let's put sections 2–6 together on something with a real shape: predicting
which customers are about to cancel, from a handful of account attributes.

First, the data. In a real project this line would just be
`pd.read_csv("customers.csv")` — here we fabricate a small stand-in dataset
so the example runs on its own, using a plain, one-line business rule
("flexible contract + several support tickets → likely to churn") instead
of anything you'd need to study to follow along:

```python
import numpy as np
import pandas as pd
import smarttab

rng = np.random.default_rng(42)
n = 2000

df = pd.DataFrame({
    "tenure_months": rng.integers(1, 72, n),
    "monthly_charges": rng.normal(70, 20, n).round(2),
    "contract_type": rng.choice(["month-to-month", "one-year", "two-year"], n),
    "support_tickets": rng.poisson(1.5, n),
    "has_addon": rng.integers(0, 2, n),
})

df["churned"] = ((df["contract_type"] == "month-to-month") & (df["support_tickets"] >= 2)).astype(int)
flip = rng.random(n) < 0.15   # a bit of noise so it's not a trivially perfect rule
df.loc[flip, "churned"] = 1 - df.loc[flip, "churned"]
```

Now the part that's actually about SmartTab:

```python
model = smarttab.fit(df, target="churned")

print("Model chosen:", model.model_name)
print("Metrics:", model.metrics)
```

That's the whole training step — no manual train/test split, no scaling,
no encoding `contract_type` by hand (it's a plain string column). Now apply
it to a couple of new signups:

```python
new_customers = pd.DataFrame([
    {"tenure_months": 2, "monthly_charges": 95.0, "contract_type": "month-to-month",
     "support_tickets": 4, "has_addon": 0},
    {"tenure_months": 48, "monthly_charges": 60.0, "contract_type": "two-year",
     "support_tickets": 0, "has_addon": 1},
])
print(model.predict(new_customers))   # [1 0] — the first looks like a churn risk, the second doesn't

model.report("reports/churn_model")
model.save("models/churn_model.smarttab")
```

Open `reports/churn_model/report.html` afterward and you'll find exactly
which features drove the prediction (feature importance and SHAP), what
threshold was used to turn a probability into "will churn" / "won't
churn", and the full metric breakdown on data the model never saw during
training.

This is the shape of most SmartTab projects: load real data, `fit()`, look
at the report, `predict()` on what you actually need to predict. The next
few sections dig into the levers you'll reach for once "just call `fit()`"
isn't quite enough — starting with the parameters people actually change,
then two domains (medical, financial) where the default settings usually
need a deliberate second look.

## 8. Parameters You'll Actually Use

The full parameter list is long (section 14 has all of it), but in practice
most projects only ever touch a handful of them:

```python
model = smarttab.fit(
    df, target="label",
    model="auto",                    # or "catboost" / "lightgbm"
    ensemble="none",                 # or "voting" / "stacking" / "auto"
    test_size=0.3,                   # train/test split ratio
    optimize=True,                   # hyperparameter search on/off
    time_limit=0,                    # wall-clock cap for the whole fit(), in seconds
    threshold_optimization=True,     # tune the decision threshold instead of using 0.5
    objective="mcc",                 # which metric that threshold search maximizes
    multi_threshold_ensemble=False,  # confidence scoring alongside each prediction
    threshold_models=4,              # how many confidence levels, if the above is True
)
```

- **`model=`** — leave it on `"auto"` unless you have a specific reason to
  force CatBoost or LightGBM. `"auto"` picks based on dataset size and
  categorical cardinality (details in HowItWorks.md).
- **`ensemble=`** — `"none"` trains one model. `"auto"` is the "I want the
  best result I can get and I'm willing to pay a bit more compute for it"
  setting — full details in [section 12](#12-ensembles).
- **`test_size=`** — the fraction of data held out for the final,
  never-trained-on evaluation. `0.3` (70/30 split) is a safe default for
  most dataset sizes; shrink it toward `0.1`–`0.15` once you have tens of
  thousands of rows and want more data for training.
- **`optimize=`** — `False` skips the hyperparameter search entirely and
  uses sane defaults instead. Useful for a fast first pass while you're
  still exploring a dataset; flip it back to `True` for the model you'll
  actually keep.
- **`time_limit=`** — a wall-clock budget in seconds for the *entire*
  `fit()` call, AutoGluon-style. `0` means no limit. Set this instead of
  guessing at `n_trials=` when you know how long you're willing to wait but
  not how large your dataset is.
- **`threshold_optimization=`, `objective=`, `multi_threshold_ensemble=`,
  `threshold_models=`** — these four control how a probability becomes a
  final prediction, and whether you get a confidence score alongside it.
  They matter enough to have their own deep dive: [section 13](#13-threshold-optimization-and-confidence-scores).
  The two domain sections right after this one (medical, financial) show
  them in a concrete setting before that deep dive gets into the mechanics.

## 9. Medical and Other Sensitive Domains

In screening-type problems — a diagnostic test, a safety check, anything
where a missed positive case is far more costly than a false alarm — the
default settings are the wrong starting point for two reasons:

1. **The default `objective="mcc"` balances all kinds of errors evenly.**
   In screening, missing a sick patient (a false negative) is usually much
   worse than flagging a healthy one for a second look (a false positive).
   You want the threshold search biased toward catching positives —
   `objective="balanced_accuracy"` is usually the right amount of bias for
   this (it weighs both classes equally regardless of how rare positives
   are, so a rare disease doesn't get ignored). Reach for
   `objective="recall"` only when a missed positive is so costly that
   false alarms are basically free by comparison: raw recall has no
   counterweight at all, so pushed far enough it will happily flag
   *everyone* as positive if that's what maximizes it — fine if a human
   reviews every flagged case anyway, not fine if you wanted a selective
   filter.
2. **A single yes/no threshold hides how close the call was.** A patient
   the model is 51% confident about and one it's 99% confident about get
   the exact same binary label. In a clinical workflow, you want the
   borderline cases flagged for a human to look at, not silently decided
   by whichever side of 0.5 they landed on — that's what
   `multi_threshold_ensemble=True` is for.

Again, the data first — a stand-in for a real screening dataset, using one
plain rule ("high glucose plus a family history" tends toward a diabetic
label) instead of a formula to study:

```python
import numpy as np
import pandas as pd
import smarttab

rng = np.random.default_rng(7)
n = 3000

patients = pd.DataFrame({
    "age": rng.integers(20, 85, n),
    "bmi": rng.normal(27, 5, n).round(1),
    "glucose_level": rng.normal(100, 25, n).round(1),
    "family_history": rng.integers(0, 2, n),
    "physical_activity_hours": rng.gamma(2, 2, n).round(1),
})

patients["diabetic"] = ((patients["glucose_level"] > 130) & (patients["family_history"] == 1)).astype(int)
flip = rng.random(n) < 0.04
patients.loc[flip, "diabetic"] = 1 - patients.loc[flip, "diabetic"]
```

Now fit with `objective="balanced_accuracy"` and `multi_threshold_ensemble=True`,
and actually use the confidence score it gives back — not just print it, but
turn it into something you'd hand to a clinician: which predictions to trust
and which to double-check.

```python
model = smarttab.fit(
    patients, target="diabetic",
    objective="balanced_accuracy",      # bias toward catching positives, without ignoring false alarms entirely
    multi_threshold_ensemble=True,      # get a confidence score per prediction
    threshold_models=4,
    verbose=0,
)

new_patients = pd.DataFrame([
    {"age": 61, "bmi": 31.2, "glucose_level": 148, "family_history": 1, "physical_activity_hours": 1.0},
    {"age": 34, "bmi": 23.5, "glucose_level": 91,  "family_history": 0, "physical_activity_hours": 5.5},
    {"age": 45, "bmi": 26.0, "glucose_level": 126, "family_history": 1, "physical_activity_hours": 2.5},
])

labels, confidence = model.predict(new_patients)

results = new_patients[["age", "glucose_level", "family_history"]].copy()
results["prediction"] = labels
results["needs_review"] = confidence <= 0.5   # route the borderline calls to a clinician
print(results.to_string(index=False))
```

```
 age  glucose_level  family_history  prediction  needs_review
  61            148               1           1         False
  34             91               0           0         False
  45            126               1           0          True
```

The first two are confident calls the ladder agrees on at every level; the
third — elevated glucose, a family history, but not clearly over the line —
is exactly the case `multi_threshold_ensemble` is for: instead of a silent
"not diabetic" decided by whichever side of one threshold it landed on, it
comes back flagged for a second look. Where that `needs_review` cutoff
should sit (0.5? 0.75?) is a workflow decision for whoever owns the clinical
process, not something SmartTab decides for you — it just gives you the
number.

```python
model.report("reports/diabetes_screening", patients.drop(columns=["diabetic"]), patients["diabetic"])
```

`model.report(...)` for this model includes a full table of the confidence
ladder's levels (the precision/recall trade-off at each cutoff), which is
worth reviewing with whoever owns the clinical or safety workflow before
this goes anywhere near production.

## 10. Financial Domain

Credit and fraud problems usually pull in the opposite direction from
medical screening: false positives have a direct, immediate cost too —
rejecting a good loan applicant or freezing a legitimate transaction loses
revenue and trust, not just goodwill. This is exactly the case `"mcc"` (the
default `objective`) is designed for: it accounts for all four outcomes
(true/false positive/negative) evenly rather than optimizing recall or
precision in isolation, which matters when both kinds of mistakes are
expensive in real money.

For a decision this consequential, it's also usually worth paying for
`ensemble="auto"` — SmartTab will only keep the extra complexity if it
actually beats a single tuned model, so you're not trading interpretability
for nothing.

```python
import numpy as np
import pandas as pd
import smarttab

rng = np.random.default_rng(3)
n = 3000

loans = pd.DataFrame({
    "annual_income": rng.normal(55000, 20000, n).round(0),
    "loan_amount": rng.normal(15000, 8000, n).round(0),
    "credit_score": rng.normal(680, 60, n).round(0),
    "existing_defaults": rng.poisson(0.3, n),
})

loans["defaulted"] = ((loans["credit_score"] < 620) & (loans["existing_defaults"] >= 1)).astype(int)
flip = rng.random(n) < 0.05
loans.loc[flip, "defaulted"] = 1 - loans.loc[flip, "defaulted"]

model = smarttab.fit(
    loans, target="defaulted",
    ensemble="auto",   # worth the extra compute for a high-stakes decision
    objective="mcc",   # the default — balances false approvals against false rejections
    verbose=0,
)

print("Chosen strategy:", model.model_name)
print("Metrics:", model.metrics)

new_applicants = pd.DataFrame([
    {"annual_income": 42000, "loan_amount": 18000, "credit_score": 590, "existing_defaults": 1},
    {"annual_income": 95000, "loan_amount": 12000, "credit_score": 740, "existing_defaults": 0},
])
print(model.predict(new_applicants))   # [1 0] — the thin-credit applicant flagged, the strong one approved

model.report("reports/loan_default_model")
model.save("models/loan_default_model.smarttab")
```

Two things worth checking in the report before trusting a model like this
with real decisions: `dataset_profile.class_imbalance_ratio` (defaults are
almost always the minority class — SmartTab applies balanced class
weighting automatically when this is detected, see section 13) and
`feature_importance` (make sure the model is leaning on signals a regulator
or an auditor would consider legitimate, not a proxy for something it
shouldn't be).

**Is `objective="precision"` a better fit than `"mcc"` here?** It's a fair
question — in lending, a false positive means rejecting a good customer,
and precision is literally "how many of the applicants I flagged actually
defaulted." But precision alone has no counterweight, the same way raw
recall didn't in section 9: pushed to its logical extreme, a precision-only
search can satisfy itself by flagging almost nobody — a handful of the most
obvious defaults — and quietly let most real defaulters through with an
"approved" label, since it's never penalized for the ones it missed. Testing
this on a messier, more realistic version of the dataset above
(continuous features instead of a clean-cut rule) made the failure mode
obvious: `objective="precision"` reached 100% precision by only catching
13% of actual defaulters (recall ≈ 0.13), while `objective="mcc"` on the
same data caught 91% of them (recall ≈ 0.91) at a still-reasonable 36%
precision. That's a worse outcome for a lender who's trying to catch risk,
not just avoid false alarms. `mcc` (or `balanced_accuracy`) is the safer
first choice; if you do reach for `objective="precision"`, check
`model.metrics["recall"]` afterward to make sure it hasn't collapsed.

If your priorities lean more toward "never let a bad account slip through"
than balanced accuracy, the same `objective="recall"` idea from the medical
section applies here too — it's a business decision, not a technical one,
and SmartTab just needs you to name it.

## 11. Task Types Beyond Binary Classification

Everything above used binary classification (`churned`, `diabetic`,
`defaulted`: 0 or 1). SmartTab detects five other task types automatically
from the *shape* of `target` — there's no `task_type=` parameter to set.

| Task type | How it's detected | `target` |
|---|---|---|
| Binary classification | one column, ≤ 2 unique values | a string |
| Multi-class | one column, 3–20 unique discrete values, or string/categorical | a string |
| Regression | one column, continuous (> 20 unique values or has decimals) | a string |
| **Multi-label** | multiple columns, all 0/1 or bool | a **list of strings** |
| **Multi-output regression** | multiple columns, all continuous | a **list of strings** |
| **Ranking** | `group_id=` is given | a string (the relevance/label column) |

```python
# Binary / multi-class / regression — a single target column, same as every example so far
smarttab.fit(df, target="churned")

# Multi-label — several 0/1 columns predicted together
smarttab.fit(df, target=["has_cats", "has_dogs", "has_birds"])
preds = model.predict(new_df)         # shape (n, 3) — one 0/1 per label
proba = model.predict_proba(new_df)   # shape (n, 3) — positive-class probability per label

# Multi-output regression — several continuous columns predicted together
smarttab.fit(df, target=["price", "demand"])
preds = model.predict(new_df)               # shape (n, 2)
print(model.metrics["rmse_per_output"])     # [rmse_price, rmse_demand]

# Ranking — group_id is the query/session column; target is a relevance score per row
smarttab.fit(df, target="relevance", group_id="query_id")
scores = model.predict(new_df)                                   # raw ranking scores, not probabilities
metrics = model.evaluate(X_test, y_test, groups=query_ids_test)   # groups is required here
print(metrics["ndcg@10"])
```

A few things specific to these three:

- **Multi-label and multi-output regression** train one independent model
  per target column under the hood (via
  `sklearn.multioutput.MultiOutputClassifier`/`Regressor`) rather than
  chasing each library's native multi-output loss — simpler and more
  stable across both CatBoost and LightGBM, at the cost of training time
  scaling roughly linearly with the number of target columns.
- **Ranking** never splits rows from the same `group_id` across train/test
  or CV folds. `predict_proba()` doesn't apply to a ranking model (there's
  no probability, only a relative score) and raises `SmartTabError` if you
  call it; `evaluate()`/`report()` require `groups=` since NDCG is only
  meaningful within a group.
- **Known limitations of these three, for now**: `ensemble=` is only
  available for binary/multiclass/regression — using it with
  multilabel/multioutput/ranking raises `ConfigurationError`.
  `threshold_optimization`/`multi_threshold_ensemble` do work for
  multilabel with `ensemble="none"` (see section 13); they don't apply to
  multi-output regression or ranking, which have no decision threshold to
  speak of. SHAP only runs for single models on
  binary/multiclass/regression/ranking — it's silently skipped (not an
  error, just a missing chart) for multilabel/multioutput.

## 12. Ensembles

`ensemble=` has four modes:

| Value | What happens |
|---|---|
| `"none"` (default) | The single model from `model=`, no extra cost |
| `"voting"` | CatBoost + LightGBM + XGBoost, combined via **weighted** soft voting |
| `"stacking"` | The same three models, combined via a logistic/ridge meta-learner |
| `"auto"` | A decision engine that only builds an ensemble when it's worth it — see below |

### `ensemble="voting"` / `"stacking"`

```python
model = smarttab.fit(df, target="label", ensemble="voting")
```

Training data is split three ways: 60% to train all three base models, 20%
to train the meta-learner (stacking only), 20% to score everything. Each
base model gets an equal share of the `n_trials` budget. For voting, each
base model's weight comes from its own held-out score — a stronger model
gets a bigger vote, but none collapses to zero. The result (strategy,
validation score, per-model scores) is available in `model.ensemble_info`.

The trade-off: each base model trains on less data than it would as a
standalone model, in exchange for both the voting weights and the final
comparison being computed with zero data leakage.

### `ensemble="auto"` — the decision engine

```python
model = smarttab.fit(df, target="label", ensemble="auto")
```

Rather than always training three models (expensive), this tunes just
CatBoost and LightGBM first and compares them:

1. If one is clearly ahead (≥ 1% relative difference), it's retrained on
   the full training set and returned as-is — XGBoost is never touched and
   no ensemble is built.
2. If they're close (< 1%), the full voting/stacking comparison above runs.
3. The resulting ensemble is compared against the best single model from
   step 1 — if it doesn't actually win, SmartTab falls back to the single
   model rather than shipping unnecessary complexity.

```python
model = smarttab.fit("data.csv", target="label", ensemble="auto", n_trials=30)
print(model.model_name)          # "catboost" | "lightgbm" | "voting" | "stacking"
print(model.ensemble_info)       # None if it decided a single model was enough
```

The full decision trail is in `model.notes`, so you can see exactly why it
landed where it did.

`ensemble=` (any mode other than `"none"`) is only available for
binary/multiclass/regression — it raises `ConfigurationError` for
multilabel/multioutput/ranking (section 11).

## 13. Threshold Optimization and Confidence Scores

This is the deep dive that sections 8–10 pointed to. Four parameters, and
what they mean depends on the task type.

### The problem

CatBoost/LightGBM/XGBoost output *probabilities*. Turning that into a hard
prediction needs a cutoff, and the textbook cutoff (0.5, or argmax for
multi-class) is rarely the one that maximizes the metric you actually care
about — especially on imbalanced data. `threshold_optimization=True`
(the default) searches for a better one on held-out data instead.

### `objective=` — what the search maximizes

Default is `"mcc"` (Matthews Correlation Coefficient) — a metric that stays
meaningful on imbalanced data and, unlike raw accuracy, can't be gamed by
always predicting the majority class. Other options:
`"f1"`, `"precision"`, `"recall"`, `"accuracy"`, `"balanced_accuracy"`,
`"roc_auc"` (this last one is threshold-invariant by definition, so it's a
no-op on the threshold itself — the threshold stays at its default and only
the ROC AUC number gets reported).

`objective=` only affects which threshold gets picked — it has nothing to
do with `metrics=`, which controls what the hyperparameter search itself
optimizes. Those two are completely independent.

### What "threshold" means per task type

| Task type | What gets tuned | Applied to `predict()` by default? |
|---|---|---|
| Binary | a cutoff on P(positive class) | Yes — `model.decision_threshold` |
| Multi-class | a **reject cutoff** on the top predicted class's probability | No — informational only (`model.reject_threshold`) unless `multi_threshold_ensemble=True` |
| Multi-label | one independent cutoff per label | Yes — `model.per_label_thresholds` |

Binary and multi-label thresholds are direct, drop-in replacements for a
0.5 cutoff, so they apply automatically. Multi-class is different: there's
no single "positive class" to threshold, so a threshold here means
*rejecting* a prediction as too uncertain — and automatically turning some
predictions into `None` by default would be a much bigger, more surprising
behavior change than swapping one number for another. So for multi-class,
`threshold_optimization=True` computes and stores `model.reject_threshold`,
but `predict()` keeps returning plain argmax labels until you explicitly
opt in with `multi_threshold_ensemble=True`.

```python
model = smarttab.fit(df, target="spam")
print(model.decision_threshold)   # e.g. 0.42, not the textbook 0.5

model_multiclass = smarttab.fit(df, target="species")   # 3 classes
print(model_multiclass.reject_threshold)                # e.g. 0.34 — informational only
print(model_multiclass.predict(new_df))                 # still plain argmax labels

model_multilabel = smarttab.fit(df, target=["has_cats", "has_dogs", "has_birds"])
print(model_multilabel.per_label_thresholds)             # e.g. [0.42, 0.55, 0.31] — applied automatically
```

When scoring candidate thresholds for multi-class, a rejected row counts as
**wrong**, not excluded — this stops the search from inflating its score by
rejecting almost everything except the most obvious cases.

### `multi_threshold_ensemble=` and `threshold_models=` — confidence scores

A single threshold, however well-chosen, only ever says yes or no. Setting
`multi_threshold_ensemble=True` builds `threshold_models` thresholds (default
4) instead of one — spanning lenient to strict, all computed from the same
trained model's probabilities (no extra models are trained) — and
`predict()` returns `(labels, confidence)` instead of just labels.
`confidence` is the fraction of those levels that still agree with the
final prediction: 1.0 means even the strictest level agrees (a confident
call), a low value means only the most lenient level does (borderline).

```python
model = smarttab.fit(df, target="sick", objective="recall",
                      multi_threshold_ensemble=True, threshold_models=4)

labels, confidence = model.predict(new_df)
```

That tuple-unpacking always works, but the same return value is also usable
without unpacking it — `result = model.predict(new_df)` gives you
`result.label` and `result.probability` directly, plus `result.csv` /
`result.json` for exporting both columns at once. See
[section 4](#4-making-predictions) for the full rundown.

Per task type:

- **Binary**: the ladder spans the recall/precision trade-off curve. Each
  level reports `threshold`, `precision`, `recall`, `f1`, `accuracy`,
  `predicted_positive_rate`.
- **Multi-class**: the ladder spans coverage (what fraction of rows would
  be accepted at each cutoff). Each level reports `threshold`, `coverage`,
  `accuracy_on_accepted`, `accuracy_overall`, `n_accepted`. This is the one
  case where turning `multi_threshold_ensemble` on *does* change
  `predict()`'s labels: rows below `reject_threshold` come back as `None`.
- **Multi-label**: one full ladder per label; `predict()` returns a
  `(n_samples, n_labels)` confidence matrix, one column per label.

On a highly separable and/or heavily imbalanced dataset, a ladder built
naively from the full recall/precision (or coverage) curve can degenerate:
one end collapses to "predicts literally everyone" and the other to
"predicts almost nobody" — neither is a useful confidence tier, and having
either in the ladder just adds noise to the confidence score. SmartTab
constrains ladder candidates to the same sane threshold range the single
decision threshold itself is chosen from, and additionally requires every
level to actually predict positive (or accept, for multi-class) for at
least 1% of rows — so every level is always a genuinely usable operating
point, not a mathematical extreme of the curve.

This works with `ensemble="voting"`/`"stacking"`/`"auto"` for binary and
multi-class too (the ladder is built from the same held-out slice used to
score the base models — no extra fit). `ensemble=` isn't available for
multi-label at all (section 12), independent of this feature.

**This entire behavior is opt-in.** With `multi_threshold_ensemble=False`
(the default), `predict()` returns exactly what it always did — a plain
array, never a tuple.

`report()` adds a matching section (table + chart) whenever
`multi_threshold_ensemble=True`, and `report_dict["threshold_ladder"]` /
`report_dict["threshold_ladder_summary"]` carry the same data as JSON. All
of it — the ladder, the flag, the thresholds — round-trips through
`save()`/`load()`.

### A note on imbalanced data

Independent of anything above: whenever SmartTab detects class imbalance
(minority/majority ratio < 0.1), it automatically applies balanced class
weighting during training (`auto_class_weights="Balanced"` for CatBoost,
`class_weight="balanced"` for LightGBM) and shifts the default optimization
metric away from raw accuracy (to ROC AUC for binary, macro-F1 for
multi-class) — regardless of `optimize=`, and even if you pass `params=`
explicitly. This happens whether or not you're also using
`threshold_optimization`/`objective=`; the two mechanisms are independent
and both fire automatically once imbalance is detected.

## 14. Full Parameter Reference

```python
smarttab.fit(
    data, target, group_id=None,
    model="auto", ensemble="none",
    clean="auto", missing="auto", categorical="auto", scaling="auto",
    outlier="auto", feature_selection="auto",
    test_size=0.3,
    validation="auto", cv="auto",
    optimize=True, optimizer="auto", n_trials="auto", timeout=None,
    time_limit=0, threshold_optimization=True, objective="mcc",
    multi_threshold_ensemble=False, threshold_models=4,
    device="auto", cpu_threads="auto", gpu_memory="auto", ram_limit="auto",
    metrics="auto", params=None,
    report=True, explain=True,
    random_state=42, verbose=1,
)
```

| Parameter | Default | Notes |
|---|---|---|
| `target` | — | Required. A string for single-target problems, or a **list of strings** for multi-label/multi-output regression — section 11 |
| `group_id` | `None` | Column name for the query/group id. Giving it means the problem is **ranking** (`target` must be a single column) — section 11 |
| `model` | `"auto"` | `"auto"` \| `"catboost"` \| `"lightgbm"`. Ignored whenever `ensemble` isn't `"none"` |
| `ensemble` | `"none"` | `"none"` \| `"voting"` \| `"stacking"` \| `"auto"` — section 12. Only for binary/multiclass/regression |
| `missing` | `"auto"` | Only `"auto"` is supported: median for numeric columns, a reserved token for categorical |
| `categorical` | `"auto"` | Only `"auto"`: ordinal-integer encoding (not one-hot) |
| `scaling` | `"auto"` | `"auto"` = no-op (tree models don't need scaling). Explicit values: `"standard"`, `"minmax"`, `"robust"`, `"none"` |
| `outlier` | `"auto"` | `"auto"` only detects and reports outliers, never removes rows. `"remove"` enables IQR-based removal (training data only, before the split) |
| `feature_selection` | `"auto"` | Drops only near-duplicate numeric columns (correlation ≥ 0.999) |
| `test_size` | `0.3` | Held-out test fraction (70/30 split by default) |
| `validation` | `"auto"` | Validation strategy during the hyperparameter search: `"auto"` (k-fold for most datasets, single holdout above 200,000 rows), or explicit `"kfold"` / `"holdout"` |
| `cv` | `"auto"` | Fold count: `"auto"` = 3 for most datasets, 5 at ≥ 20,000 rows. Or an explicit integer ≥ 2 |
| `optimize` | `True` | `False` skips the hyperparameter search and uses sane defaults (faster, lower quality) |
| `optimizer` | `"auto"` | `"auto"`/`"tpe"` = Optuna's TPE sampler, `"random"` = random sampler. Grid search is never used |
| `n_trials` | `"auto"` | Trial count, scaled to dataset size and hardware. Effectively overridden by `time_limit` when that's set |
| `timeout` | `None` | Seconds, for that one Optuna call only (use this if you want manual control without `time_limit`'s budgeting logic) |
| `time_limit` | `0` | ⭐ Wall-clock cap (seconds) for the **entire** `fit()` call, AutoGluon-style. `0` = unlimited |
| `threshold_optimization` | `True` | ⭐ Tunes a decision threshold (binary/multilabel) or reject threshold (multiclass) instead of each library's raw default — section 13 |
| `objective` | `"mcc"` | ⭐ Which metric the threshold search maximizes — section 13 |
| `multi_threshold_ensemble` | `False` | ⭐ Builds a confidence ladder and makes `predict()` return `(labels, confidence)` — section 13 |
| `threshold_models` | `4` | Number of ladder levels, only used when `multi_threshold_ensemble=True`. Must be ≥ 2 |
| `device` | `"auto"` | `"auto"` \| `"cpu"` \| `"gpu"`. Auto only uses the GPU when enough free VRAM is available *and* the dataset is large enough (≥ 5,000 rows); falls back to CPU automatically on any GPU failure |
| `cpu_threads` | `"auto"` | `"auto"` = physical cores − 1 |
| `gpu_memory` | `"auto"` | Currently informational only |
| `ram_limit` | `"auto"` | `"auto"` = up to 85% of free RAM (always leaves ≥ 15% free). Can be a fraction (0–1) or an absolute value in MB |
| `metrics` | `"auto"` | Primary hyperparameter-search metric: `rmse` for regression, `roc_auc` for binary, `f1_macro` for multi-class |
| `params` | `None` | Manual hyperparameter dict for the chosen library. When given, the automatic search is skipped entirely and only the tree count is discovered via a quick early-stopping probe |
| `report` | `True` | Auto-writes a report to `smarttab_reports/<model>_<timestamp>/` at the end of `fit()` |
| `explain` | `True` | Computes feature importance (and SHAP, for single models) |
| `random_state` | `42` | Seed for reproducibility everywhere (splits, models, Optuna) |
| `verbose` | `1` | `0` = silent, `1` = stage messages + a live progress bar during the hyperparameter search, `2` = full debug output |

Fields available on the returned model object: `model_name`, `task_type`,
`metrics`, `best_params`, `primary_metric`, `feature_importance` (DataFrame),
`shap_importance` (DataFrame or `None`), `ensemble_info` (dict or `None`),
`decision_threshold` (binary only), `reject_threshold` (multiclass only),
`per_label_thresholds` (multilabel only), `objective`,
`multi_threshold_ensemble`, `threshold_ladder`, `dataset_profile`,
`hardware_profile`, `resource_plan`, `timings`, `notes`, `class_labels`.

## 15. Persistence Format

`.smarttab` is a zip archive:

```
model.cbm                 # if CatBoost — native format, stable across versions
model.joblib               # if LightGBM/XGBoost — joblib (tied to the library version at save time)
base_models/<name>.*       # if voting/stacking — each base model saved separately
meta_model.joblib          # stacking only — the meta-learner (LogisticRegression/Ridge)
pipeline.joblib            # the fitted SmartCleaningPipeline
target_encoder.joblib      # classification only
meta.json                  # everything else: hyperparameters, dataset/hardware profiles, metrics,
                           # decision_threshold, reject_threshold, per_label_thresholds, objective,
                           # multi_threshold_ensemble, threshold_ladder, and (for voting) per-model
                           # weights (voting_weights)
```

`smarttab.load(path)` reconstructs the full model — trained estimator(s),
cleaning pipeline, target encoder, dataset/hardware profiles. Feature
importance is recomputed from the loaded model. The result is immediately
usable for `predict()`/`predict_proba()`.

## 16. Errors You Might See

| Exception | When |
|---|---|
| `DataValidationError` | Empty/invalid data, file not found, unsupported file extension, target/group column missing, or a multi-column `target` with mixed types (not all binary, not all continuous) |
| `ConfigurationError` | An invalid parameter value (e.g. `test_size=1.5`, `cv=1`, `ensemble="bad"`, `time_limit=-1`), or `ensemble` set to anything but `"none"` on multilabel/multioutput/ranking |
| `UnsupportedModelError` | `model="xgboost"` (XGBoost is only reachable via `ensemble="voting"/"stacking"/"auto"`) |
| `SmartTabError` | The base exception; also raised by `predict_proba()` on regression/ranking, `evaluate()`/`report()` on ranking without `groups=`, or `report()` called with no evaluation data available |

## 17. What's Not Here Yet

- SHAP for ensembles (voting/stacking) and for multilabel/multi-output
  regression — currently only for single models on
  binary/multiclass/regression/ranking.
- `ensemble=` (voting/stacking/auto) for multilabel/multi-output
  regression/ranking — currently only binary/multiclass/regression.
- Native multi-output losses (e.g. CatBoost's `MultiRMSE`/`MultiLogloss`)
  instead of N independent models for multilabel/multi-output regression —
  the current approach (`MultiOutputClassifier`/`Regressor`) trades some
  speed for being simpler and more stable across both libraries.
- A standalone CLI (`smarttab` from the command line).
- Image, audio, and free-text modeling (out of scope for this library —
  SmartTab is for tabular data).
