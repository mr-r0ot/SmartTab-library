# SmartTab 3.0 API Reference

## Public functions

### `smarttab.fit(data, target=None, group_id=None, *, y=None, modality="auto", modalities=None, **options)`

Unified entry point.

- DataFrame/tabular file: provide `target` or external `y`.
- Direct raw samples: provide `y` and an explicit `modality` when inference would be ambiguous.
- Mixed DataFrame: provide `modalities={column: "text"|"image"|"audio"|"video"|"tabular"}`.

### Direct helpers

```python
fit_text(texts, y, **options)
fit_images(images, y, **options)
fit_audio(audio, y, **options)
fit_videos(videos, y, **options)
fit_folder(folder, modality="image", label_from="parent", extensions=None, **options)
```

### `smarttab.load(path, *, trusted=False)`

Loads a trusted `.smarttab` bundle. `trusted=True` is mandatory before deserialization.

## Core modeling options

| Parameter | Values/default | Purpose |
|---|---|---|
| `task_type` | `auto` | Explicit task override. |
| `model` | `auto`, `catboost`, `lightgbm` | Single learner selection. |
| `ensemble` | `none`, `auto`, `voting`, `stacking` | OOF model combination. |
| `fusion` | `auto`, `early`, `late`, `hybrid` | Full-view versus modality specialists. |
| `optimize` | `True` | Enables baseline-aware Optuna search. |
| `n_trials` | `auto` | Exact trial count when integer. |
| `time_limit` | `0` | Global fit deadline; zero is unlimited. |
| `metrics` | `auto` | Primary internal optimization metric. |
| `params` | `None` | Expert parameters for one explicit/single model. |

## Multimodal options

| Parameter | Default | Purpose |
|---|---:|---|
| `modalities` | `auto` | Column-to-modality mapping. |
| `feature_budget` | `auto` | Global/modality/column generated-feature caps or `FeatureSpaceConfig`. |
| `speed_accuracy` | `0.5` | Continuous automatic resource/quality control. |
| `multimodal_backend` | `auto` | `classical`, `pretrained`, `hybrid`, or automatic. |
| `allow_model_download` | `False` | Permission to obtain pretrained weights. |
| `media_error_policy` | `warn` | `error`, `warn`, or `zero`. |
| `feature_cache` | `False` | Memory cache or persistent cache directory. |
| `batch_size` | `auto` | Deep inference batch size. |
| `feature_workers` | `auto` | Bounded CPU media workers. |
| `modality_params` | `None` | Per-modality/per-column extractor overrides. |

### `FeatureSpaceConfig`

```python
smarttab.FeatureSpaceConfig(
    total_features=512,
    modality_limits={"text": 300},
    column_limits={"review": 240},
    speed_accuracy=0.6,
    backend="hybrid",
    allow_model_download=True,
    error_policy="warn",
    batch_size=24,
    workers=4,
    cache=".cache/smarttab",
    modality_params={...},
    random_state=42,
    device="auto",
)
```

### Selected extractor options

Text:

- `vectorizer`: `auto`, `tfidf`, `hashing`
- `max_chars`
- `max_svd_fit_rows`
- `word_ngram_range`, `char_ngram_range`
- `min_df`, `max_df`, `sublinear_tf`
- `model_name`, `input_mode`, `encoding`

Image:

- `analysis_size`
- `histogram_bins`
- `model_name`

Audio:

- `sample_rate`
- `max_seconds`
- `frame_ms`, `hop_ms`
- `n_mels`, `n_mfcc`
- `model_name`

Video:

- `max_frames`
- `max_fit_frames`
- `analysis_size`
- `model_name`

Place options under a modality or source-column key:

```python
modality_params={
    "text": {"vectorizer": "hashing"},
    "review": {"max_chars": 40_000},
}
```

Column-level options override modality-level options.

## Split and leakage options

| Parameter | Values |
|---|---|
| `split_strategy` | `auto`, `random`, `group`, `stratified_group`, `temporal` |
| `test_size` | float in `(0, 1)` |
| `time_column` | required for temporal split |
| `group_id` | required for ranking/group splits |
| `duplicate_policy` | `drop`, `keep`, `error` |
| `target_missing` | `error`, `drop` |
| `leakage_policy` | `drop`, `error`, `warn`, `ignore` |
| `schema_policy` | `strict`, `coerce` |

## Cleaning options

- `clean`: `auto`, `minimal`, `none`
- `missing`: `auto`, `median`
- `categorical`: `auto`, `native`, `ordinal`
- `scaling`: `auto`, `none`, `standard`, `minmax`, `robust`
- `outlier`: `auto`, `keep`, `remove`
- `feature_selection`: `auto`, `none`

`clean="none"` disables optional removal but retains mandatory schema adaptation, imputation, datetime conversion, modality extraction, and categorical encoding.

## Ensemble controls

- `ensemble_models_limit`: 1–10
- `ensemble_min_gain`: minimum OOF gain
- `diversity_correlation_limit`: maximum tolerated prediction correlation
- `meta_model`: `auto`, `linear`, `catboost`, `lightgbm`
- `xgboost_policy`: `auto`, `never`, `always`

Removed parameters `multi_threshold_ensemble` and `threshold_models` raise `ConfigurationError`.

## Resource controls

- `device`: `auto`, CPU, or supported GPU selector
- `cpu_threads`
- `ram_limit`
- `gpu_memory`
- `feature_workers`
- `batch_size`

RAM/VRAM controls are estimator/admission budgets, not operating-system process hard limits.

## `SmartTabModel`

### Prediction

- `predict(X)`
- `predict_proba(X)` for classification
- `evaluate(X, y, groups=None)`

Single-modality models accept a direct raw sample or a batch. Mixed models require the raw DataFrame schema.

### Introspection

- `feature_space`: fitted budgets, backends, groups, errors, and fusion
- `transform_features(X)`: exact bounded matrix used by final learners
- `metrics`
- `feature_importance`
- `ensemble_info`
- `dataset_profile`
- `hardware_profile`
- `resource_plan`
- `notes`
- `timings`

### Artifacts

- `report(folder=None, X=None, y=None, groups=None)`
- `save(path)`

## Exceptions

- `ConfigurationError`
- `DataValidationError`
- `UnsupportedModelError`
- `TimeLimitExceeded`
- `SmartTabError`


















# Chapter LAST — SmartTab Complete API Reference

## 2.1 Overview

SmartTab provides a high-level API designed around a simple principle:

> Give SmartTab your data. SmartTab builds, evaluates, explains, and saves a machine learning system.

The main public API consists of:

| API | Purpose |
|---|---|
| `fit()` | Train a complete ML pipeline |
| `audit()` | Analyze dataset quality without training |
| `fit_text()` | Train directly from text samples |
| `fit_images()` | Train directly from images |
| `fit_audio()` | Train directly from audio |
| `fit_videos()` | Train directly from videos |
| `fit_folder()` | Train from organized media folders |
| `load()` | Restore a trained model |


---

# 2.2 Main Training API: fit()

## Function Signature

```python
smarttab.fit(
    data,
    target=None,
    group_id=None,
    *,
    y=None,
    modality="auto",
    modalities=None,
    **kwargs
)
````

`fit()` is the core SmartTab function.

It performs the complete workflow:

```
Input Data
    |
    v
Validation
    |
    v
Cleaning
    |
    v
Feature Engineering
    |
    v
Hardware Analysis
    |
    v
Model Selection
    |
    v
Optimization
    |
    v
Training
    |
    v
Evaluation
    |
    v
Explainability
    |
    v
SmartTabModel
```

---

# Parameters

## data

Input dataset.

Supported formats:

| Type             | Example              |
| ---------------- | -------------------- |
| Pandas DataFrame | `pd.DataFrame()`     |
| CSV path         | `"data.csv"`         |
| Parquet path     | `"data.parquet"`     |
| Text samples     | `["hello", "world"]` |
| Images           | image paths / arrays |
| Audio            | WAV / arrays         |
| Video            | video paths          |

Example:

```python
model = smarttab.fit(
    "customer.csv",
    target="churn"
)
```

---

## target

Target column name.

Example:

```python
model = smarttab.fit(
    data,
    target="price"
)
```

Multiple targets are supported:

```python
model = smarttab.fit(
    data,
    target=[
        "price",
        "tax"
    ]
)
```

---

## y

External labels.

Useful when data and labels are separated.

Example:

```python
X = [
    "good product",
    "bad product"
]

y = [
    1,
    0
]


model = smarttab.fit_text(
    X,
    y
)
```

---

## group_id

Defines groups that must stay together during splitting.

Useful for:

* Medical patients
* Users
* Devices
* Sessions

Example:

```python
model = smarttab.fit(
    data,
    target="fraud",
    group_id="customer_id"
)
```

SmartTab prevents:

```
Customer A
    |
    +---- Train

Customer A
    |
    +---- Test
```

which would create data leakage.

---

# 2.3 Dataset Quality System

Before training, SmartTab automatically checks:

## Missing Values

Example:

```
Age:
20
35
NaN
50
```

SmartTab detects:

* Missing percentage
* Important columns affected
* Cleaning strategy

---

## Duplicate Detection

Example:

```
Row 1:
Age=20
Income=5000


Row 2:
Age=20
Income=5000
```

SmartTab can:

* Keep duplicates
* Remove duplicates
* Raise an error

Configuration:

```python
duplicate_policy="drop"
```

---

## Conflicting Label Detection

Detects:

```
Same Features

Input:
Age=25
Income=5000


Labels:

Row 1 -> Fraud

Row 2 -> Not Fraud
```

Configuration:

```python
data_science={
    "conflicting_labels":"drop"
}
```

---

# 2.4 Automatic Model Selection

SmartTab automatically chooses between:

| Algorithm | Best Use Case                 |
| --------- | ----------------------------- |
| CatBoost  | Complex patterns              |
| LightGBM  | Large datasets                |

Example:

```python
model = smarttab.fit(
    data,
    target="target"
)
```

No algorithm selection required.

---

# 2.5 Optimization API

SmartTab supports automatic hyperparameter optimization.

Enable:

```python
model = smarttab.fit(
    data,
    target="target",
    optimize=True
)
```

Control search:

```python
model = smarttab.fit(
    data,
    target="target",
    n_trials=100
)
```

Optimization searches:

* Learning rate
* Tree depth
* Regularization
* Number of estimators
* Algorithm-specific parameters

---

# 2.6 Ensemble System

SmartTab supports automatic ensembles.

Enable:

```python
model = smarttab.fit(
    data,
    target="target",
    ensemble="auto"
)
```

Architecture:

```
             Dataset

                |
                v

       Candidate Models

       CatBoost
       LightGBM

                |
                v

      Performance Analysis

                |
                v

       Diversity Analysis

                |
                v

        Voting / Stacking

                |
                v

          Final Model
```

Supported:

* Voting Ensemble
* Stacking Ensemble
* Automatic ensemble decision engine

---

# 2.7 Hardware Profiling

SmartTab automatically detects:

* CPU cores
* RAM
* GPU availability
* GPU memory

Example output:

```python
model.hardware_profile
```

Example:

```json
{
 "cpu_threads":8,
 "ram":"32GB",
 "gpu_available":true
}
```

---

# 2.8 Explainability API

SmartTab automatically generates explanations.

## Feature Importance

```python
model.feature_importance
```

Example:

```
income       0.42
age          0.21
location     0.13
```

---

## SHAP Explanation

```python
model.shap_importance
```

Provides:

* Global explanation
* Feature contribution
* Model behavior analysis

---

# 2.9 Evaluation System

SmartTab automatically evaluates models.

## Classification Metrics

| Metric    | Meaning                     |
| --------- | --------------------------- |
| Accuracy  | Overall correctness         |
| Precision | Positive prediction quality |
| Recall    | Detection ability           |
| F1        | Precision/Recall balance    |
| ROC-AUC   | Ranking quality             |
| Log Loss  | Probability quality         |

## Regression Metrics

| Metric | Meaning            |
| ------ | ------------------ |
| RMSE   | Prediction error   |
| MAE    | Average error      |
| R²     | Explained variance |

Access:

```python
model.metrics
```

---

# 2.10 Uncertainty System

SmartTab supports reliable predictions.

Components:

## Probability Calibration

Improves prediction probabilities.

Methods:

* Sigmoid calibration
* Isotonic calibration

---

## Conformal Prediction

Provides prediction confidence:

Example:

```
Prediction:

Class: Fraud

Confidence interval:
85%-95%
```

---

## OOD Detection

Detects unfamiliar samples.

Example:

Training:

```
Cars dataset
```

Input:

```
Airplane image
```

SmartTab can identify:

```
Out-of-distribution sample
```

---

# 2.11 Multimodal Training API

## Text

```python
model = smarttab.fit_text(
    texts,
    y
)
```

Example:

```python
texts=[
 "excellent product",
 "terrible service"
]

labels=[
 1,
 0
]

model = fit_text(texts, labels)
```

---

## Images

```python
model = smarttab.fit_images(
    images,
    y
)
```

Supports:

* File paths
* PIL images
* NumPy arrays
* Bytes

---

## Audio

```python
model = smarttab.fit_audio(
    audio,
    y
)
```

Supports:

* WAV files
* Audio arrays
* `(sample_rate, waveform)`

---

## Video

```python
model = smarttab.fit_videos(
    videos,
    y
)
```

---

## Folder Training

Expected structure:

```
dataset/

    cat/
        1.jpg
        2.jpg

    dog/
        3.jpg
        4.jpg
```

Usage:

```python
model = smarttab.fit_folder(
    "dataset",
    modality="image"
)
```

---

# 2.12 Saving and Loading Models

Save:

```python
model.save(
    "model.smarttab"
)
```

Load:

```python
from smarttab import load


model = load(
    "model.smarttab"
)
```

Loaded models keep:

* Pipeline
* Encoder
* Features
* Ensemble information
* Explainability
* Calibration
* Drift reference

---

# 2.13 SmartTabModel Object

The output of training:

```python
SmartTabModel
```

Contains:

| Property           | Description             |
| ------------------ | ----------------------- |
| model_name         | Selected algorithm      |
| metrics            | Evaluation results      |
| feature_importance | Feature impact          |
| dataset_profile    | Dataset information     |
| hardware_profile   | Hardware information    |
| ensemble_info      | Ensemble details        |
| timings            | Performance information |
| notes              | Training decisions      |

---

Methods:

## predict()

```python
model.predict(X)
```

## predict_proba()

```python
model.predict_proba(X)
```

## report()

```python
model.report(
    "report_folder"
)
```

Generates:

```
HTML Report
Charts
Metrics
Explainability
```

---

# 2.14 Complete Example

```python
from smarttab import fit


model = fit(
    "customer_data.csv",
    target="churn",
    optimize=True,
    ensemble="auto",
    explain=True,
    report=True,
    time_limit=300
)


print(model.metrics)


prediction = model.predict(
    new_customers
)
```

Result:

```
Dataset analyzed
        |
        v
Cleaning completed
        |
        v
Best model selected
        |
        v
Optimization completed
        |
        v
Ensemble created
        |
        v
Model explained
        |
        v
Production model ready
```

