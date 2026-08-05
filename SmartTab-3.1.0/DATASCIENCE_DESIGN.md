# Data Science, Cleaning, Validation, and Monitoring Design

SmartTab 3.1.0 treats data quality as part of the fitted model contract. Analysis that can affect training is fitted only on the outer training partition and is replayed unchanged at prediction time. The untouched holdout remains an evaluation surface, not a source of cleaning decisions.

## Public controls

Beginners can use `data_science="auto"`. Specialists can pass `DataScienceConfig` or a dictionary.

```python
import smarttab

policy = smarttab.DataScienceConfig(
    quality_policy="strict",
    conflicting_labels="error",
    row_missing_threshold=0.90,
    numeric_imputation="iterative",
    add_missing_indicators=True,
    rare_category_min_frequency=0.005,
    rare_category_max_categories=96,
    numeric_transform="auto",
    winsorize=0.01,
    max_model_features=512,
    calibration="isotonic",
    conformal=True,
    conformal_alpha=0.05,
    ood_detection=True,
    drift_monitoring=True,
    modality_dropout=0.10,
)

model = smarttab.fit(data, target="label", data_science=policy)
```

A pre-fit audit is also available without training:

```python
quality = smarttab.audit(
    data,
    target="label",
    modalities={"review": "text", "photo": "image"},
)
print(quality.quality_score)
print(quality.recommendations)
quality_dict = quality.to_dict()
```

## Audit coverage

The audit records:

- duplicate rows, including rows containing arrays or media payloads;
- duplicate feature patterns with contradictory labels;
- missing targets, nearly empty rows, nearly empty columns, and per-column missing rates;
- infinities, robust quantiles, skew, zero rate, and IQR outlier rate;
- categorical cardinality, singleton categories, rare-row share, and dominant-category share;
- mixed Python value types;
- empty media values and sampled unreadable file paths;
- target missingness, cardinality, and bounded class distributions;
- actionable issues with severity, affected columns, counts/rates, and recommended remediation.

The quality score is a compact triage signal. It is not a statistical guarantee and never replaces inspecting the issue list.

## Train-only preprocessing order

1. Validate the raw schema and target policy.
2. Create the immutable outer split using random, temporal, group, or stratified-group logic.
3. Audit the training partition and apply strict/error policies.
4. Remove training rows above the configured missingness threshold.
5. Resolve duplicate rows and contradictory labels according to policy.
6. Detect target leakage, identifiers, constants, duplicate columns, and near-duplicate numeric features.
7. Extract bounded text/image/audio/video features on training data.
8. Add missing indicators.
9. Apply train-fitted winsorization when enabled.
10. Impute numeric and categorical values.
11. Group rare/unseen categorical values using fitted frequency rules.
12. Encode categories and expand datetimes.
13. Apply train-fitted log1p or Yeo-Johnson transformations to strongly skewed numeric columns.
14. Apply optional scaling.
15. Enforce the global model-feature budget with train-only supervised ranking.
16. Fit CatBoost/LightGBM, calibration, conformal uncertainty, OOD reference, and drift reference.

No holdout statistic controls these transformations.

## Imputation and categorical robustness

Numeric imputation supports median, mean, constant, KNN, and iterative strategies. Missing-indicator columns preserve informative absence. Rare-category grouping prevents uncontrolled cardinality and gives unseen production categories a deterministic destination. Prediction schema policy remains strict by default; `schema_policy="coerce"` must be selected explicitly when absent tabular columns should be inserted and processed by fitted imputers.

## Distribution control

Winsorization is train-fitted and bounded. Automatic transformation examines training skew and selects log1p only where the support permits it; otherwise Yeo-Johnson is available. These controls reduce pathological tails without deleting evaluation rows or rewriting target values.

## Multimodal task adaptation

Raw modality extractors produce bounded classical and optional pretrained features. `supervised_adaptation="auto"` can append a small Partial Least Squares projection fitted against the training target, then reapply the local and global feature caps. This is a task-aware adapter over frozen features, not unrestricted end-to-end encoder training. It preserves CatBoost/LightGBM as the final learner and remains usable on CPU.

Because the adapter uses labels, the outer holdout is the authoritative unbiased evaluation. Benchmark and report metrics are always computed on that untouched partition.

## Missing-modality robustness

For mixed datasets, bounded modality-dropout augmentation duplicates only a capped fraction of training rows while masking selected modality groups. This teaches the final learner to use remaining modalities rather than collapsing when one source is absent. The exact expansion and affected groups are stored in the model bundle and report.

## Calibration and conformal outputs

Classification probabilities can be calibrated by sigmoid or isotonic regression on a dedicated training-side calibration split. Conformal prediction produces finite-sample classification sets or regression intervals at the configured alpha. Public methods include:

```python
model.predict_with_uncertainty(X)
model.predict_set(X)          # classification
model.predict_interval(X)     # regression
```

Coverage assumes exchangeability. Severe drift can invalidate that assumption; the report states this limitation explicitly.

## OOD and drift monitoring

The OOD detector uses robust distances, training quantile ranges, and missingness over the transformed model matrix. Drift monitoring stores raw numeric quantile bins, raw categorical frequencies, and transformed-feature references. `model.drift_report(X)` reports PSI or Jensen-Shannon divergence, missingness/location shifts, per-column severity, transformed-feature counts, and an overall score.

```python
scores = model.ood_score(new_rows)
drift = model.drift_report(new_rows)
```

These are diagnostic controls, not proof that a shifted prediction remains correct.

## Reporting contract

`report.json` is the complete machine-readable artifact. `report.html` exposes:

- quality score, issue table, remediation list, target audit;
- numeric, categorical, and raw-modality summaries;
- before/after missingness and feature counts;
- every imputed, indicated, grouped, clipped, transformed, dropped, or budget-pruned feature;
- multimodal allocation and task-aware adapter diagnostics;
- probability calibration, conformal coverage/set size, and OOD rates;
- raw/transformed drift details;
- holdout quality, model metrics, feature importance, timing, hardware budgets, and ensemble evidence.

## Operational boundaries

- A clean quality score does not prove labels are semantically correct.
- Statistical drift does not identify the business cause of drift.
- KNN/iterative imputation can be materially slower and should be benchmarked on large data.
- End-to-end fine-tuning of large modality encoders is deliberately not automatic; it conflicts with bounded CPU operation and the CatBoost/LightGBM final-learner contract. Pretrained embeddings and bounded supervised adapters are the supported task-adaptation path.
- Domain-specific validation remains mandatory for medical, financial, safety-critical, and regulated use.
