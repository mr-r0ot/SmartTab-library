# Ensemble and Multimodal Fusion Contract

SmartTab ensembles use CatBoost and LightGBM as core learners. XGBoost is an optional numeric diversity candidate and is never required.

## Invariants

- Every candidate is evaluated through out-of-fold predictions.
- The meta-model is trained only on OOF features.
- The outer holdout is never used for candidate construction, pruning, weights, stacking, thresholds, or stopping.
- `ensemble_models_limit` is a maximum, not a target.
- Correlated prediction streams are pruned unless measurable incremental gain remains.
- All retained members are refit on all training rows.
- Every modality specialist persists its exact feature subset.
- `ensemble="auto"` returns one model when added complexity does not justify itself.

## Candidate families

The engine may create:

- CatBoost and LightGBM full-view anchors;
- bounded variants specialized for MCC, F1, recall, precision, calibration, regularization, RMSE, or MAE;
- tabular-only specialists;
- text/image/audio/video specialists;
- source-column specialists when useful;
- one conditional XGBoost diversity candidate for compatible numeric matrices.

Optuna is used only for anchor search. Specialist variants derive from bounded anchor configurations so search cost does not multiply by every member.

## Fusion modes

### Early

All candidates see the complete feature matrix. This is the lowest-complexity path and is appropriate when one modality dominates or the dataset is small.

### Late

Candidates are built around tabular and modality-specific feature subsets. OOF predictions, not raw features, are combined. One full-view safety anchor may remain.

### Hybrid

Full-view and specialist candidates compete in one OOF selection pool. This supports interactions through full-view learners while preserving strong modality-specific experts.

### Auto

Uses hybrid fusion when at least two useful groups exist; otherwise early fusion.

## Selection objective

For classification, candidate ordering prioritizes:

1. MCC;
2. F1 within a small MCC tolerance;
3. recall within an F1 tolerance;
4. lower log loss.

For regression:

1. lower RMSE;
2. lower MAE;
3. higher R².

Pairwise prediction correlation and incremental ensemble gain are considered before retaining another member.

## Voting

Weighted soft-voting coefficients are optimized from OOF predictions. Equal weights are evaluated but not assumed. Weights are normalized and persisted.

## Stacking

The meta-model receives only OOF predictions. `meta_model="auto"` benchmarks compact linear, CatBoost, and LightGBM combiners under the shared budget. The chosen combiner is refit on all OOF rows.

## Final refit

After structure selection:

1. every retained base candidate is reconstructed with its selected parameters;
2. each candidate is fit on all training rows using only its persisted feature subset;
3. voting weights or the stacking model are attached;
4. the ensemble is evaluated exactly once on the untouched outer holdout.

## Audit data

Reports and bundles record:

- strategy and fusion mode;
- candidate algorithm, specialization, feature group, feature count, and parameters;
- OOF score and timing;
- retained members and feature subsets;
- voting weights or meta-model;
- prediction-correlation matrix;
- single-versus-ensemble gain and fallback reason.
