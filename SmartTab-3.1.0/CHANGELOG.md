# Changelog

All notable changes are documented here.

## 3.1.0 — 2026-07-29

### Added

- Public `smarttab.audit()` and `DataScienceConfig` for data-quality policy without model fitting.
- Structured quality findings for nearly empty rows/columns, infinities, skew, IQR outliers, high-cardinality and rare categories, mixed value types, missing/unreadable media, target health, and contradictory duplicate labels.
- Train-fitted missing indicators, configurable numeric imputation, rare-category grouping, winsorization, log1p/Yeo-Johnson transforms, richer datetime expansion, and supervised global feature-budget selection.
- Probability calibration, conformal classification sets/regression intervals, robust transformed-feature OOD scoring, and raw/transformed drift monitoring.
- Public `predict_set`, `predict_interval`, `predict_with_uncertainty`, `ood_score`, `data_quality`, and `drift_report` model methods.
- Bounded missing-modality dropout augmentation for mixed training data.
- Bounded task-aware PLS adapters over classical or frozen pretrained modality features.
- Detailed data-science, cleaning, adaptation, uncertainty, and drift sections in HTML/JSON reports.
- Reproducible process-isolated CPU benchmark harness and reference results for tabular, text, image, audio, video, and mixed data.

### Changed

- Bundle format advanced to version 5 to persist quality, uncertainty, OOD, drift, and task-adapter state.
- CPU profiling no longer invokes `py-cpuinfo`; platform files and bounded native probes are cached to avoid repeated subprocess/thread-runtime stalls.
- Native BLAS/OpenMP thread pools are bounded with `threadpoolctl` around every learner fit, preventing cumulative oversubscription across repeated CatBoost/LightGBM workloads.
- Explicit voting/stacking diversity selection now prefers a second core algorithm before retaining same-family variants.

### Fixed

- All-NaN modality features causing unstable supervised-adapter fitting.
- Constant features generating noisy supervised ranking warnings.
- Deprecated categorical dtype checks and assignment warnings in preprocessing.
- Audio benchmark envelope exponentiation producing invalid values.
- Cross-case native runtime state contaminating CPU benchmark measurements.

## 3.0.0 — 2026-07-26

### Breaking

- Expanded SmartTab from tabular-only AutoML to a bounded multimodal pipeline for text, image, audio, and video.
- Bundle format advanced to version 4 to persist modality extractors and ensemble feature subsets.
- Pretrained backends require explicit model-download permission and optional dependencies.

### Added

- Unified `fit()` support for mixed DataFrames and direct raw-modality samples.
- `fit_text`, `fit_images`, `fit_audio`, `fit_videos`, and `fit_folder` helpers.
- Explicit/conservative modality detection and persisted raw modality schema.
- `FeatureSpaceConfig` with global, per-modality, and per-column hard limits.
- Continuous `speed_accuracy`, extractor workers, batching, caching, and modality-specific parameters.
- Rich text entropy/script/lexical/structural descriptors, bounded TF-IDF or hashing, LSA, and optional multilingual embeddings.
- Image geometry/color/entropy/edge/sharpness/symmetry/frequency/orientation/spatial descriptors and optional timm embeddings.
- Audio waveform/spectral/log-mel/MFCC/chroma/entropy/trend descriptors and optional wav2vec2/HuBERT embeddings.
- Video temporal sampling, frame aggregation, motion/scene-change features, container bytes, FPS metadata, and optional frame embeddings.
- Early, late, hybrid, and automatic multimodal fusion.
- OOF tabular/modality/source specialists with persisted feature subsets.
- `model.feature_space` and `model.transform_features()` introspection.
- Bounded processing for huge text files, long waveforms, high-resolution images, and long videos.
- Full task routing across binary, multiclass, regression, multilabel, multi-output regression, and ranking.

### Fixed

- Explicit raw media paths being misinterpreted as tabular input files.
- Direct audio/video metadata dictionaries and `(fps, frames)` tuples being misinterpreted as tabular rows/batches.
- Media-cache collisions caused by truncated array/tuple `repr` values.
- Forced ensemble reports omitting the resolved fusion strategy.
- Large text strings causing unsafe path probes or unbounded document processing.

## 2.0.0 — 2026-07-26

### Breaking

- Removed `multi_threshold_ensemble` and `threshold_models`.
- `smarttab.load()` now requires `trusted=True` before deserializing a bundle.
- Prediction input uses the raw training schema and rejects malformed inputs by default.
- Numeric targets with more than two unique values are conservatively inferred as regression; use `task_type="multiclass"` for numeric class labels.
- Minimum Python version is 3.10 and minimum pandas version is 2.0.

### Added

- Explicit task override and random, temporal, group, and stratified-group outer splits.
- Train-only profiling and train-only outlier removal.
- Cross-type exact-copy and deterministic target leakage detection.
- Strict/coercive prediction schema policies.
- Native categorical representation for CatBoost and LightGBM.
- Baseline-aware, budgeted optimization with exact trial accounting.
- OOF voting and stacking with full-data base-model refit.
- Conditional XGBoost inclusion based on incremental OOF gain.
- PR AUC, Brier score, log loss, and expected calibration error.
- Versioned, hash-checked, safely extracted `.smarttab` bundles.
- Explicit static-chart export status.
- Wheel/sdist build, CI matrix, wheel smoke test, and trusted PyPI publishing workflows.
- Estimator-level RAM/VRAM admission budgets mapped to CatBoost and LightGBM controls.
- Early validation for unknown LightGBM expert parameters and managed-parameter collisions.
- Automatic explainability mode: SHAP is attempted only when a report is requested.
- Optional automatic report output directory.
- Original-label decoding for string-valued multilabel targets.

### Fixed

- Jinja report template missing from built wheels.
- `clean` and several configuration values being accepted without meaningful validation.
- Full-dataset profiling and outlier removal leaking holdout information.
- Text/categorical target copies not being identified as leakage.
- Missing prediction columns silently becoming zero-filled features.
- ndarray prediction using transformed rather than raw feature names.
- inconsistent input handling between `predict()` and `evaluate()`.
- `optimize=False` entering optimization in ensemble paths.
- explicit `n_trials` being silently increased.
- unfair single-versus-ensemble score comparison.
- ensemble base learners not being refit on all training rows.
- report JSON differing from the returned object.
- static chart failures being swallowed.
- unsafe archive extraction and unacknowledged joblib trust boundary.
- repeated native CPU probing that could stall cumulative fit workloads after learner thread pools were initialized.
- multilabel predictions returning internal binary codes instead of the original target labels.

## 0.1.0

Initial alpha implementation.
