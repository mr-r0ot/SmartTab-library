# SmartTab 3.1.0 Validation Report

Validation date: 2026-07-29

## Scope

SmartTab 3.1.0 extends the bounded multimodal 3.0 architecture with train-only data-quality auditing, configurable robust preprocessing, calibrated uncertainty, conformal prediction, out-of-distribution scoring, drift monitoring, missing-modality augmentation, bounded supervised feature adaptation, richer reports, and reproducible CPU benchmarks.

This report describes the exact source tree and release artifacts built in the validation environment. It is not a claim of state-of-the-art accuracy on arbitrary real-world datasets.

## Implemented data-science capabilities

### Data-quality audit

The public `smarttab.audit()` API returns a structured `DataQualityReport` with attribute access, mapping-style access, and `to_dict()` serialization. The audit covers:

- missing, infinite, constant, almost-empty, duplicate, and near-empty rows or columns;
- duplicate samples with conflicting labels;
- target missingness, imbalance, cardinality, and suspicious distributions;
- numeric skew, robust outlier rates, and non-finite values;
- categorical cardinality, rare-category concentration, mixed Python types, and missingness;
- unreadable or missing image, audio, and video references;
- issue severity, affected counts and rates, quality score, and concrete recommendations.

### Train-only preprocessing

The cleaning pipeline fits all learned transformations on the training partition only and replays the same state during evaluation and inference. Added controls include:

- median, mean, constant, KNN, and iterative numeric imputation;
- missing-value indicator features;
- train-fitted rare-category grouping and unknown-category handling;
- robust winsorization;
- automatic or explicit `log1p` and Yeo-Johnson distribution transforms;
- datetime decomposition;
- leakage, constant, ID-like, duplicate, and near-duplicate feature handling;
- supervised mutual-information/variance feature budgeting;
- strict raw-schema validation;
- configurable handling of target missingness, duplicate rows, conflicting labels, and almost-empty rows.

### Uncertainty and monitoring

- Sigmoid and isotonic probability calibration with before/after ECE, log loss, and Brier diagnostics.
- Split-conformal prediction sets for classification.
- Split-conformal intervals for regression.
- OOD scoring over the transformed feature space.
- Raw-column and transformed-feature drift monitoring using PSI, Jensen-Shannon divergence, missing-rate shift, and location shift.
- `predict_with_uncertainty()`, `predict_set()`, `predict_interval()`, `ood_score()`, and `drift_report()` APIs.

### Multimodal robustness and adaptation

- Controlled missing-modality augmentation during training.
- Hard global, per-modality, and per-column feature limits.
- Optional bounded supervised PLS adaptation over extracted classical or frozen pretrained features.
- The adapter is deliberately not described as full end-to-end neural fine-tuning; it is a target-aware bounded projection suitable for CPU operation and small-to-medium datasets.
- Thread-controlled parallel feature extraction and learner fitting.

### Reporting

Generated HTML and JSON reports now contain:

- dataset and target quality summaries;
- issue tables with severity, affected rate, and action;
- numeric, categorical, and media diagnostics;
- every learned cleaning decision and before/after feature/missing counts;
- feature-budget and supervised-adaptation diagnostics;
- calibration, conformal, OOD, and drift summaries;
- evaluation metrics, ensemble diagnostics, feature importance, and optional SHAP status;
- explicit static-chart export status rather than silent failure.

## Automated test results

The release was tested in isolated unit and integration runs to avoid cumulative OpenMP/BLAS learner state contaminating unrelated test processes.

| Suite | Result |
|---|---:|
| Unit tests | 167 passed |
| Integration tests | 84 passed |
| Optional skips | 1 (`pyarrow` not installed) |
| Total passing tests | **251** |

A branch-enabled full coverage run completed with:

- 7,382 executable statements;
- 2,674 branches;
- 80.14% statement coverage;
- 65.22% branch coverage;
- **76.17% combined line/branch coverage**.

After the final whitespace-only formatting cleanup in the trainer, 14 targeted hardware and end-to-end data-science tests were rerun successfully. The final unit suite was also rerun: 167 passed and one optional skip.

## Packaging validation

### Wheel

- File: `smarttab-3.1.0-py3-none-any.whl`
- 60 archive members.
- Metadata name/version: `smarttab` / `3.1.0`.
- Python requirement: `>=3.10,<3.14`.
- Direct `threadpoolctl>=3.2` dependency present.
- Jinja report template included.
- Data-science, robust-cleaning, multimodal-adaptation, persistence, and reporting modules included.
- No `__pycache__`, `.pyc`, or `.pyo` members.

The wheel was installed into a clean target directory outside the source tree with `--no-deps`. The installed package passed the wheel smoke test, covering audit, advanced cleaning, calibration, conformal prediction, OOD scoring, drift reporting, report generation, secure persistence, reload equality, and text processing.

All 53 importable SmartTab submodules were imported from the installed wheel.

A quick CPU tabular benchmark was run against the installed wheel and produced ROC-AUC 0.9978 with 36 final features.

### Source distribution

- File: `smarttab-3.1.0.tar.gz`.
- Built with `setuptools.build_meta`.
- Installed with `--no-build-isolation --no-deps` into a separate target directory.
- Imported as SmartTab 3.1.0 with the public audit and `DataScienceConfig` APIs present.

### Syntax validation

All 89 Python files under `src`, `tests`, and `benchmarks` were parsed successfully with Python's AST parser.

## CPU benchmark results

Each case ran in a fresh process on CPU only, with no model or dataset downloads. Optimization and ensemble construction were disabled to measure the deterministic base pipeline rather than spend the benchmark budget searching.

Reference environment:

- Python 3.13.5;
- Linux x86-64;
- 5 exposed AMD EPYC CPU cores;
- 5.93 GB RAM;
- NumPy 2.3.5;
- pandas 2.2.3;
- scikit-learn 1.8.0.

| Data type | Dataset | Rows | Metric | Result | Fit time | Prediction throughput | Peak RSS increase | Features |
|---|---|---:|---|---:|---:|---:|---:|---:|
| Tabular | Breast Cancer Wisconsin plus controlled quality defects | 569 | ROC-AUC | 0.9907 | 0.712 s | 19,204/s | 38.6 MB | 36 |
| Text | Installed scikit-learn versus pandas source snippets | 420 | ROC-AUC | 0.9887 | 2.527 s | 370/s | 49.7 MB | 160 |
| Image | scikit-learn Digits, classes 0–4 | 800 | Macro F1 | 0.9320 | 2.464 s | 631/s | 42.0 MB | 87 |
| Audio | Deterministic procedural acoustic-event families | 360 | Macro F1 | 1.0000 | 2.034 s | 302/s | 43.4 MB | 123 |
| Video | Moving Digits, classes 0–3 | 180 | Macro F1 | 1.0000 | 6.525 s | 56/s | 40.7 MB | 149 |
| Mixed | Tabular + text + image + audio Mixed Digits | 240 | Macro F1 | 0.9772 | 4.142 s | 184/s | 127.3 MB | 221 |

The five primary cases completed in 31.39 seconds. The mixed case completed in 7.54 seconds in a separate clean process.

The audio and video datasets are deterministic procedural system tests with deliberately learnable structure. Their perfect scores demonstrate pipeline execution, bounded feature extraction, reporting, persistence, and inference; they do not establish general real-world audio or video accuracy. The text task distinguishes two installed codebases and is not a general language-understanding benchmark.

## Environment limitations

- No physical CUDA GPU was available. GPU admission, parameter mapping, and fallback behavior are tested, but real GPU throughput and VRAM behavior remain to be validated on hardware.
- `sentence-transformers` and `timm` were unavailable; no pretrained weights were downloaded. Offline classical and hybrid-fallback paths were validated.
- PyTorch 2.10.0 CPU and torchaudio 2.10.0 CPU were present. The reference benchmark intentionally used CPU classical backends.
- `pyarrow` was unavailable, producing the single optional Parquet/Feather skip.
- Ruff, Mypy, and Twine were unavailable in the local tool mirror. Their commands remain configured in CI/release workflows; package metadata, wheel contents, source-distribution installation, and installed-wheel behavior were validated independently.
- Full end-to-end neural encoder fine-tuning is not implemented. SmartTab 3.1.0 provides frozen pretrained embeddings when dependencies/weights are available and bounded supervised PLS adaptation. This avoids silently turning a CPU-oriented boosting library into an uncontrolled deep-training framework.

## Conclusion

SmartTab 3.1.0 materially strengthens data science correctness, cleaning transparency, uncertainty estimation, monitoring, and reproducible CPU operation. The release artifacts are installable and the supported CPU paths are validated. Real-world accuracy still depends on dataset quality, split design, domain fit, labels, and optional pretrained backends; benchmark results must not be presented as universal or state-of-the-art guarantees.
