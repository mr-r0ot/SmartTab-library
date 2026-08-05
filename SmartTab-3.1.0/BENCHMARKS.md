# CPU Benchmarks

SmartTab 3.1.0 includes a reproducible, network-free CPU benchmark harness. Each case runs in a fresh process to isolate OpenMP, BLAS, CatBoost, LightGBM, decoder, allocator, and cache state. Results describe the exact build and machine below; they are not universal latency or accuracy guarantees.

## Reproduce

```bash
python benchmarks/run_cpu_benchmarks.py --output benchmark_results/cpu
python benchmarks/run_cpu_benchmarks.py --cases mixed --output benchmark_results/cpu_mixed
```

Use `--quick` for CI-scale smoke execution. The benchmark fixes common native thread environment variables, uses `device="cpu"`, disables Optuna and ensemble construction, creates a report and model bundle, then measures full-dataset prediction throughput in the same fresh case process.

## Reference environment

- SmartTab 3.1.0
- Python 3.13.5
- Linux 6.12 x86-64
- AMD EPYC 9V74, 5 physical/logical CPUs exposed to the container
- 5.93 GB RAM
- NumPy 2.3.5
- pandas 2.2.3
- scikit-learn 1.8.0
- no network downloads
- CPU-only classical feature backends

## Results

| Case | Dataset | Rows | Learner | Metric | Value | Fit seconds | Prediction samples/s | Peak RSS increase | Final features |
|---|---|---:|---|---|---:|---:|---:|---:|---:|
| Tabular | Breast Cancer Wisconsin Diagnostic plus controlled quality defects | 569 | LightGBM | ROC-AUC | 0.9907 | 0.712 | 19,204.3 | 38.6 MB | 36 |
| Text | Installed scikit-learn versus pandas source snippets | 420 | LightGBM | ROC-AUC | 0.9887 | 2.527 | 370.2 | 49.7 MB | 160 |
| Image | Optical Recognition of Handwritten Digits, classes 0–4 | 800 | LightGBM | macro F1 | 0.9320 | 2.464 | 630.6 | 42.0 MB | 87 |
| Audio | Deterministic procedural acoustic-event families | 360 | LightGBM | macro F1 | 1.0000 | 2.034 | 301.6 | 43.4 MB | 123 |
| Video | Moving Digits, classes 0–3 with class-dependent motion | 180 | LightGBM | macro F1 | 1.0000 | 6.525 | 56.3 | 40.7 MB | 149 |
| Mixed | Tabular + text + image + audio Mixed Digits | 240 | CatBoost | macro F1 | 0.9772 | 4.142 | 183.7 | 127.3 MB | 221 |

The five primary cases completed in 31.39 seconds. The mixed case completed in 7.54 seconds in a separate clean run.

## What these numbers establish

- The installed package can fit, report, persist, reload, and predict on every supported raw-data family using CPU-only bounded features.
- The data-quality pipeline handles injected missing values, outliers, and rare categories in the tabular case.
- Feature growth remains bounded: final matrices ranged from 36 to 221 columns.
- Text, image, audio, video, and mixed decoding/extraction paths complete without a network dependency.

## What these numbers do not establish

- The procedural audio and Moving Digits video cases are system tests with deliberately learnable structure. Their perfect macro F1 must not be interpreted as real-world audio/video quality.
- The text task distinguishes two installed codebases; it is not a general language-understanding benchmark.
- Results use one seed and one holdout. Competitive claims require repeated seeds, confidence intervals, external datasets, and baselines such as raw CatBoost/LightGBM, linear models over embeddings, and task-specific fine-tuned encoders.
- GPU and optional pretrained backends are outside this CPU reference run.

Machine-readable summaries are stored in `benchmarks/reference_results/cpu_2026-07-29.json` and `.csv`.
