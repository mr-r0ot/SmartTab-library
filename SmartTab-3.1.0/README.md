# SmartTab

SmartTab is a bounded multimodal AutoML library for **tabular data, raw text, images, audio, and video**. It keeps the beginner API deliberately small while exposing explicit controls for feature-space size, modality encoders, CPU/GPU execution, fusion, optimization, validation, reporting, and persistence.

The final predictive learners are **CatBoost** and **LightGBM**. XGBoost remains an optional diversity candidate inside the ensemble engine and is never a core requirement.

## Design goals

- One API for tabular, single-modality, and mixed-modality datasets.
- No unbounded TF-IDF vocabulary, frame expansion, spectrogram expansion, or embedding dimension.
- Useful CPU-only defaults with no model downloads.
- Optional pretrained embeddings when their dependencies and weights are explicitly enabled.
- Train-only feature fitting and leakage-safe outer evaluation.
- Real OOF voting/stacking rather than threshold copies.
- Reproducible feature transforms and versioned model bundles.

## Installation

```bash
python -m pip install smarttab
```

Core installation supports:

- tabular data;
- raw text;
- images from paths, Pillow images, bytes, and NumPy arrays;
- WAV audio from paths, bytes, arrays, and `(sample_rate, waveform)` tuples;
- videos supplied as frame arrays/lists.

Optional extras:

```bash
python -m pip install "smarttab[audio]"          # additional audio formats via soundfile
python -m pip install "smarttab[video]"          # video container decoding via OpenCV
python -m pip install "smarttab[multimodal]"     # audio + video decoders
python -m pip install "smarttab[text-deep]"       # SentenceTransformer text embeddings
python -m pip install "smarttab[vision-deep]"     # timm image/frame embeddings
python -m pip install "smarttab[audio-deep]"      # torchaudio speech embeddings
python -m pip install "smarttab[multimodal-deep]" # all multimodal deep backends
python -m pip install "smarttab[shap]"            # SHAP explanations
python -m pip install "smarttab[report-static]"   # PNG chart export
python -m pip install "smarttab[all]"             # non-deep optional integrations
```

Python 3.10–3.13 is supported.

## Simplest APIs

### Raw text

```python
import smarttab

model = smarttab.fit_text(
    ["excellent product", "broken on arrival", "works reliably", "poor quality"],
    [1, 0, 1, 0],
)

label = model.predict("reliable and well built").single
probability = model.predict_proba("reliable and well built")
```

Text samples may be strings, UTF-8 bytes, or local text-file paths.

### Images

```python
model = smarttab.fit_images(image_paths, labels)
predictions = model.predict(["new/a.jpg", "new/b.jpg"])
```

Images may be paths, Pillow objects, encoded bytes, grayscale/RGB/RGBA arrays, or batches of arrays.

### Audio

```python
model = smarttab.fit_audio(wav_paths, labels)
model.predict((16_000, waveform))
model.predict({"sample_rate": 16_000, "waveform": waveform})
```

### Video

```python
model = smarttab.fit_videos(video_paths, labels)
model.predict((30.0, frame_array))
model.predict({"frames": frame_array, "fps": 30.0, "has_audio": True})
```

A frame array has shape `(frames, height, width, channels)`. Container bytes and local paths are supported when the video decoder extra is installed. Video visual features are intentionally separated from soundtrack features; use an additional audio column when both streams matter.

### Folder classification

Expected layout:

```text
dataset/
├── class_a/
│   ├── 001.jpg
│   └── 002.jpg
└── class_b/
    ├── 003.jpg
    └── 004.jpg
```

```python
model = smarttab.fit_folder("dataset", modality="image")
```

The same helper supports `modality="audio"` and `modality="video"`.

## Mixed multimodal data

```python
import pandas as pd
import smarttab

frame = pd.DataFrame(
    {
        "age": ages,
        "country": countries,
        "review": reviews,
        "photo": image_paths,
        "voice": audio_paths,
        "label": labels,
    }
)

model = smarttab.fit(
    frame,
    target="label",
    modalities={
        "review": "text",
        "photo": "image",
        "voice": "audio",
    },
    ensemble="auto",
    fusion="auto",
)
```

`modalities="auto"` performs conservative detection. Explicit declarations are preferred for short text, array-valued data, or ambiguous object columns.

## Supported tasks

The same feature pipeline feeds the existing task engine:

| Task | `task_type` | Output |
|---|---|---|
| Binary classification | `binary` | one label per sample |
| Multiclass classification | `multiclass` | one label per sample |
| Regression | `regression` | one numeric value |
| Multilabel classification | `multilabel` | one label per target column |
| Multi-output regression | `multioutput_regression` | one value per target column |
| Learning to rank | `ranking` | one ranking score per row |

Numeric targets with more than two unique values are conservatively inferred as regression. Numeric class IDs should set `task_type="multiclass"` explicitly.

## Bounded feature space

Feature generation has three independent limits:

1. a global multimodal feature cap;
2. per-modality caps;
3. per-source-column caps.

```python
model = smarttab.fit(
    frame,
    target="label",
    modalities={"review": "text", "photo": "image"},
    feature_budget={
        "total": 512,
        "text": 320,
        "image": 192,
        "review": 256,
        "photo": 160,
    },
)
```

Generated features are ranked on training data only. When the total exceeds the hard cap, SmartTab performs bounded variance/supervised selection and replays the selected feature schema at prediction time.

For a fully explicit configuration:

```python
space = smarttab.FeatureSpaceConfig(
    total_features=768,
    modality_limits={"text": 384, "image": 256, "audio": 192},
    column_limits={"review": 320, "thumbnail": 128},
    speed_accuracy=0.75,
    backend="hybrid",
    allow_model_download=True,
    batch_size=32,
    workers=6,
    cache=".smarttab-feature-cache",
    device="auto",
    modality_params={
        "text": {
            "vectorizer": "hashing",
            "max_chars": 80_000,
            "model_name": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        },
        "image": {"analysis_size": 192, "model_name": "mobilenetv3_small_100"},
        "audio": {"max_seconds": 30, "n_mfcc": 20},
        "video": {"max_frames": 20, "max_fit_frames": 1200},
    },
)

model = smarttab.fit(frame, target="label", feature_budget=space)
```

`model.feature_space` exposes the fitted budgets, generated dimensions, backend actually used, modality groups, extraction errors, and fusion strategy. `model.transform_features(X)` returns the exact matrix consumed by CatBoost/LightGBM.

## Feature families

### Text

Classical text features include:

- character/token entropy;
- script ratios for Latin, Arabic/Persian, Cyrillic, and CJK;
- lexical diversity, hapax ratio, word-length distribution, sentence/line structure;
- URL, email, punctuation, symbol, digit, repetition, and case statistics;
- bounded word and character TF-IDF or stateless hashing;
- bounded LSA projections;
- optional multilingual SentenceTransformer embeddings with dimensionality reduction.

Very long documents are bounded by sampling the beginning, middle, and end. Large corpora can use hashing and fit LSA on a deterministic bounded sample.

### Image

Classical image features include:

- geometry, aspect ratio, alpha coverage;
- color/channel statistics and histograms;
- entropy, brightness, contrast, saturation, colorfulness;
- gradients, edge density, orientation histogram/entropy;
- Laplacian sharpness, symmetry, center-border contrast, blockiness;
- frequency-domain summaries and spatial-grid statistics;
- optional timm embeddings reduced to the allocated budget.

### Audio

Classical audio features include:

- duration, channels, RMS, peak, crest factor, silence and clipping ratios;
- amplitude entropy, zero-crossing rate, skewness, kurtosis;
- spectral centroid, bandwidth, rolloff, flatness, and spectral entropy;
- bounded band-energy, log-mel, MFCC, chroma, segment, and temporal-trend summaries;
- optional wav2vec2/HuBERT embeddings reduced to the allocated budget.

Long waveforms preserve beginning, middle, and end instead of blindly truncating the tail.

### Video

Video features use bounded temporal sampling:

- frame count, FPS, duration, dimensions, and aspect ratio;
- image features aggregated by mean, standard deviation, and temporal trend;
- coarse motion magnitude, motion trend, and scene-cut ratio;
- optional pretrained image embeddings on sampled frames.

The number of decoded frames, frames used to fit reducers, image resolution, and final output dimension are independently bounded.

## Speed versus accuracy

`speed_accuracy` is a continuous control from `0.0` to `1.0`:

```python
fast = smarttab.fit_text(texts, y, speed_accuracy=0.1, optimize=False)
accurate = smarttab.fit_text(
    texts,
    y,
    speed_accuracy=0.9,
    multimodal_backend="hybrid",
    allow_model_download=True,
    optimize=True,
    ensemble="auto",
)
```

It influences automatic feature budget, text/vectorizer strategy, analysis resolution, audio duration, video frame count, reducer iterations, and batch size. Hard limits remain explicit and override automatic allocation.

CPU extraction uses bounded worker pools through `feature_workers`. Pretrained inference is batched and uses CUDA when requested and available. CatBoost/LightGBM resource controls remain independent through `device`, `cpu_threads`, `ram_limit`, and `gpu_memory`.

## Multimodal fusion

| Mode | Behavior |
|---|---|
| `early` | One learner sees the complete bounded feature matrix. |
| `late` | Modality/tabular specialists are emphasized and combined from OOF predictions. |
| `hybrid` | Full-view learners and modality specialists compete together. |
| `auto` | Uses hybrid fusion when multiple useful feature groups exist; otherwise early fusion. |

```python
model = smarttab.fit(
    frame,
    target="label",
    ensemble="auto",
    fusion="hybrid",
    ensemble_models_limit=5,
    ensemble_min_gain=0.002,
    diversity_correlation_limit=0.98,
    meta_model="auto",
)
```

Every ensemble candidate is evaluated through out-of-fold predictions. Voting weights and stacking meta-models are learned only from OOF data. Retained base learners are refit on all training rows. Added complexity is rejected when it does not improve the common objective.

## Optimization

`optimize=False` runs zero Optuna trials and fits deterministic defaults.

`optimize=True`:

- evaluates a baseline first;
- searches compact high-impact CatBoost/LightGBM spaces;
- respects explicit `n_trials` exactly;
- uses early stopping and bounded search rows;
- retains tuned parameters only when they beat the baseline;
- shares one wall-clock deadline with feature extraction, ensemble construction, final fitting, evaluation, and optional reporting.

## Production schema and prediction

`schema_policy="strict"` rejects missing/extra columns, malformed arrays, invalid numeric/datetime values, and empty inputs. Single-modality models accept direct raw samples at prediction time.

```python
model.predict(raw_sample)
model.predict([raw_sample_a, raw_sample_b])
model.predict(dataframe_batch)
```

Use `schema_policy="coerce"` only when missing tabular columns should be inserted and processed by fitted imputers.

## Data science, cleaning, and validation

Run a quality audit without fitting a model:

```python
quality = smarttab.audit(
    frame,
    target="label",
    modalities={"review": "text", "photo": "image"},
)

print(quality.quality_score)
for issue in quality.issues:
    print(issue.severity, issue.message, issue.recommended_action)
```

Automatic mode enables train-only quality checks, missing indicators, median imputation, rare-category grouping, bounded outlier clipping, skew-aware transforms, calibration, conformal uncertainty, OOD scoring, drift references, and missing-modality robustness where applicable.

```python
policy = smarttab.DataScienceConfig(
    quality_policy="strict",
    conflicting_labels="error",
    numeric_imputation="iterative",
    rare_category_min_frequency=0.005,
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

model = smarttab.fit(frame, target="label", data_science=policy)
```

Production diagnostics:

```python
prediction = model.predict_with_uncertainty(new_rows)
sets = model.predict_set(new_rows)          # classification
lower, upper = model.predict_interval(new_rows)  # regression
scores = model.ood_score(new_rows)
drift = model.drift_report(new_rows)
```

For multimodal sources, `supervised_adaptation="auto"` appends a bounded target-aware PLS projection over classical or frozen pretrained features. It is a CPU-compatible task adapter, not unrestricted end-to-end encoder fine-tuning.

See `DATASCIENCE_DESIGN.md` for the complete preprocessing order, leakage boundary, policy definitions, uncertainty assumptions, and reporting contract.

## Reporting

Reports contain:

- quality score, actionable issues, target health, numeric/categorical/media summaries;
- train-only duplicates, conflicting labels, leakage findings, imputation, rare grouping, clipping, distribution transforms, datetime expansion, and feature-budget actions;
- modality detection, allocated/generated/selected dimensions, backend and extraction errors;
- model/ensemble/fusion candidates and OOF selection evidence;
- holdout metrics, confusion/ROC/PR or regression diagnostics;
- Brier score, log loss, calibration error, conformal coverage/set size, OOD rates, drift diagnostics, MCC, balanced accuracy, and task-specific metrics;
- feature importance and optional SHAP;
- timing, resource budgets, and static-export status.

```python
report = model.report("reports/run-001")
```

## Persistence and trust boundary

```python
model.save("model.smarttab")
loaded = smarttab.load("model.smarttab", trusted=True)
```

A bundle stores the fitted feature extractors, selected dimensions, cleaning schema, ensemble member feature subsets, model payloads, runtime versions, manifest, file sizes, and SHA-256 hashes. Pretrained model weights are not embedded; they must remain available through their normal local cache when a persisted deep model is used.

`.smarttab` bundles contain joblib/pickle-compatible payloads. Never load an untrusted bundle.

## CPU reference benchmark

The bundled process-isolated benchmark covers every supported data family without network downloads:

| Data | Rows | Metric | Value | Fit time | Final features |
|---|---:|---|---:|---:|---:|
| Tabular | 569 | ROC-AUC | 0.9907 | 0.712 s | 36 |
| Text | 420 | ROC-AUC | 0.9887 | 2.527 s | 160 |
| Image | 800 | macro F1 | 0.9320 | 2.464 s | 87 |
| Audio | 360 | macro F1 | 1.0000 | 2.034 s | 123 |
| Video | 180 | macro F1 | 1.0000 | 6.525 s | 149 |
| Mixed | 240 | macro F1 | 0.9772 | 4.142 s | 221 |

Audio and video use deterministic procedural datasets and validate the system path, not general real-world accuracy. Full methodology, environment, limitations, and machine-readable results are in `BENCHMARKS.md` and `benchmarks/reference_results/`.

## Status

Version `3.1.0` is a beta release. Data-quality, uncertainty, drift, CPU classical paths, task routing, feature budgets, persistence, reporting, and installed-wheel operation are tested. Deep/GPU behavior still depends on the exact PyTorch, CUDA, model-weight, codec, and driver environment and requires deployment-specific validation.

## Development

```bash
git clone https://github.com/mr-r0ot/SmartTab-library.git
cd SmartTab-library
python -m pip install -e ".[dev]"
python -m ruff check src tests
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -p pytest_cov.plugin --cov=smarttab
python -m build
python -m twine check dist/*
```

See `DATASCIENCE_DESIGN.md`, `BENCHMARKS.md`, `MULTIMODAL_DESIGN.md`, `ENSEMBLE_DESIGN.md`, `HowItWorks.md`, `documents.md`, `CONTRIBUTING.md`, `SECURITY.md`, and `RELEASING.md`.

## License

MIT License.
