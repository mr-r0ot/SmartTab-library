# Multimodal Design Contract

SmartTab converts raw modalities into a bounded, replayable, dense `float32` feature matrix before CatBoost/LightGBM training. The multimodal layer does not bypass the existing leakage, split, optimization, evaluation, reporting, or persistence boundaries.

## Invariants

1. Every feature extractor is fitted on training rows only.
2. The outer holdout never participates in vectorizer vocabulary, SVD/PCA, feature ranking, model selection, voting weights, stacking, threshold selection, or calibration.
3. Generated multimodal dimensions never exceed `FeatureSpaceConfig.total_features`.
4. Per-modality and per-column budgets are upper bounds, not requested minimums.
5. Prediction reproduces the exact fitted feature names and order.
6. Raw source columns are never silently replaced with arbitrary zero vectors.
7. Default classical execution performs no network download.
8. Pretrained backends are opt-in and fall back according to `media_error_policy`.
9. All generated values are finite or explicit missing values handled by fitted imputers.
10. Ensemble specialists receive only their persisted feature subset.

## Pipeline

```text
raw DataFrame / direct raw samples
        │
        ├── conservative modality detection or explicit declarations
        │
        ├── per-column budget allocation
        │
        ├── fitted modality extractor
        │      ├── classical bounded descriptors
        │      ├── optional pretrained embeddings
        │      └── fitted SVD/PCA reduction
        │
        ├── global train-only feature ranking if over budget
        │
        ├── standard SmartTab imputation/encoding/scaling
        │
        ├── early / late / hybrid feature groups
        │
        └── CatBoost / LightGBM / OOF fusion
```

## Budget resolution

For `feature_budget="auto"`:

```text
total = 256 + round(768 × speed_accuracy)
```

The automatic modality allocation uses normalized weights:

- text: 0.40
- image: 0.24
- audio: 0.18
- video: 0.18

Explicit modality and column limits override the proportional allocation. The global cap remains final authority. Raw tabular columns are not counted in the generated multimodal budget because CatBoost/LightGBM consume them directly without one-hot expansion.

## Global ranking

If generated dimensions exceed the global cap, each feature receives:

- log-scaled variance score;
- optional train-only univariate classification/regression score across all target columns.

The highest-scoring features are retained with stable tie/order behavior. The resulting names are persisted and enforced at transform time.

## Text extractor

### Classical path

- 34 language-independent structural/statistical descriptors;
- bounded word and character TF-IDF, or stateless hashing for large/speed-focused corpora;
- LSA with bounded output dimension;
- deterministic bounded-row SVD fitting for very large corpora.

### Input bounding

`max_chars` limits each document. Oversized content preserves beginning, middle, and end. File input is read with a bounded byte budget instead of loading the complete file.

### Deep path

SentenceTransformer embeddings are normalized, batched, and optionally reduced with PCA. Deep model objects are excluded from pickle state and reloaded from the normal model cache after bundle loading.

## Image extractor

The classical descriptor combines geometry, alpha, color distributions, HSV, entropy, gradient statistics, orientation histogram, sharpness, symmetry, compression/blockiness proxy, frequency summaries, and spatial grids. Analysis uses a bounded thumbnail independent of original resolution.

The deep path uses a `timm` feature model with global pooling and a fitted PCA cap.

## Audio extractor

Audio is converted to mono floating-point data and resampled to the configured rate. Long signals retain beginning, middle, and end. Classical descriptors include waveform, temporal, spectral, mel, MFCC, chroma, band-energy, entropy, silence, clipping, and trend statistics.

The deep path batches padded waveforms, propagates valid lengths, pools only valid encoded frames, and reduces embeddings to the assigned budget.

## Video extractor

Video decoding samples a bounded set of temporal positions. A separate bounded image extractor is fitted on at most `max_fit_frames`. Frame features are aggregated using mean, standard deviation, and temporal trend. Motion and scene-change descriptors are computed on a fixed coarse grid.

Video soundtrack is deliberately modeled as a separate audio source column. This keeps codec handling explicit and allows visual/audio specialists to be independently selected during late or hybrid fusion.

## Backends

- `classical`: deterministic local features only.
- `pretrained`: prioritize pretrained embeddings, retaining required structural descriptors.
- `hybrid`: combine classical descriptors and pretrained embeddings within one cap.
- `auto`: classical by default; deep execution is selected only at high `speed_accuracy`, with explicit download permission and installed dependencies.

## Error policies

- `error`: stop on the first undecodable sample.
- `warn`: record bounded error details and emit missing features for fitted imputation.
- `zero`: continue without retaining per-row error details.

Errors and fallback notes are exposed through `model.feature_space` and reports.

## Cache semantics

Feature caching applies to image, audio, and video rows. Cache keys include extractor type, feature cap, model name, and a content-aware signature:

- file path + size + modification time;
- complete byte payload hash;
- array shape, dtype, and bytes;
- recursively hashed tuple/list/dictionary content.

Only generated row vectors are cached. Fitted reducers/vectorizers remain part of the model pipeline.

## Fusion groups

The cleaning pipeline emits stable groups:

- `all`
- `tabular`
- `source:<column>`
- `modality:text`
- `modality:image`
- `modality:audio`
- `modality:video`

Late/hybrid ensemble members persist their exact column list. Prediction validates that every required member feature exists.

## Memory and complexity bounds

Let:

- `N` = samples;
- `B` = global generated feature budget;
- `Fv` = sampled frames per video;
- `S` = analyzed audio samples per item.

The final generated matrix is bounded by `O(N × B)` and stored as `float32`. Video frame extraction is bounded by `O(N × Fv)`. Audio signal work is bounded by configured duration/sample rate. Text lexical intermediate dimensions are bounded by vocabulary/hash caps and the final LSA dimension.

These are library-level algorithmic bounds. Decoder implementations and pretrained runtimes can allocate additional native memory; deployment resource validation remains required.
