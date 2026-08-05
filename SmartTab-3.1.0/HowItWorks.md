# How SmartTab 3.0 Works

SmartTab 3.0 is a leakage-aware multimodal AutoML pipeline. Raw tabular values, text, images, audio, and video are transformed into a bounded feature matrix, then modeled by CatBoost/LightGBM or an OOF fusion ensemble.

## End-to-end stages

1. Normalize DataFrame, file, folder, or direct raw-modality input.
2. Validate target shape, task override, duplicate policy, and missing-target policy.
3. Create an immutable random, temporal, group, or stratified-group outer holdout.
4. Profile only training rows and resolve explicit/conservative modalities.
5. Detect target leakage across numeric, categorical, text, and exact/deterministic mappings.
6. Fit one bounded extractor per raw modality column on training rows only.
7. Apply global feature-budget selection and the standard cleaning pipeline.
8. Profile CPU/RAM/GPU and resolve learner/extractor resource controls.
9. Select CatBoost/LightGBM, or build early/late/hybrid OOF candidates.
10. Run baseline-aware optimization only when enabled.
11. Refit the selected structure on all training rows.
12. Evaluate once on the untouched holdout.
13. Generate report/explanations and persist the exact replayable pipeline.

## Input forms

```python
smarttab.fit(frame, target="label", modalities={"review": "text"})
smarttab.fit_text(texts, y)
smarttab.fit_images(images, y)
smarttab.fit_audio(audio, y)
smarttab.fit_videos(videos, y)
smarttab.fit_folder("root/class/files", modality="image")
```

Direct raw samples are stored internally in a one-column DataFrame, so task handling, splitting, cleaning, reporting, and persistence remain identical to ordinary tabular input.

## Modality resolution

Explicit `modalities={column: modality}` declarations take priority. Auto detection is conservative:

- known file suffixes identify image/audio/video paths;
- array rank/shape identifies raw media in common cases;
- long/high-uniqueness string columns with whitespace are treated as text;
- ambiguous columns remain tabular.

## Feature-space control

`feature_budget` can be:

- `"auto"`;
- a global integer;
- a dictionary of total/modality/column caps;
- a complete `FeatureSpaceConfig` object.

Every extractor obeys a per-column limit. If the combined generated matrix exceeds the global cap, features are selected using train-only variance and task-aware univariate scores. Selected names are persisted and strictly reproduced.

## Classical and pretrained paths

The default backend is local and classical. It does not download models.

- Text: structural/entropy/script features + bounded TF-IDF or hashing + LSA.
- Image: geometry/color/entropy/edge/frequency/spatial descriptors.
- Audio: waveform/spectral/log-mel/MFCC/chroma/entropy/trend descriptors.
- Video: bounded frame sampling + image aggregation + motion/scene-change descriptors.

Optional pretrained encoders are batched and reduced to the same hard budget:

- SentenceTransformers for text;
- timm models for images/video frames;
- torchaudio wav2vec2/HuBERT for audio.

## Speed/accuracy control

`speed_accuracy` influences automatic feature counts, text strategy, reducer effort, image resolution, analyzed audio duration, sampled video frames, and batch size. It never overrides explicit hard caps.

`feature_workers` controls bounded CPU decode/extraction concurrency. `batch_size` controls pretrained inference batches. `device`, `cpu_threads`, `ram_limit`, and `gpu_memory` control final learners and supported deep runtimes.

## Fusion

The cleaning pipeline exposes stable feature groups for all features, raw tabular features, each source column, and each modality.

- `early`: full matrix only;
- `late`: modality/tabular specialists combined from OOF predictions;
- `hybrid`: full-view learners and specialists compete together;
- `auto`: hybrid for multiple useful groups, otherwise early.

Candidate selection, voting weights, and stacking meta-models use OOF predictions only. The outer holdout remains untouched.

## Task engine

The multimodal layer is task-agnostic. The same generated matrix supports binary, multiclass, regression, multilabel, multi-output regression, and ranking. Ambiguous numeric targets should use an explicit `task_type`.

## Persistence

A `.smarttab` bundle stores:

- raw schema and modality declarations;
- fitted vectorizers/reducers/extractors;
- selected feature names and groups;
- cleaning/target encoders;
- CatBoost/LightGBM and ensemble member feature subsets;
- manifest, hashes, runtime versions, metrics, and report metadata.

Pretrained weights are referenced through their normal local cache and are not duplicated inside the bundle. Loading requires `trusted=True` because joblib/pickle-compatible objects are present.
