"""Explicit feature-space and extractor controls."""

import smarttab

space = smarttab.FeatureSpaceConfig(
    total_features=512,
    modality_limits={"text": 320, "image": 192},
    column_limits={"review": 280, "photo": 160},
    speed_accuracy=0.75,
    backend="classical",  # switch to hybrid with deep extras and download permission
    allow_model_download=False,
    batch_size=24,
    workers=4,
    cache=".smarttab-feature-cache",
    modality_params={
        "text": {
            "vectorizer": "hashing",
            "max_chars": 60_000,
            "max_svd_fit_rows": 30_000,
        },
        "image": {
            "analysis_size": 192,
            "histogram_bins": 12,
        },
        "audio": {
            "sample_rate": 16_000,
            "max_seconds": 24,
            "n_mfcc": 20,
        },
        "video": {
            "max_frames": 18,
            "max_fit_frames": 1_000,
        },
    },
)

# frame must contain the declared raw columns and target.
# model = smarttab.fit(
#     frame,
#     target="label",
#     modalities={"review": "text", "photo": "image"},
#     feature_budget=space,
#     ensemble="auto",
#     fusion="hybrid",
# )
