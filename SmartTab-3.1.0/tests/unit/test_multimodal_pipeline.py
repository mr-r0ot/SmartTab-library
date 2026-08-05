from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from smarttab.analysis.dataset_analyzer import TaskType
from smarttab.multimodal.config import resolve_feature_space_config
from smarttab.multimodal.detector import resolve_column_modalities
from smarttab.multimodal.pipeline import MultiModalFeaturePipeline


def _images(n: int):
    result = []
    for i in range(n):
        image = np.zeros((16, 16, 3), dtype=np.uint8)
        image[..., i % 3] = 80 + i
        result.append(image)
    return result


def test_mixed_pipeline_enforces_global_and_column_budgets(tmp_path):
    frame = pd.DataFrame({
        "review": [f"review number {i} with useful semantic words" for i in range(30)],
        "photo": _images(30),
    })
    y = pd.Series([i % 2 for i in range(30)])
    config = resolve_feature_space_config(
        {"total": 80, "review": 52, "photo": 28},
        speed_accuracy=0.4,
        backend="classical",
        allow_model_download=False,
        error_policy="warn",
        batch_size="auto",
        cache=str(tmp_path / "cache"),
        modality_params=None,
        random_state=3,
        device="cpu",
    )
    pipeline = MultiModalFeaturePipeline(
        config,
        {"review": "text", "photo": "image"},
        TaskType.BINARY,
    )
    features = pipeline.fit_transform(frame, y)
    replay = pipeline.transform(frame.iloc[:3])

    assert features.shape[1] <= 80
    assert pipeline.report_.allocated_features["review"] == 52
    assert pipeline.report_.allocated_features["photo"] == 28
    assert replay.shape == (3, features.shape[1])
    assert list(replay.columns) == list(features.columns)
    assert any((tmp_path / "cache").glob("*.npy"))


def test_detector_recognizes_array_modalities_and_long_text():
    frame = pd.DataFrame({
        "image": [np.zeros((8, 8, 3), dtype=np.uint8)] * 4,
        "audio": [np.zeros(100, dtype=np.float32)] * 4,
        "video": [np.zeros((3, 8, 8, 3), dtype=np.uint8)] * 4,
        "text": ["long free-form text with several words and details"] * 4,
        "number": [1, 2, 3, 4],
    })
    resolved = resolve_column_modalities(frame, list(frame.columns), "auto")

    assert resolved["image"] == "image"
    assert resolved["audio"] == "audio"
    assert resolved["video"] == "video"
    assert resolved["text"] == "text"
    assert "number" not in resolved


def test_detector_explicit_declarations_and_invalid_inputs(tmp_path):
    from smarttab.exceptions import ConfigurationError
    from smarttab.multimodal.detector import resolve_column_modalities

    frame = pd.DataFrame({"a": ["x", "y"], "b": [1, 2]})
    assert resolve_column_modalities(frame, ["a", "b"], {"a": "text", "b": "tabular"}) == {
        "a": "text"
    }
    with pytest.raises(ConfigurationError, match="unknown feature columns"):
        resolve_column_modalities(frame, ["a", "b"], {"missing": "text"})
    with pytest.raises(ConfigurationError, match="must be one of"):
        resolve_column_modalities(frame, ["a", "b"], {"a": "unknown"})
    with pytest.raises(ConfigurationError, match="exactly one"):
        resolve_column_modalities(frame, ["a", "b"], "text")
    with pytest.raises(ConfigurationError, match="modalities must"):
        resolve_column_modalities(frame, ["a", "b"], ["text"])


def test_detector_recognizes_paths_tuples_lists_pil_and_short_tabular(tmp_path):
    from PIL import Image

    from smarttab.multimodal.detector import infer_series_modality, infer_value_modality

    assert infer_value_modality("picture.PNG") == "image"
    assert infer_value_modality("sound.WAV") == "audio"
    assert infer_value_modality("clip.MP4") == "video"
    assert infer_value_modality("short") == "tabular"
    assert infer_value_modality("a sufficiently long free-form sentence with words") == "text"
    assert infer_value_modality((8000, np.zeros(20, dtype=np.float32))) == "audio"
    assert infer_value_modality((np.zeros(20, dtype=np.float32), 8000)) == "audio"
    assert infer_value_modality([np.zeros((3, 3, 3), dtype=np.uint8)]) == "video"
    assert infer_value_modality(Image.new("RGB", (3, 3))) == "image"
    assert infer_value_modality(object()) == "tabular"
    assert infer_series_modality(pd.Series([None, None], dtype=object)) == "tabular"
    assert infer_series_modality(pd.Series(["a", "b", "c"])) == "tabular"
