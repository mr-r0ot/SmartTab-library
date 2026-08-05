from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from smarttab.multimodal.base import BaseFeatureExtractor
from smarttab.multimodal.common import (
    bounded_int,
    entropy,
    linear_trend,
    pad_or_trim_features,
    safe_float32_frame,
    stable_signature,
    summarize,
)


def test_entropy_and_summary_cover_empty_integer_and_float_inputs():
    assert entropy(np.array([])) == 0.0
    assert entropy(np.array([0, 0, 1, 1], dtype=np.uint8)) == pytest.approx(1.0)
    assert entropy(np.array([0.1, 0.2, 0.3, np.nan]), bins=3) >= 0.0
    assert summarize(np.array([np.nan]), "x") == {
        "x_mean": 0.0,
        "x_std": 0.0,
        "x_min": 0.0,
        "x_max": 0.0,
        "x_q25": 0.0,
        "x_q50": 0.0,
        "x_q75": 0.0,
    }
    result = summarize(np.array([1.0, 2.0, 3.0]), "v")
    assert result["v_mean"] == 2.0
    assert result["v_q50"] == 2.0


def test_safe_frame_pad_trim_trend_and_bounds():
    frame = safe_float32_frame(
        np.array([[1.0, np.inf], [np.nan, 2.0]]),
        ["a", "b"],
        pd.Index([3, 4]),
    )
    assert frame.dtypes.tolist() == [np.dtype("float32"), np.dtype("float32")]
    assert np.isnan(frame.loc[3, "b"])
    np.testing.assert_array_equal(pad_or_trim_features(np.array([1, 2, 3]), 2), [1, 2])
    padded = pad_or_trim_features(np.array([1]), 3)
    assert padded[0] == 1 and np.isnan(padded[1:]).all()
    assert linear_trend(np.array([1.0])) == 0.0
    assert linear_trend(np.array([np.nan, np.nan])) == 0.0
    assert linear_trend(np.array([1.0, 2.0, 3.0])) == pytest.approx(1.0)
    assert bounded_int(2.9, 0, 2) == 2
    assert bounded_int(-4, 0, 2) == 0


def test_stable_signature_is_content_aware_for_all_media_containers(tmp_path):
    array_a = np.arange(12, dtype=np.float32).reshape(3, 4)
    array_b = array_a.copy()
    array_b[-1, -1] += 1
    assert stable_signature(array_a) != stable_signature(array_b)
    assert stable_signature(b"abc") != stable_signature(b"abd")
    assert stable_signature((8000, array_a)) == stable_signature((8000, array_a.copy()))
    assert stable_signature({"waveform": array_a, "sr": 8000}) == stable_signature(
        {"sr": 8000, "waveform": array_a.copy()}
    )
    assert stable_signature([array_a, b"x"]) != stable_signature([array_b, b"x"])
    assert stable_signature(None) == stable_signature(None)

    path = tmp_path / "payload.bin"
    path.write_bytes(b"first")
    first = stable_signature(path)
    path.write_bytes(b"second payload")
    second = stable_signature(path)
    assert first != second
    assert stable_signature("ordinary text") == stable_signature("ordinary text")

    image = Image.fromarray(np.zeros((4, 4, 3), dtype=np.uint8))
    assert stable_signature(image) == stable_signature(image.copy())


class _DummyExtractor(BaseFeatureExtractor):
    modality = "dummy"

    def fit_transform(self, values, y=None):
        return pd.DataFrame(index=values.index)

    def transform(self, values):
        return pd.DataFrame(index=values.index)


def test_base_error_policies():
    warning = _DummyExtractor(max_features=4, error_policy="warn")
    warning._record_error("problem")
    assert warning.errors_ == ["problem"]

    zero = _DummyExtractor(max_features=4, error_policy="zero")
    zero._record_error("ignored")
    assert zero.errors_ == []

    strict = _DummyExtractor(max_features=4, error_policy="error")
    with pytest.raises(ValueError, match="fatal"):
        strict._record_error("fatal")
