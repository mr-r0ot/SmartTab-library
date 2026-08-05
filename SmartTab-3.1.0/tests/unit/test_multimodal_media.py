from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from smarttab.multimodal.audio import AudioFeatureExtractor
from smarttab.multimodal.image import ImageFeatureExtractor
from smarttab.multimodal.video import VideoFeatureExtractor


def test_image_features_capture_entropy_edges_and_respect_budget():
    images = []
    for i in range(8):
        image = np.zeros((24, 32, 3), dtype=np.uint8)
        image[:, ::2, i % 3] = 255
        image += i
        images.append(image)
    extractor = ImageFeatureExtractor(max_features=48, backend="classical")
    frame = extractor.fit_transform(pd.Series(images, dtype=object))

    assert frame.shape == (8, 48)
    assert "gray_entropy" in frame.columns
    assert "edge_density" in frame.columns
    transformed = extractor.transform(pd.Series(images[:2], dtype=object))
    np.testing.assert_allclose(frame.iloc[:2], transformed, rtol=1e-5, atol=1e-6)


def test_audio_features_include_mfcc_chroma_and_spectral_entropy():
    sample_rate = 8000
    audio = []
    for frequency in (180, 220, 330, 440, 550, 660):
        time = np.arange(sample_rate // 3) / sample_rate
        waveform = (0.3 * np.sin(2 * np.pi * frequency * time)).astype(np.float32)
        audio.append((sample_rate, waveform))
    extractor = AudioFeatureExtractor(max_features=72, backend="classical", sample_rate=8000)
    frame = extractor.fit_transform(pd.Series(audio, dtype=object))

    assert frame.shape == (6, 72)
    assert any(name.startswith("mfcc_") for name in frame.columns)
    assert any(name.startswith("chroma_") for name in frame.columns)
    assert any("spectral_entropy" in name for name in frame.columns)


def test_video_features_aggregate_frame_and_motion_information():
    videos = []
    for direction in range(5):
        frames = []
        for index in range(7):
            frame = np.zeros((20, 20, 3), dtype=np.uint8)
            if direction % 2:
                frame[:, index : index + 3, 0] = 255
            else:
                frame[index : index + 3, :, 2] = 255
            frames.append(frame)
        videos.append(np.stack(frames))
    extractor = VideoFeatureExtractor(max_features=64, backend="classical", max_frames=5)
    frame = extractor.fit_transform(pd.Series(videos, dtype=object))

    assert frame.shape == (5, 64)
    assert "motion_mean" in frame.columns
    assert any(name.endswith("_trend") for name in frame.columns)
    assert extractor.frame_extractor_ is not None
    assert extractor.frame_extractor_.max_features < 64


def test_audio_wav_bytes_are_supported():
    import io
    import wave

    sample_rate = 8000
    time = np.arange(sample_rate // 10) / sample_rate
    waveform = (0.3 * np.sin(2 * np.pi * 440 * time) * 32767).astype(np.int16)
    payload = io.BytesIO()
    with wave.open(payload, "wb") as writer:
        writer.setnchannels(1)
        writer.setsampwidth(2)
        writer.setframerate(sample_rate)
        writer.writeframes(waveform.tobytes())

    extractor = AudioFeatureExtractor(max_features=40, backend="classical", sample_rate=8000)
    frame = extractor.fit_transform(pd.Series([payload.getvalue(), payload.getvalue()], dtype=object))
    assert frame.shape == (2, 40)
    assert np.isfinite(frame.to_numpy()).all()


def test_video_accepts_fps_frame_tuple_and_metadata_dictionary():
    frames = []
    for index in range(5):
        frame = np.zeros((18, 20, 3), dtype=np.uint8)
        frame[:, index : index + 2, 1] = 255
        frames.append(frame)
    array = np.stack(frames)
    values = pd.Series(
        [
            (25.0, array),
            {"frames": array, "fps": 10.0, "has_audio": True},
        ],
        dtype=object,
    )
    extractor = VideoFeatureExtractor(max_features=48, backend="classical", max_frames=4)
    result = extractor.fit_transform(values)

    assert result.shape == (2, 48)
    assert result.loc[0, "fps"] == 25.0
    assert result.loc[1, "fps"] == 10.0
    assert result.loc[1, "has_audio"] == 1.0


def test_image_supports_path_bytes_pil_grayscale_and_error_policies(tmp_path):
    import io

    from PIL import Image

    rgb = np.zeros((12, 16, 3), dtype=np.uint8)
    rgb[..., 0] = 180
    pil = Image.fromarray(rgb)
    path = tmp_path / "sample.png"
    pil.save(path)
    payload = io.BytesIO()
    pil.save(payload, format="PNG")
    grayscale = np.arange(12 * 16, dtype=np.uint8).reshape(12, 16)

    extractor = ImageFeatureExtractor(
        max_features=36,
        backend="classical",
        workers=2,
        error_policy="warn",
    )
    frame = extractor.fit_transform(
        pd.Series([path, payload.getvalue(), pil, grayscale, None], dtype=object)
    )
    assert frame.shape == (5, 36)
    assert extractor.errors_
    assert frame.iloc[-1].isna().all()

    strict = ImageFeatureExtractor(max_features=16, backend="classical", error_policy="error")
    with pytest.raises(ValueError, match="could not be decoded"):
        strict.fit_transform(pd.Series([None], dtype=object))


def test_audio_supports_dict_array_file_and_error_fallback(tmp_path):
    import wave

    sample_rate = 4000
    time = np.arange(sample_rate // 8) / sample_rate
    waveform = (0.2 * np.sin(2 * np.pi * 300 * time)).astype(np.float32)
    path = tmp_path / "tone.wav"
    pcm = (waveform * 32767).astype(np.int16)
    with wave.open(str(path), "wb") as writer:
        writer.setnchannels(1)
        writer.setsampwidth(2)
        writer.setframerate(sample_rate)
        writer.writeframes(pcm.tobytes())

    stereo = np.column_stack([waveform, waveform * 0.5])
    extractor = AudioFeatureExtractor(
        max_features=44,
        backend="classical",
        sample_rate=4000,
        workers=2,
        error_policy="warn",
    )
    frame = extractor.fit_transform(
        pd.Series(
            [
                {"samples": waveform, "sr": sample_rate},
                waveform,
                stereo,
                path,
                None,
            ],
            dtype=object,
        )
    )
    assert frame.shape == (5, 44)
    assert extractor.errors_
    assert frame.iloc[-1].isna().all()


@pytest.mark.optional
def test_video_path_and_container_bytes_when_opencv_is_available(tmp_path):
    cv2 = pytest.importorskip("cv2")

    path = tmp_path / "sample.avi"
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"MJPG"),
        8.0,
        (20, 16),
    )
    if not writer.isOpened():
        pytest.skip("OpenCV video writer codec is unavailable")
    for index in range(6):
        frame = np.zeros((16, 20, 3), dtype=np.uint8)
        frame[:, index : index + 3, 2] = 200
        writer.write(frame)
    writer.release()

    payload = path.read_bytes()
    extractor = VideoFeatureExtractor(
        max_features=40,
        backend="classical",
        max_frames=4,
        workers=2,
    )
    result = extractor.fit_transform(pd.Series([path, payload], dtype=object))
    assert result.shape == (2, 40)
    assert (result["sampled_frames"] > 0).all()


def test_hybrid_media_without_download_falls_back_to_classical():
    image = np.zeros((18, 20, 3), dtype=np.uint8)
    image[:, ::2, 1] = 255
    image_extractor = ImageFeatureExtractor(
        max_features=40,
        backend="hybrid",
        allow_model_download=False,
    )
    image_frame = image_extractor.fit_transform(pd.Series([image, image], dtype=object))
    assert image_extractor.backend_used_ == "classical"
    assert image_extractor.embedding_names_ == []
    assert image_frame.shape[1] <= 40

    sample_rate = 4000
    time = np.arange(sample_rate // 8) / sample_rate
    waveform = (0.2 * np.sin(2 * np.pi * 260 * time)).astype(np.float32)
    audio_extractor = AudioFeatureExtractor(
        max_features=44,
        backend="hybrid",
        allow_model_download=False,
        sample_rate=sample_rate,
    )
    audio_frame = audio_extractor.fit_transform(
        pd.Series([(sample_rate, waveform), (sample_rate, waveform)], dtype=object)
    )
    assert audio_extractor.backend_used_ == "classical"
    assert audio_extractor.embedding_names_ == []
    assert audio_frame.shape[1] <= 44

    frames = np.stack([np.roll(image, shift=index, axis=1) for index in range(4)])
    video_extractor = VideoFeatureExtractor(
        max_features=48,
        backend="hybrid",
        allow_model_download=False,
        max_frames=4,
    )
    video_frame = video_extractor.fit_transform(pd.Series([frames, frames], dtype=object))
    assert video_extractor.backend_used_ == "classical"
    assert video_extractor.frame_extractor_ is not None
    assert video_extractor.frame_extractor_.embedding_names_ == []
    assert video_frame.shape[1] <= 48
