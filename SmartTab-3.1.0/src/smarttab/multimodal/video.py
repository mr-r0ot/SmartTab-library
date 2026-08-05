"""Video features built from bounded temporal sampling and image embeddings."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import tempfile
from typing import Any

import numpy as np
import pandas as pd

from smarttab.multimodal.base import BaseFeatureExtractor
from smarttab.multimodal.common import linear_trend, safe_float32_frame, summarize
from smarttab.multimodal.image import ImageFeatureExtractor


class VideoFeatureExtractor(BaseFeatureExtractor):
    modality = "video"

    def __init__(
        self,
        *,
        max_features: int,
        backend: str = "auto",
        speed_accuracy: float = 0.5,
        allow_model_download: bool = False,
        batch_size: int = 16,
        workers: int = 1,
        random_state: int = 42,
        error_policy: str = "warn",
        device: str = "auto",
        model_name: str = "mobilenetv3_small_100",
        max_frames: int | str = "auto",
        analysis_size: int | str = "auto",
        max_fit_frames: int | str = "auto",
        **_: Any,
    ) -> None:
        super().__init__(max_features=max_features, error_policy=error_policy)
        self.backend = backend
        self.speed_accuracy = float(speed_accuracy)
        self.allow_model_download = allow_model_download
        self.batch_size = int(batch_size)
        self.workers = max(1, int(workers))
        self.random_state = int(random_state)
        self.device = device
        self.model_name = model_name
        self.max_frames = int(6 + 26 * self.speed_accuracy if max_frames == "auto" else max_frames)
        self.analysis_size = analysis_size
        self.max_fit_frames = int(256 + 1792 * self.speed_accuracy if max_fit_frames == "auto" else max_fit_frames)
        self.frame_extractor_: ImageFeatureExtractor | None = None
        self.metadata_names_: list[str] = []
        self.aggregate_names_: list[str] = []
        self.backend_used_ = "classical"
        self.notes_: list[str] = []

    def fit_transform(self, values: pd.Series, y: Any = None) -> pd.DataFrame:
        loaded = self._load_batch(values)
        all_frames: list[np.ndarray] = []
        for frames, _ in loaded:
            all_frames.extend(frames)
        if not all_frames:
            all_frames = [np.zeros((32, 32, 3), dtype=np.uint8)]
        elif len(all_frames) > self.max_fit_frames:
            indices = _sample_indices(len(all_frames), self.max_fit_frames)
            all_frames = [all_frames[index] for index in indices]
        metadata_budget = min(18, max(8, self.max_features // 8))
        frame_budget = max(8, (self.max_features - metadata_budget) // 3)
        self.frame_extractor_ = ImageFeatureExtractor(
            max_features=frame_budget,
            backend=self.backend,
            speed_accuracy=self.speed_accuracy,
            allow_model_download=self.allow_model_download,
            batch_size=self.batch_size,
            workers=self.workers,
            random_state=self.random_state,
            error_policy=self.error_policy,
            device=self.device,
            model_name=self.model_name,
            analysis_size=self.analysis_size,
        )
        self.frame_extractor_.fit_transform(pd.Series(all_frames, dtype=object))
        self.backend_used_ = self.frame_extractor_.backend_used_
        matrix, names = self._aggregate_loaded(loaded, values.index)
        matrix = matrix[:, : self.max_features]
        self.feature_names_ = names[: matrix.shape[1]]
        self._fitted = True
        return safe_float32_frame(matrix, self.feature_names_, values.index)

    def transform(self, values: pd.Series) -> pd.DataFrame:
        if not self._fitted or self.frame_extractor_ is None:
            raise RuntimeError("VideoFeatureExtractor must be fitted before transform()")
        loaded = self._load_batch(values)
        matrix, _ = self._aggregate_loaded(loaded, values.index)
        return safe_float32_frame(matrix[:, : len(self.feature_names_)], self.feature_names_, values.index)

    def _load_batch(self, values: pd.Series) -> list[tuple[list[np.ndarray], dict[str, float]]]:
        items = [(value, index) for index, value in values.items()]
        if self.workers > 1 and len(items) > 1:
            with ThreadPoolExecutor(max_workers=min(self.workers, len(items))) as executor:
                return list(executor.map(lambda args: self._safe_load(*args), items))
        return [self._safe_load(value, index) for value, index in items]

    def _safe_load(self, value: Any, index: Any) -> tuple[list[np.ndarray], dict[str, float]]:
        try:
            return self._load_video(value)
        except Exception as exc:
            self._record_error(f"video row {index!r} could not be decoded: {exc}")
            return [], self._empty_metadata()

    def _load_video(self, value: Any) -> tuple[list[np.ndarray], dict[str, float]]:
        if value is None or (isinstance(value, float) and np.isnan(value)):
            raise ValueError("missing video")
        if isinstance(value, dict):
            if "frames" in value:
                frames_value = value["frames"]
                fps = float(value.get("fps", 0.0) or 0.0)
                frames, metadata = self._load_video(frames_value)
                metadata["fps"] = fps if fps > 0 else metadata.get("fps", np.nan)
                if fps > 0:
                    metadata["duration_seconds"] = float(metadata["frame_count"] / fps)
                if "has_audio" in value:
                    metadata["has_audio"] = float(bool(value["has_audio"]))
                return frames, metadata
            for key in ("path", "bytes", "payload"):
                if key in value:
                    frames, metadata = self._load_video(value[key])
                    if "has_audio" in value:
                        metadata["has_audio"] = float(bool(value["has_audio"]))
                    return frames, metadata
            raise ValueError("video dictionary must contain frames, path, bytes, or payload")
        if (
            isinstance(value, tuple)
            and len(value) == 2
            and isinstance(value[0], (int, float))
        ):
            fps = float(value[0])
            frames, metadata = self._load_video(value[1])
            if fps > 0:
                metadata["fps"] = fps
                metadata["duration_seconds"] = float(metadata["frame_count"] / fps)
            return frames, metadata
        if isinstance(value, np.ndarray):
            array = np.asarray(value)
            if array.ndim != 4 or array.shape[-1] not in (1, 3, 4):
                raise ValueError(f"video array must have shape (frames, height, width, channels), got {array.shape}")
            frames = [array[index] for index in _sample_indices(len(array), self.max_frames)]
            h, w = array.shape[1:3]
            return frames, {
                "frame_count": float(len(array)), "sampled_frames": float(len(frames)),
                "fps": np.nan, "duration_seconds": np.nan, "width": float(w), "height": float(h),
                "aspect_ratio": float(w / max(1, h)), "has_audio": np.nan,
            }
        if isinstance(value, (list, tuple)) and value and isinstance(value[0], np.ndarray):
            indices = _sample_indices(len(value), self.max_frames)
            frames = [np.asarray(value[index]) for index in indices]
            h, w = frames[0].shape[:2]
            return frames, {
                "frame_count": float(len(value)), "sampled_frames": float(len(frames)),
                "fps": np.nan, "duration_seconds": np.nan, "width": float(w), "height": float(h),
                "aspect_ratio": float(w / max(1, h)), "has_audio": np.nan,
            }
        if isinstance(value, (str, Path)):
            return self._load_video_path(Path(value).expanduser())
        if isinstance(value, (bytes, bytearray, memoryview)):
            # OpenCV expects a seekable filename for container formats. Keep the
            # temporary file lifetime strictly scoped to decoding.
            temporary_path: Path | None = None
            try:
                with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as handle:
                    handle.write(bytes(value))
                    temporary_path = Path(handle.name)
                return self._load_video_path(temporary_path)
            finally:
                if temporary_path is not None:
                    temporary_path.unlink(missing_ok=True)
        raise TypeError(f"unsupported video value type {type(value).__name__}")

    def _load_video_path(self, path: Path) -> tuple[list[np.ndarray], dict[str, float]]:
        if not path.exists() or not path.is_file():
            raise FileNotFoundError(path)
        try:
            import cv2
        except ImportError as exc:
            raise RuntimeError("video path decoding requires: pip install smarttab[video]") from exc
        capture = cv2.VideoCapture(str(path))
        try:
            if not capture.isOpened():
                raise ValueError(f"OpenCV could not open {path}")
            frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
            width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
            indices = _sample_indices(max(1, frame_count), self.max_frames)
            frames: list[np.ndarray] = []
            for frame_index in indices:
                capture.set(cv2.CAP_PROP_POS_FRAMES, int(frame_index))
                ok, frame = capture.read()
                if ok:
                    frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            if not frames:
                raise ValueError("no decodable video frames")
            return frames, {
                "frame_count": float(frame_count),
                "sampled_frames": float(len(frames)),
                "fps": fps,
                "duration_seconds": float(frame_count / fps) if fps > 0 else np.nan,
                "width": float(width),
                "height": float(height),
                "aspect_ratio": float(width / max(1, height)),
                "has_audio": np.nan,
            }
        finally:
            capture.release()

    def _aggregate_loaded(
        self,
        loaded: list[tuple[list[np.ndarray], dict[str, float]]],
        index: pd.Index,
    ) -> tuple[np.ndarray, list[str]]:
        if self.frame_extractor_ is None:
            raise RuntimeError("frame extractor is unavailable")
        rows: list[list[float]] = []
        names: list[str] | None = None
        for frames, metadata in loaded:
            motion = self._motion_features(frames)
            metadata_all = {**metadata, **motion, "missing_video": float(not frames)}
            if frames:
                frame_features = self.frame_extractor_.transform(pd.Series(frames, dtype=object)).to_numpy()
            else:
                frame_features = np.full((1, len(self.frame_extractor_.feature_names_)), np.nan, dtype=np.float32)
            aggregate: dict[str, float] = {}
            for feature_index, feature_name in enumerate(self.frame_extractor_.feature_names_):
                values = frame_features[:, feature_index]
                aggregate[f"frame_{feature_name}_mean"] = float(np.nanmean(values)) if np.isfinite(values).any() else np.nan
                aggregate[f"frame_{feature_name}_std"] = float(np.nanstd(values)) if np.isfinite(values).any() else np.nan
                aggregate[f"frame_{feature_name}_trend"] = linear_trend(values)
            combined = {**metadata_all, **aggregate}
            if names is None:
                names = list(combined)
                self.metadata_names_ = list(metadata_all)
                self.aggregate_names_ = list(aggregate)
            rows.append([combined[name] for name in names])
        names = names or list(self._empty_metadata())
        return np.asarray(rows, dtype=np.float32), names

    @staticmethod
    def _motion_features(frames: list[np.ndarray]) -> dict[str, float]:
        if len(frames) < 2:
            return {
                "motion_mean": 0.0, "motion_std": 0.0, "motion_max": 0.0,
                "scene_cut_ratio": 0.0, "motion_trend": 0.0,
            }
        gray_frames = []
        for frame in frames:
            array = np.asarray(frame, dtype=np.float32)
            if array.ndim == 3:
                array = array[..., :3].mean(axis=2)
            max_value = float(np.nanmax(array)) if array.size else 1.0
            if max_value > 1.5:
                array = array / 255.0
            # Fixed coarse grid makes motion independent of source resolution.
            y_idx = np.linspace(0, array.shape[0] - 1, 48).astype(int)
            x_idx = np.linspace(0, array.shape[1] - 1, 48).astype(int)
            gray_frames.append(array[np.ix_(y_idx, x_idx)])
        differences = np.asarray([
            np.mean(np.abs(right - left)) for left, right in zip(gray_frames[:-1], gray_frames[1:], strict=True)
        ])
        return {
            "motion_mean": float(differences.mean()),
            "motion_std": float(differences.std()),
            "motion_max": float(differences.max()),
            "scene_cut_ratio": float((differences > max(0.25, np.quantile(differences, 0.9))).mean()),
            "motion_trend": linear_trend(differences),
        }

    @staticmethod
    def _empty_metadata() -> dict[str, float]:
        return {
            "frame_count": np.nan, "sampled_frames": 0.0, "fps": np.nan,
            "duration_seconds": np.nan, "width": np.nan, "height": np.nan,
            "aspect_ratio": np.nan, "has_audio": np.nan,
            "motion_mean": np.nan, "motion_std": np.nan, "motion_max": np.nan,
            "scene_cut_ratio": np.nan, "motion_trend": np.nan, "missing_video": 1.0,
        }


def _sample_indices(frame_count: int, max_frames: int) -> list[int]:
    if frame_count <= max_frames:
        return list(range(frame_count))
    return np.unique(np.linspace(0, frame_count - 1, max_frames).round().astype(int)).tolist()
