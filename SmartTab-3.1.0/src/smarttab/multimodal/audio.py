"""Audio feature extraction with bounded signal descriptors and optional neural embeddings."""

from __future__ import annotations

import importlib.util
from concurrent.futures import ThreadPoolExecutor
import io
import math
import wave
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.fft import dct
from scipy.signal import resample_poly
from scipy.stats import kurtosis, skew
from sklearn.decomposition import PCA

from smarttab.multimodal.base import BaseFeatureExtractor
from smarttab.multimodal.common import entropy, linear_trend, safe_float32_frame, summarize


class AudioFeatureExtractor(BaseFeatureExtractor):
    modality = "audio"

    def __init__(
        self,
        *,
        max_features: int,
        backend: str = "auto",
        speed_accuracy: float = 0.5,
        allow_model_download: bool = False,
        batch_size: int = 8,
        workers: int = 1,
        random_state: int = 42,
        error_policy: str = "warn",
        device: str = "auto",
        model_name: str = "wav2vec2_base",
        sample_rate: int | str = "auto",
        max_seconds: float | str = "auto",
        frame_ms: float = 25.0,
        hop_ms: float = 10.0,
        n_mels: int = 24,
        n_mfcc: int = 13,
        **_: Any,
    ) -> None:
        super().__init__(max_features=max_features, error_policy=error_policy)
        self.backend = backend
        self.speed_accuracy = float(speed_accuracy)
        self.allow_model_download = allow_model_download
        self.batch_size = max(1, min(16, int(batch_size)))
        self.workers = max(1, int(workers))
        self.random_state = int(random_state)
        self.device = device
        self.model_name = model_name
        self.target_sample_rate = int(16000 if sample_rate == "auto" else sample_rate)
        self.max_seconds = float(12 + 48 * self.speed_accuracy if max_seconds == "auto" else max_seconds)
        self.frame_ms = float(frame_ms)
        self.hop_ms = float(hop_ms)
        self.n_mels = max(8, int(n_mels))
        self.n_mfcc = max(4, int(n_mfcc))

        self.classical_names_: list[str] = []
        self.classical_indices_: list[int] = []
        self.embedding_names_: list[str] = []
        self.embedding_reducer_: PCA | None = None
        self.backend_used_ = "classical"
        self.notes_: list[str] = []
        self._deep_model: Any = None
        self._deep_device = "cpu"

    def fit_transform(self, values: pd.Series, y: Any = None) -> pd.DataFrame:
        classical_matrix, classical_names = self._classical_batch(values)
        classical_budget = min(classical_matrix.shape[1], self.max_features)
        use_deep = self._should_use_embeddings() and self.max_features - classical_budget >= 8
        if use_deep:
            classical_budget = min(classical_matrix.shape[1], max(20, int(self.max_features * 0.45)))
        ordered_names = self._diverse_feature_order(classical_names)
        name_to_index = {name: index for index, name in enumerate(classical_names)}
        self.classical_names_ = ordered_names[:classical_budget]
        self.classical_indices_ = [name_to_index[name] for name in self.classical_names_]
        parts = [classical_matrix[:, self.classical_indices_]]
        names = list(self.classical_names_)
        deep_budget = self.max_features - classical_budget
        if use_deep and deep_budget:
            embeddings = self._encode_embeddings(values)
            if embeddings is not None:
                components = min(deep_budget, embeddings.shape[1], max(1, embeddings.shape[0] - 1))
                if components >= 2 and components < embeddings.shape[1]:
                    self.embedding_reducer_ = PCA(
                        n_components=components,
                        svd_solver="randomized",
                        random_state=self.random_state,
                    )
                    embeddings = self.embedding_reducer_.fit_transform(embeddings)
                else:
                    embeddings = embeddings[:, :components]
                self.embedding_names_ = [f"embedding_{i:04d}" for i in range(embeddings.shape[1])]
                parts.append(embeddings)
                names.extend(self.embedding_names_)
                self.backend_used_ = "hybrid"
            else:
                self.notes_.append("pretrained audio embeddings unavailable; classical features used")
        matrix = np.column_stack(parts)[:, : self.max_features]
        self.feature_names_ = names[: matrix.shape[1]]
        self._fitted = True
        return safe_float32_frame(matrix, self.feature_names_, values.index)

    def transform(self, values: pd.Series) -> pd.DataFrame:
        if not self._fitted:
            raise RuntimeError("AudioFeatureExtractor must be fitted before transform()")
        classical_matrix, _ = self._classical_batch(values)
        parts = [classical_matrix[:, self.classical_indices_]]
        if self.embedding_names_:
            embeddings = self._encode_embeddings(values)
            if embeddings is None:
                raise RuntimeError(f"pretrained audio encoder {self.model_name!r} is unavailable")
            if self.embedding_reducer_ is not None:
                embeddings = self.embedding_reducer_.transform(embeddings)
            parts.append(embeddings[:, : len(self.embedding_names_)])
        matrix = np.column_stack(parts)[:, : len(self.feature_names_)]
        return safe_float32_frame(matrix, self.feature_names_, values.index)

    @staticmethod
    def _diverse_feature_order(names: list[str]) -> list[str]:
        essential = [
            name for name in (
                "original_duration_seconds", "analyzed_duration_seconds", "sample_rate", "channels",
                "wave_rms", "wave_peak", "zero_crossing_rate", "silence_ratio", "clipping_ratio",
                "amplitude_entropy", "spectral_centroid_mean", "spectral_bandwidth_mean",
                "spectral_rolloff_mean", "spectral_flatness_mean", "spectral_entropy_mean",
                "mfcc_00_mean", "chroma_00_mean",
            ) if name in names
        ]
        groups = [
            [name for name in names if name in {
                "original_duration_seconds", "analyzed_duration_seconds", "sample_rate", "channels",
                "wave_mean", "wave_std", "wave_rms", "wave_peak", "crest_factor",
                "zero_crossing_rate", "silence_ratio", "clipping_ratio", "amplitude_entropy", "rms_trend",
            } and name not in essential],
            [name for name in names if name.startswith("mfcc_")],
            [name for name in names if name.startswith("chroma_")],
            [name for name in names if name.startswith("band_energy_")],
            [name for name in names if name.startswith("spectral_")],
            [name for name in names if name.startswith("frame_")],
            [name for name in names if name.startswith("segment_")],
        ]
        ordered: list[str] = list(essential)
        positions = [0] * len(groups)
        while True:
            added = False
            for index, group in enumerate(groups):
                if positions[index] < len(group):
                    ordered.append(group[positions[index]])
                    positions[index] += 1
                    added = True
            if not added:
                break
        ordered.extend(name for name in names if name not in set(ordered))
        deduplicated: list[str] = []
        seen: set[str] = set()
        for name in ordered:
            if name not in seen:
                deduplicated.append(name)
                seen.add(name)
        return deduplicated

    def _classical_batch(self, values: pd.Series) -> tuple[np.ndarray, list[str]]:
        items = list(values.items())
        if self.workers > 1 and len(items) > 1:
            with ThreadPoolExecutor(max_workers=min(self.workers, len(items))) as executor:
                feature_rows = list(executor.map(self._classical_row, items))
        else:
            feature_rows = [self._classical_row(item) for item in items]
        names = list(feature_rows[0]) if feature_rows else list(self._empty_feature_dict())
        rows = [[features[name] for name in names] for features in feature_rows]
        return np.asarray(rows, dtype=np.float32), names

    def _classical_row(self, item: tuple[Any, Any]) -> dict[str, float]:
        index, value = item
        try:
            waveform, sample_rate, channels, original_duration = self._load_audio(value)
            return self._classical_features(waveform, sample_rate, channels, original_duration)
        except Exception as exc:
            self._record_error(f"audio row {index!r} could not be decoded: {exc}")
            return self._empty_feature_dict()

    def _load_audio(self, value: Any) -> tuple[np.ndarray, int, int, float]:
        if value is None or (isinstance(value, float) and np.isnan(value)):
            raise ValueError("missing audio")
        sample_rate = self.target_sample_rate
        channels = 1
        if isinstance(value, dict):
            waveform = np.asarray(value.get("waveform", value.get("samples")))
            sample_rate = int(value.get("sample_rate", value.get("sr", sample_rate)))
        elif isinstance(value, tuple) and len(value) == 2:
            first, second = value
            if isinstance(first, (int, float)):
                sample_rate, waveform = int(first), np.asarray(second)
            elif isinstance(second, (int, float)):
                waveform, sample_rate = np.asarray(first), int(second)
            else:
                raise ValueError("audio tuple must be (sample_rate, waveform) or (waveform, sample_rate)")
        elif isinstance(value, np.ndarray):
            waveform = np.asarray(value)
        elif isinstance(value, (str, Path)):
            path = Path(value).expanduser()
            if not path.exists() or not path.is_file():
                raise FileNotFoundError(path)
            waveform, sample_rate = _read_audio_file(path)
        elif isinstance(value, (bytes, bytearray, memoryview)):
            waveform, sample_rate = _read_audio_bytes(bytes(value))
        else:
            raise TypeError(f"unsupported audio value type {type(value).__name__}")

        waveform = np.asarray(waveform, dtype=np.float32)
        if waveform.ndim == 2:
            if waveform.shape[0] <= 8 and waveform.shape[0] < waveform.shape[1]:
                channels = waveform.shape[0]
                waveform = waveform.mean(axis=0)
            else:
                channels = waveform.shape[1]
                waveform = waveform.mean(axis=1)
        elif waveform.ndim != 1:
            raise ValueError(f"unsupported waveform shape {waveform.shape}")
        if waveform.size == 0:
            raise ValueError("empty waveform")
        waveform = np.nan_to_num(waveform, nan=0.0, posinf=0.0, neginf=0.0)
        maximum = float(np.max(np.abs(waveform)))
        if maximum > 1.5:
            waveform = waveform / max(maximum, 1e-9)
        original_duration = float(len(waveform) / max(1, sample_rate))
        if sample_rate != self.target_sample_rate:
            divisor = math.gcd(sample_rate, self.target_sample_rate)
            waveform = resample_poly(
                waveform,
                self.target_sample_rate // divisor,
                sample_rate // divisor,
            ).astype(np.float32)
            sample_rate = self.target_sample_rate
        max_samples = max(1, int(self.max_seconds * sample_rate))
        if len(waveform) > max_samples:
            # Preserve beginning, middle, and end rather than truncating only the tail.
            third = max_samples // 3
            middle_start = max(0, len(waveform) // 2 - third // 2)
            waveform = np.concatenate([
                waveform[:third],
                waveform[middle_start : middle_start + third],
                waveform[-(max_samples - 2 * third) :],
            ])
        return waveform.astype(np.float32, copy=False), sample_rate, channels, original_duration

    def _classical_features(
        self,
        waveform: np.ndarray,
        sample_rate: int,
        channels: int,
        original_duration: float,
    ) -> dict[str, float]:
        frame_length = max(64, int(sample_rate * self.frame_ms / 1000.0))
        hop_length = max(32, int(sample_rate * self.hop_ms / 1000.0))
        frames = _frame_signal(waveform, frame_length, hop_length)
        windowed = frames * np.hanning(frame_length)[None, :]
        magnitude = np.abs(np.fft.rfft(windowed, axis=1)) + 1e-9
        power = magnitude**2
        frequencies = np.fft.rfftfreq(frame_length, 1.0 / sample_rate)
        power_sum = power.sum(axis=1) + 1e-12
        centroid = (power * frequencies[None, :]).sum(axis=1) / power_sum
        bandwidth = np.sqrt(
            (power * (frequencies[None, :] - centroid[:, None]) ** 2).sum(axis=1) / power_sum
        )
        cumulative = np.cumsum(power, axis=1)
        rolloff_indices = np.argmax(cumulative >= 0.85 * cumulative[:, -1:], axis=1)
        rolloff = frequencies[rolloff_indices]
        flatness = np.exp(np.mean(np.log(magnitude), axis=1)) / np.mean(magnitude, axis=1)
        normalized_power = power / power_sum[:, None]
        spectral_entropy = -(normalized_power * np.log2(normalized_power + 1e-12)).sum(axis=1)
        rms = np.sqrt(np.mean(frames**2, axis=1))
        zcr = np.mean(np.diff(np.signbit(frames), axis=1), axis=1)
        peak = np.max(np.abs(waveform))
        features: dict[str, float] = {
            "original_duration_seconds": original_duration,
            "analyzed_duration_seconds": float(len(waveform) / sample_rate),
            "sample_rate": float(sample_rate),
            "channels": float(channels),
            "wave_mean": float(waveform.mean()),
            "wave_std": float(waveform.std()),
            "wave_rms": float(np.sqrt(np.mean(waveform**2))),
            "wave_peak": float(peak),
            "wave_skew": float(skew(waveform, bias=False)) if len(waveform) > 2 else 0.0,
            "wave_kurtosis": float(kurtosis(waveform, bias=False)) if len(waveform) > 3 else 0.0,
            "crest_factor": float(peak / max(1e-9, np.sqrt(np.mean(waveform**2)))),
            "zero_crossing_rate": float(np.mean(np.diff(np.signbit(waveform)))),
            "silence_ratio": float((np.abs(waveform) < 0.01).mean()),
            "clipping_ratio": float((np.abs(waveform) > 0.995).mean()),
            "amplitude_entropy": entropy(waveform, bins=64),
            "rms_trend": linear_trend(rms),
        }
        for prefix, values in (
            ("frame_rms", rms),
            ("frame_zcr", zcr),
            ("spectral_centroid", centroid),
            ("spectral_bandwidth", bandwidth),
            ("spectral_rolloff", rolloff),
            ("spectral_flatness", flatness),
            ("spectral_entropy", spectral_entropy),
        ):
            features.update(summarize(values, prefix))

        nyquist = sample_rate / 2.0
        bands = [(0, 250), (250, 500), (500, 1000), (1000, 2000), (2000, 4000), (4000, 8000), (8000, nyquist + 1)]
        mean_power = power.mean(axis=0)
        total_power = mean_power.sum() + 1e-12
        for index, (low, high) in enumerate(bands):
            mask = (frequencies >= low) & (frequencies < min(high, nyquist + 1))
            features[f"band_energy_{index:02d}"] = float(mean_power[mask].sum() / total_power) if mask.any() else 0.0

        mel_filters = _mel_filterbank(sample_rate, frame_length, self.n_mels)
        mel_energy = np.maximum(power @ mel_filters.T, 1e-10)
        log_mel = np.log(mel_energy)
        mfcc = dct(log_mel, type=2, axis=1, norm="ortho")[:, : self.n_mfcc]
        for index in range(mfcc.shape[1]):
            features[f"mfcc_{index:02d}_mean"] = float(mfcc[:, index].mean())
            features[f"mfcc_{index:02d}_std"] = float(mfcc[:, index].std())

        chroma = np.zeros((power.shape[0], 12), dtype=np.float64)
        valid = frequencies > 27.5
        midi = np.zeros_like(frequencies)
        midi[valid] = 69 + 12 * np.log2(frequencies[valid] / 440.0)
        pitch_classes = np.mod(np.rint(midi).astype(int), 12)
        for pitch in range(12):
            mask = valid & (pitch_classes == pitch)
            if mask.any():
                chroma[:, pitch] = power[:, mask].sum(axis=1)
        chroma /= chroma.sum(axis=1, keepdims=True) + 1e-12
        for pitch in range(12):
            features[f"chroma_{pitch:02d}_mean"] = float(chroma[:, pitch].mean())

        segments = np.array_split(waveform, 8)
        segment_rms = np.asarray([np.sqrt(np.mean(segment**2)) if len(segment) else 0.0 for segment in segments])
        for index, value in enumerate(segment_rms):
            features[f"segment_rms_{index:02d}"] = float(value)
        features["segment_rms_trend"] = linear_trend(segment_rms)
        return features

    def _empty_feature_dict(self) -> dict[str, float]:
        silence = np.zeros(self.target_sample_rate, dtype=np.float32)
        return {name: np.nan for name in self._classical_features(silence, self.target_sample_rate, 1, 1.0)}

    def _should_use_embeddings(self) -> bool:
        if self.backend == "classical":
            return False
        if self.backend in {"pretrained", "hybrid"}:
            return True
        return bool(
            self.speed_accuracy >= 0.8
            and self.allow_model_download
            and importlib.util.find_spec("torchaudio") is not None
        )

    def _load_deep_model(self) -> bool:
        if self._deep_model is not None:
            return True
        if not self.allow_model_download or importlib.util.find_spec("torchaudio") is None:
            return False
        try:
            import torch
            import torchaudio

            bundles = {
                "wav2vec2_base": torchaudio.pipelines.WAV2VEC2_BASE,
                "wav2vec2_large": torchaudio.pipelines.WAV2VEC2_LARGE,
                "hubert_base": torchaudio.pipelines.HUBERT_BASE,
            }
            if self.model_name not in bundles:
                raise ValueError(f"unknown torchaudio model {self.model_name!r}")
            bundle = bundles[self.model_name]
            self.target_sample_rate = int(bundle.sample_rate)
            if self.device in {"gpu", "cuda"}:
                if not torch.cuda.is_available():
                    raise RuntimeError("GPU audio embeddings requested but CUDA is unavailable")
                self._deep_device = "cuda"
            elif self.device == "cpu":
                self._deep_device = "cpu"
            else:
                self._deep_device = "cuda" if torch.cuda.is_available() else "cpu"
            self._deep_model = bundle.get_model().eval().to(self._deep_device)
            return True
        except Exception as exc:
            self._deep_model = None
            self._record_error(f"could not load audio embedding model {self.model_name!r}: {exc}")
            return False

    def _encode_embeddings(self, values: pd.Series) -> np.ndarray | None:
        if not self._load_deep_model():
            return None
        import torch

        prepared: list[tuple[int, np.ndarray]] = []
        result_rows: list[np.ndarray | None] = [None] * len(values)
        for position, (index, value) in enumerate(values.items()):
            try:
                waveform, sample_rate, _, _ = self._load_audio(value)
                if sample_rate != self.target_sample_rate:
                    divisor = math.gcd(sample_rate, self.target_sample_rate)
                    waveform = resample_poly(
                        waveform,
                        self.target_sample_rate // divisor,
                        sample_rate // divisor,
                    ).astype(np.float32)
                prepared.append((position, waveform.astype(np.float32, copy=False)))
            except Exception as exc:
                self._record_error(f"audio embedding row {index!r} failed: {exc}")

        reference: np.ndarray | None = None
        for offset in range(0, len(prepared), self.batch_size):
            chunk = prepared[offset : offset + self.batch_size]
            lengths_np = np.asarray([len(waveform) for _, waveform in chunk], dtype=np.int64)
            max_length = int(lengths_np.max(initial=1))
            padded = np.zeros((len(chunk), max_length), dtype=np.float32)
            for row_index, (_position, waveform) in enumerate(chunk):
                padded[row_index, : len(waveform)] = waveform
            tensor = torch.from_numpy(padded).to(self._deep_device)
            lengths = torch.from_numpy(lengths_np).to(self._deep_device)
            try:
                with torch.inference_mode():
                    feature_layers, output_lengths = self._deep_model.extract_features(tensor, lengths)
            except TypeError:
                with torch.inference_mode():
                    feature_layers, output_lengths = self._deep_model.extract_features(tensor)
            encoded = feature_layers[-1]
            if output_lengths is None:
                output_lengths = torch.full(
                    (encoded.shape[0],), encoded.shape[1], device=encoded.device, dtype=torch.long
                )
            for row_index, (position, _waveform) in enumerate(chunk):
                valid_length = max(1, min(int(output_lengths[row_index].item()), encoded.shape[1]))
                row = np.asarray(
                    encoded[row_index, :valid_length].mean(dim=0).detach().cpu(),
                    dtype=np.float32,
                )
                result_rows[position] = row
                reference = np.zeros_like(row)

        if reference is None:
            return None
        return np.vstack([row if row is not None else reference for row in result_rows])

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        state["_deep_model"] = None
        return state


def _frame_signal(signal: np.ndarray, frame_length: int, hop_length: int) -> np.ndarray:
    if len(signal) < frame_length:
        signal = np.pad(signal, (0, frame_length - len(signal)))
    n_frames = 1 + max(0, (len(signal) - frame_length) // hop_length)
    shape = (n_frames, frame_length)
    strides = (signal.strides[0] * hop_length, signal.strides[0])
    return np.lib.stride_tricks.as_strided(signal, shape=shape, strides=strides).copy()


def _read_audio_file(path: Path) -> tuple[np.ndarray, int]:
    try:
        import soundfile as sf

        waveform, sample_rate = sf.read(path, always_2d=False, dtype="float32")
        return np.asarray(waveform), int(sample_rate)
    except Exception:
        if path.suffix.lower() != ".wav":
            raise RuntimeError(
                "decoding this audio format requires the 'audio' extra: pip install smarttab[audio]"
            )
        return _read_wave_bytes(path.read_bytes())


def _read_audio_bytes(payload: bytes) -> tuple[np.ndarray, int]:
    try:
        import soundfile as sf

        waveform, sample_rate = sf.read(io.BytesIO(payload), always_2d=False, dtype="float32")
        return np.asarray(waveform), int(sample_rate)
    except Exception:
        return _read_wave_bytes(payload)


def _read_wave_bytes(payload: bytes) -> tuple[np.ndarray, int]:
    with wave.open(io.BytesIO(payload), "rb") as reader:
        channels = reader.getnchannels()
        sample_width = reader.getsampwidth()
        sample_rate = reader.getframerate()
        frames = reader.readframes(reader.getnframes())
    dtype_map = {1: np.uint8, 2: np.int16, 4: np.int32}
    if sample_width not in dtype_map:
        raise ValueError(f"unsupported WAV sample width {sample_width}")
    array = np.frombuffer(frames, dtype=dtype_map[sample_width])
    if sample_width == 1:
        array = (array.astype(np.float32) - 128.0) / 128.0
    else:
        scale = float(2 ** (8 * sample_width - 1))
        array = array.astype(np.float32) / scale
    if channels > 1:
        array = array.reshape(-1, channels)
    return array, int(sample_rate)


def _mel_filterbank(sample_rate: int, n_fft: int, n_mels: int) -> np.ndarray:
    n_frequencies = n_fft // 2 + 1
    min_mel = 0.0
    max_mel = 2595.0 * np.log10(1.0 + (sample_rate / 2.0) / 700.0)
    mel_points = np.linspace(min_mel, max_mel, n_mels + 2)
    hz_points = 700.0 * (10 ** (mel_points / 2595.0) - 1.0)
    bins = np.floor((n_fft + 1) * hz_points / sample_rate).astype(int)
    bins = np.clip(bins, 0, n_frequencies - 1)
    filters = np.zeros((n_mels, n_frequencies), dtype=np.float64)
    for index in range(1, n_mels + 1):
        left, center, right = bins[index - 1], bins[index], bins[index + 1]
        if center <= left:
            center = min(n_frequencies - 1, left + 1)
        if right <= center:
            right = min(n_frequencies, center + 1)
        if center > left:
            filters[index - 1, left:center] = (np.arange(left, center) - left) / (center - left)
        if right > center:
            filters[index - 1, center:right] = (right - np.arange(center, right)) / (right - center)
    return filters
