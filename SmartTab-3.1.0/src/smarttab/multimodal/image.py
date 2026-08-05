"""Image feature extraction with bounded classical descriptors and optional timm embeddings."""

from __future__ import annotations

import importlib.util
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

from smarttab.multimodal.base import BaseFeatureExtractor
from smarttab.multimodal.common import entropy, safe_float32_frame


class ImageFeatureExtractor(BaseFeatureExtractor):
    modality = "image"

    def __init__(
        self,
        *,
        max_features: int,
        backend: str = "auto",
        speed_accuracy: float = 0.5,
        allow_model_download: bool = False,
        batch_size: int = 32,
        workers: int = 1,
        random_state: int = 42,
        error_policy: str = "warn",
        device: str = "auto",
        model_name: str = "mobilenetv3_small_100",
        analysis_size: int | str = "auto",
        histogram_bins: int = 8,
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
        self.analysis_size = (
            int(96 + 128 * self.speed_accuracy) if analysis_size == "auto" else int(analysis_size)
        )
        self.histogram_bins = max(4, min(32, int(histogram_bins)))
        self.classical_names_: list[str] = []
        self.classical_indices_: list[int] = []
        self.embedding_names_: list[str] = []
        self.embedding_reducer_: PCA | None = None
        self.backend_used_ = "classical"
        self.notes_: list[str] = []
        self._deep_model: Any = None
        self._deep_transform: Any = None
        self._deep_device = "cpu"

    def fit_transform(self, values: pd.Series, y: Any = None) -> pd.DataFrame:
        classical_matrix, classical_names = self._classical_batch(values)
        classical_budget = min(classical_matrix.shape[1], self.max_features)
        use_deep = self._should_use_embeddings() and self.max_features - classical_budget >= 8
        if use_deep:
            classical_budget = min(classical_matrix.shape[1], max(16, int(self.max_features * 0.35)))
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
                self.notes_.append("pretrained image embeddings unavailable; classical features used")
        matrix = np.column_stack(parts)[:, : self.max_features]
        self.feature_names_ = names[: matrix.shape[1]]
        self._fitted = True
        return safe_float32_frame(matrix, self.feature_names_, values.index)

    def transform(self, values: pd.Series) -> pd.DataFrame:
        if not self._fitted:
            raise RuntimeError("ImageFeatureExtractor must be fitted before transform()")
        classical_matrix, _ = self._classical_batch(values)
        parts = [classical_matrix[:, self.classical_indices_]]
        if self.embedding_names_:
            embeddings = self._encode_embeddings(values)
            if embeddings is None:
                raise RuntimeError(f"pretrained image encoder {self.model_name!r} is unavailable")
            if self.embedding_reducer_ is not None:
                embeddings = self.embedding_reducer_.transform(embeddings)
            parts.append(embeddings[:, : len(self.embedding_names_)])
        matrix = np.column_stack(parts)[:, : len(self.feature_names_)]
        return safe_float32_frame(matrix, self.feature_names_, values.index)

    @staticmethod
    def _diverse_feature_order(names: list[str]) -> list[str]:
        essential = [
            name for name in (
                "width", "height", "aspect_ratio", "megapixels", "has_alpha",
                "gray_entropy", "brightness", "contrast", "colorfulness",
                "edge_mean", "edge_density", "sharpness_laplacian_var",
                "horizontal_symmetry", "vertical_symmetry",
                "frequency_low_ratio", "frequency_spectral_entropy",
                "gradient_orientation_entropy", "blockiness",
            ) if name in names
        ]
        groups = [
            [name for name in names if name.startswith(("r_", "g_", "b_"))],
            [name for name in names if name.startswith(("h_", "s_", "v_"))],
            [name for name in names if name.startswith("gradient_bin_")],
            [name for name in names if name.startswith("grid_")],
            [name for name in names if name not in essential],
        ]
        ordered = list(essential)
        positions = [0] * len(groups)
        while True:
            added = False
            for index, group in enumerate(groups):
                while positions[index] < len(group) and group[positions[index]] in ordered:
                    positions[index] += 1
                if positions[index] < len(group):
                    ordered.append(group[positions[index]])
                    positions[index] += 1
                    added = True
            if not added:
                break
        return list(dict.fromkeys(ordered))

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
            return self._classical_features(self._load_image(value))
        except Exception as exc:
            self._record_error(f"image row {index!r} could not be decoded: {exc}")
            return self._empty_feature_dict()

    def _load_image(self, value: Any):
        from PIL import Image

        if value is None or (isinstance(value, float) and np.isnan(value)):
            raise ValueError("missing image")
        if isinstance(value, Image.Image):
            return value.copy()
        if isinstance(value, np.ndarray):
            array = np.asarray(value)
            if array.ndim == 2:
                return Image.fromarray(_to_uint8(array), mode="L")
            if array.ndim == 3 and array.shape[-1] in (1, 3, 4):
                if array.shape[-1] == 1:
                    array = array[..., 0]
                return Image.fromarray(_to_uint8(array))
            raise ValueError(f"unsupported image array shape {array.shape}")
        if isinstance(value, (str, Path)):
            path = Path(value).expanduser()
            if not path.exists() or not path.is_file():
                raise FileNotFoundError(path)
            with Image.open(path) as image:
                return image.copy()
        if isinstance(value, (bytes, bytearray, memoryview)):
            import io

            with Image.open(io.BytesIO(bytes(value))) as image:
                return image.copy()
        raise TypeError(f"unsupported image value type {type(value).__name__}")

    def _classical_features(self, image) -> dict[str, float]:
        from PIL import Image

        original_width, original_height = image.size
        has_alpha = "A" in image.getbands()
        rgba = image.convert("RGBA") if has_alpha else None
        rgb = image.convert("RGB")
        rgb.thumbnail((self.analysis_size, self.analysis_size), Image.Resampling.BILINEAR)
        array = np.asarray(rgb, dtype=np.float32) / 255.0
        gray = 0.299 * array[..., 0] + 0.587 * array[..., 1] + 0.114 * array[..., 2]
        hsv = _rgb_to_hsv(array)
        gx = np.diff(gray, axis=1, append=gray[:, -1:])
        gy = np.diff(gray, axis=0, append=gray[-1:, :])
        gradient = np.sqrt(gx * gx + gy * gy)
        orientation = (np.arctan2(gy, gx) + np.pi) % np.pi
        orientation_hist, _ = np.histogram(
            orientation, bins=8, range=(0.0, np.pi), weights=gradient
        )
        orientation_hist = orientation_hist.astype(float)
        orientation_hist /= max(1e-12, orientation_hist.sum())
        laplacian = (
            -4 * gray
            + np.roll(gray, 1, axis=0)
            + np.roll(gray, -1, axis=0)
            + np.roll(gray, 1, axis=1)
            + np.roll(gray, -1, axis=1)
        )
        rg = array[..., 0] - array[..., 1]
        yb = 0.5 * (array[..., 0] + array[..., 1]) - array[..., 2]
        colorfulness = float(np.sqrt(rg.std() ** 2 + yb.std() ** 2) + 0.3 * np.sqrt(rg.mean() ** 2 + yb.mean() ** 2))
        h, w = gray.shape
        center = gray[h // 4 : max(h // 4 + 1, 3 * h // 4), w // 4 : max(w // 4 + 1, 3 * w // 4)]
        border_mask = np.ones_like(gray, dtype=bool)
        border_mask[h // 4 : 3 * h // 4, w // 4 : 3 * w // 4] = False
        border = gray[border_mask]
        horizontal_symmetry = 1.0 - float(np.mean(np.abs(gray - np.fliplr(gray))))
        vertical_symmetry = 1.0 - float(np.mean(np.abs(gray - np.flipud(gray))))
        vertical_boundaries = np.abs(np.diff(gray, axis=1))[:, 7::8]
        horizontal_boundaries = np.abs(np.diff(gray, axis=0))[7::8, :]
        blockiness_values = []
        if vertical_boundaries.size:
            blockiness_values.append(float(vertical_boundaries.mean()))
        if horizontal_boundaries.size:
            blockiness_values.append(float(horizontal_boundaries.mean()))
        blockiness = float(np.mean(blockiness_values)) if blockiness_values else 0.0
        features: dict[str, float] = {
            "width": float(original_width),
            "height": float(original_height),
            "aspect_ratio": float(original_width / max(1, original_height)),
            "megapixels": float(original_width * original_height / 1_000_000),
            "has_alpha": float(has_alpha),
            "alpha_coverage": float(np.asarray(rgba.getchannel("A"), dtype=np.float32).mean() / 255.0) if rgba is not None else 1.0,
            "gray_entropy": entropy((gray * 255).astype(np.uint8), bins=32),
            "brightness": float(gray.mean()),
            "contrast": float(gray.std()),
            "colorfulness": colorfulness,
            "edge_mean": float(gradient.mean()),
            "edge_std": float(gradient.std()),
            "edge_density": float((gradient > np.quantile(gradient, 0.75)).mean()),
            "sharpness_laplacian_var": float(laplacian.var()),
            "horizontal_symmetry": horizontal_symmetry,
            "vertical_symmetry": vertical_symmetry,
            "center_border_contrast": float(center.mean() - border.mean()) if border.size else 0.0,
            "dark_pixel_ratio": float((gray < 0.1).mean()),
            "bright_pixel_ratio": float((gray > 0.9).mean()),
            "gradient_orientation_entropy": float(
                -(orientation_hist[orientation_hist > 0] * np.log2(orientation_hist[orientation_hist > 0])).sum()
            ),
            "blockiness": blockiness,
        }
        for bin_index, value in enumerate(orientation_hist):
            features[f"gradient_bin_{bin_index:02d}"] = float(value)
        y_edges = np.linspace(0, h, 4, dtype=int)
        x_edges = np.linspace(0, w, 4, dtype=int)
        for row in range(3):
            for column in range(3):
                region_gray = gray[y_edges[row]:y_edges[row + 1], x_edges[column]:x_edges[column + 1]]
                region_sat = hsv[y_edges[row]:y_edges[row + 1], x_edges[column]:x_edges[column + 1], 1]
                features[f"grid_{row}_{column}_brightness"] = float(region_gray.mean())
                features[f"grid_{row}_{column}_contrast"] = float(region_gray.std())
                features[f"grid_{row}_{column}_saturation"] = float(region_sat.mean())
        for channel_index, channel_name in enumerate(("r", "g", "b")):
            channel = array[..., channel_index]
            for stat_name, stat_value in (
                ("mean", channel.mean()), ("std", channel.std()),
                ("q10", np.quantile(channel, 0.1)), ("q50", np.quantile(channel, 0.5)),
                ("q90", np.quantile(channel, 0.9)),
            ):
                features[f"{channel_name}_{stat_name}"] = float(stat_value)
            hist, _ = np.histogram(channel, bins=self.histogram_bins, range=(0.0, 1.0), density=False)
            hist = hist.astype(float) / max(1, hist.sum())
            for bin_index, value in enumerate(hist):
                features[f"{channel_name}_hist_{bin_index:02d}"] = float(value)
        for channel_index, channel_name in enumerate(("h", "s", "v")):
            channel = hsv[..., channel_index]
            features[f"{channel_name}_mean"] = float(channel.mean())
            features[f"{channel_name}_std"] = float(channel.std())
        fft = np.abs(np.fft.rfft2(gray - gray.mean()))
        low = fft[: max(1, fft.shape[0] // 8), : max(1, fft.shape[1] // 8)]
        features["frequency_low_ratio"] = float(low.sum() / max(1e-9, fft.sum()))
        features["frequency_spectral_entropy"] = entropy(np.log1p(fft), bins=32)
        return features

    def _empty_feature_dict(self) -> dict[str, float]:
        # The key order must match _classical_features.
        blank = np.zeros((8, 8, 3), dtype=np.uint8)
        try:
            from PIL import Image

            result = self._classical_features(Image.fromarray(blank))
            return {name: np.nan for name in result}
        except Exception:
            base = ["width", "height", "aspect_ratio", "megapixels", "has_alpha", "alpha_coverage"]
            return {name: np.nan for name in base}

    def _should_use_embeddings(self) -> bool:
        if self.backend == "classical":
            return False
        if self.backend in {"pretrained", "hybrid"}:
            return True
        return bool(
            self.speed_accuracy >= 0.75
            and self.allow_model_download
            and importlib.util.find_spec("timm") is not None
        )

    def _load_deep_model(self) -> bool:
        if self._deep_model is not None:
            return True
        if not self.allow_model_download:
            return False
        if importlib.util.find_spec("timm") is None or importlib.util.find_spec("torch") is None:
            return False
        try:
            import timm
            import torch
            from timm.data import create_transform, resolve_model_data_config

            if self.device in {"gpu", "cuda"}:
                if not torch.cuda.is_available():
                    raise RuntimeError("GPU image embeddings requested but CUDA is unavailable")
                self._deep_device = "cuda"
            elif self.device == "cpu":
                self._deep_device = "cpu"
            else:
                self._deep_device = "cuda" if torch.cuda.is_available() else "cpu"
            self._deep_model = timm.create_model(
                self.model_name,
                pretrained=True,
                num_classes=0,
                global_pool="avg",
            ).eval().to(self._deep_device)
            self._deep_transform = create_transform(**resolve_model_data_config(self._deep_model), is_training=False)
            return True
        except Exception as exc:
            self._deep_model = None
            self._deep_transform = None
            self._record_error(f"could not load image embedding model {self.model_name!r}: {exc}")
            return False

    def _encode_embeddings(self, values: pd.Series) -> np.ndarray | None:
        if not self._load_deep_model():
            return None
        import torch

        outputs: list[np.ndarray] = []
        zero_vector: np.ndarray | None = None
        batch: list[Any] = []
        positions: list[int] = []
        result_rows: list[np.ndarray | None] = [None] * len(values)
        for position, (_, value) in enumerate(values.items()):
            try:
                image = self._load_image(value).convert("RGB")
                batch.append(self._deep_transform(image))
                positions.append(position)
            except Exception as exc:
                self._record_error(f"image embedding input {position} failed: {exc}")
            if len(batch) >= self.batch_size or (position == len(values) - 1 and batch):
                tensor = torch.stack(batch).to(self._deep_device)
                with torch.inference_mode():
                    encoded = self._deep_model(tensor)
                encoded_array = np.asarray(encoded.detach().cpu(), dtype=np.float32)
                if encoded_array.ndim > 2:
                    encoded_array = encoded_array.reshape(encoded_array.shape[0], -1)
                for pos, row in zip(positions, encoded_array, strict=True):
                    result_rows[pos] = row
                    zero_vector = np.zeros_like(row)
                batch.clear()
                positions.clear()
        if zero_vector is None:
            return None
        outputs = [row if row is not None else zero_vector for row in result_rows]
        return np.vstack(outputs)

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        state["_deep_model"] = None
        state["_deep_transform"] = None
        return state


def _to_uint8(array: np.ndarray) -> np.ndarray:
    values = np.asarray(array)
    if values.dtype == np.uint8:
        return values
    values = values.astype(np.float32)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return np.zeros(values.shape, dtype=np.uint8)
    if finite.min() >= 0 and finite.max() <= 1.0:
        values = values * 255.0
    else:
        low, high = np.quantile(finite, [0.01, 0.99])
        values = (values - low) / max(1e-9, high - low) * 255.0
    return np.clip(values, 0, 255).astype(np.uint8)


def _rgb_to_hsv(rgb: np.ndarray) -> np.ndarray:
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    maximum = np.max(rgb, axis=-1)
    minimum = np.min(rgb, axis=-1)
    delta = maximum - minimum
    hue = np.zeros_like(maximum)
    mask = delta > 1e-9
    red_mask = mask & (maximum == r)
    green_mask = mask & (maximum == g)
    blue_mask = mask & (maximum == b)
    hue[red_mask] = ((g[red_mask] - b[red_mask]) / delta[red_mask]) % 6
    hue[green_mask] = (b[green_mask] - r[green_mask]) / delta[green_mask] + 2
    hue[blue_mask] = (r[blue_mask] - g[blue_mask]) / delta[blue_mask] + 4
    hue /= 6.0
    saturation = np.divide(delta, maximum, out=np.zeros_like(delta), where=maximum > 1e-9)
    return np.stack([hue, saturation, maximum], axis=-1)
