#!/usr/bin/env python3
"""Reproducible CPU benchmark suite for every SmartTab modality.

The suite avoids network downloads. It uses two bundled public datasets and
three deterministic local/procedural modality datasets:

* Breast Cancer Wisconsin Diagnostic (scikit-learn): tabular binary.
* Installed scikit-learn vs pandas source snippets: raw-text binary.
* Optical Recognition of Handwritten Digits (scikit-learn): image multiclass.
* Procedural acoustic events: audio multiclass.
* Moving Digits derived from the scikit-learn digits images: video multiclass.
* Mixed Digits: tabular + text + image + audio, multiclass.

This is a system benchmark, not a claim of state-of-the-art model quality.
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import tempfile
import threading
import time
from typing import Any, Callable

import numpy as np
import pandas as pd
import psutil
import sklearn
from sklearn.datasets import load_breast_cancer, load_digits

import smarttab


class PeakRSS:
    def __init__(self, interval: float = 0.01) -> None:
        self.interval = interval
        self.process = psutil.Process()
        self.start_rss = 0
        self.peak_rss = 0
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def __enter__(self) -> "PeakRSS":
        self.start_rss = self.process.memory_info().rss
        self.peak_rss = self.start_rss

        def sample() -> None:
            while not self._stop.wait(self.interval):
                try:
                    self.peak_rss = max(self.peak_rss, self.process.memory_info().rss)
                except psutil.Error:
                    return

        self._thread = threading.Thread(target=sample, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *_: object) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1)
        try:
            self.peak_rss = max(self.peak_rss, self.process.memory_info().rss)
        except psutil.Error:
            pass

    @property
    def delta_mb(self) -> float:
        return max(0.0, self.peak_rss - self.start_rss) / (1024 * 1024)


def tabular_dataset(limit: int | None = None) -> tuple[pd.DataFrame, str, dict[str, Any]]:
    dataset = load_breast_cancer(as_frame=True)
    frame = dataset.frame.copy()
    frame = frame.rename(columns={"target": "label"})
    rng = np.random.default_rng(3101)
    feature_columns = [column for column in frame.columns if column != "label"]
    # Controlled data-quality defects exercise imputation, clipping, indicators,
    # skew handling, and rare-category grouping without corrupting labels.
    for column in feature_columns[:5]:
        rows = rng.choice(len(frame), size=max(2, len(frame) // 35), replace=False)
        frame.loc[rows, column] = np.nan
    frame.loc[rng.choice(len(frame), 4, replace=False), feature_columns[5]] *= 25
    frame["radius_band"] = pd.cut(
        frame["mean radius"], bins=[-np.inf, 12, 16, 20, np.inf], labels=["small", "medium", "large", "very_large"]
    ).astype(object)
    frame.loc[rng.choice(len(frame), 3, replace=False), "radius_band"] = "rare_sensor_code"
    if limit:
        frame = frame.sample(min(limit, len(frame)), random_state=42).reset_index(drop=True)
    return frame, "label", {
        "name": "Breast Cancer Wisconsin Diagnostic + controlled quality defects",
        "origin": "scikit-learn bundled dataset",
        "task": "binary",
        "modalities": {},
    }


def _source_snippets(package: Any, label: int, max_files: int, chunk_chars: int) -> list[tuple[str, int]]:
    root = Path(package.__file__).resolve().parent
    rows: list[tuple[str, int]] = []
    for path in sorted(root.rglob("*.py")):
        if any(part in {"tests", "test", "__pycache__", "vendor", "_vendor"} for part in path.parts):
            continue
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        text = " ".join(text.split())
        if len(text) < 300:
            continue
        for start in range(0, min(len(text), chunk_chars * 4), chunk_chars):
            chunk = text[start : start + chunk_chars]
            if len(chunk) >= 250:
                rows.append((chunk, label))
        if len(rows) >= max_files:
            break
    return rows[:max_files]


def text_dataset(limit: int | None = None) -> tuple[pd.DataFrame, str, dict[str, Any]]:
    import pandas as pandas_package
    import sklearn as sklearn_package

    rows = _source_snippets(sklearn_package, 0, 220, 1100)
    rows += _source_snippets(pandas_package, 1, 220, 1100)
    rng = np.random.default_rng(3102)
    rng.shuffle(rows)
    if limit:
        rows = rows[:limit]
    frame = pd.DataFrame(rows, columns=["document", "label"])
    return frame, "label", {
        "name": "Installed scikit-learn vs pandas source-code snippets",
        "origin": "local installed package source; no network download",
        "task": "binary",
        "modalities": {"document": "text"},
    }


def image_dataset(limit: int | None = None) -> tuple[pd.DataFrame, str, dict[str, Any]]:
    digits = load_digits()
    keep = digits.target < 5
    images = (digits.images[keep] / 16.0 * 255).astype(np.uint8)
    labels = digits.target[keep]
    indices = np.arange(len(labels))
    rng = np.random.default_rng(3103)
    rng.shuffle(indices)
    if limit:
        indices = indices[:limit]
    frame = pd.DataFrame({"image": [images[index] for index in indices], "label": labels[indices]})
    return frame, "label", {
        "name": "Optical Recognition of Handwritten Digits (classes 0-4)",
        "origin": "scikit-learn bundled dataset",
        "task": "multiclass",
        "modalities": {"image": "image"},
    }


def _audio_sample(class_id: int, sample_id: int, sample_rate: int = 8000) -> tuple[int, np.ndarray]:
    rng = np.random.default_rng(10_000 + class_id * 1000 + sample_id)
    duration = rng.uniform(0.32, 0.58)
    t = np.arange(int(sample_rate * duration), dtype=np.float32) / sample_rate
    frequencies = [220.0, 330.0, 440.0, 660.0]
    base = frequencies[class_id]
    phase = rng.uniform(0, 2 * np.pi)
    if class_id == 0:
        signal = np.sin(2 * np.pi * base * t + phase)
    elif class_id == 1:
        signal = np.sin(2 * np.pi * base * t + phase) + 0.35 * np.sin(2 * np.pi * base * 2 * t)
    elif class_id == 2:
        chirp = base + 180 * t / max(duration, 1e-6)
        signal = np.sin(2 * np.pi * chirp * t + phase)
    else:
        signal = np.sign(np.sin(2 * np.pi * base * t + phase)) * 0.65
    envelope_base = np.clip(np.sin(np.linspace(0, np.pi, len(t), dtype=np.float32)), 0.0, None)
    envelope = envelope_base ** rng.uniform(0.7, 1.8)
    signal = signal * envelope + rng.normal(0, 0.035, len(t))
    return sample_rate, np.asarray(signal / max(np.max(np.abs(signal)), 1e-6), dtype=np.float32)


def audio_dataset(limit: int | None = None) -> tuple[pd.DataFrame, str, dict[str, Any]]:
    per_class = 90
    rows = [(_audio_sample(label, index), label) for label in range(4) for index in range(per_class)]
    rng = np.random.default_rng(3104)
    rng.shuffle(rows)
    if limit:
        rows = rows[:limit]
    frame = pd.DataFrame(rows, columns=["audio", "label"])
    return frame, "label", {
        "name": "Deterministic procedural acoustic events",
        "origin": "generated locally; four waveform families",
        "task": "multiclass",
        "modalities": {"audio": "audio"},
    }


def _moving_digit(image: np.ndarray, label: int, sample_id: int) -> np.ndarray:
    rng = np.random.default_rng(20_000 + sample_id)
    glyph = np.kron(image / 16.0, np.ones((2, 2), dtype=np.float32))
    glyph = (glyph * 255).astype(np.uint8)
    frames: list[np.ndarray] = []
    directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]
    dy, dx = directions[label]
    start_y = int(rng.integers(5, 11))
    start_x = int(rng.integers(5, 11))
    for step in range(8):
        canvas = np.zeros((32, 32), dtype=np.uint8)
        y = int(np.clip(start_y + dy * step, 0, 16))
        x = int(np.clip(start_x + dx * step, 0, 16))
        canvas[y : y + 16, x : x + 16] = glyph
        canvas = np.clip(canvas.astype(float) + rng.normal(0, 3, canvas.shape), 0, 255).astype(np.uint8)
        frames.append(np.repeat(canvas[..., None], 3, axis=2))
    return np.stack(frames)


def video_dataset(limit: int | None = None) -> tuple[pd.DataFrame, str, dict[str, Any]]:
    digits = load_digits()
    candidates = np.flatnonzero(digits.target < 4)
    rng = np.random.default_rng(3105)
    rng.shuffle(candidates)
    desired = min(len(candidates), limit or 320)
    candidates = candidates[:desired]
    videos = [_moving_digit(digits.images[index], int(digits.target[index]), sample_id) for sample_id, index in enumerate(candidates)]
    frame = pd.DataFrame({"video": videos, "label": digits.target[candidates]})
    return frame, "label", {
        "name": "Moving Digits (digits 0-3 with class-dependent motion)",
        "origin": "generated from scikit-learn bundled digits",
        "task": "multiclass",
        "modalities": {"video": "video"},
    }


def mixed_dataset(limit: int | None = None) -> tuple[pd.DataFrame, str, dict[str, Any]]:
    digits = load_digits()
    candidates = np.flatnonzero(digits.target < 4)
    rng = np.random.default_rng(3106)
    rng.shuffle(candidates)
    candidates = candidates[: min(len(candidates), limit or 320)]
    rows = []
    class_words = ["round closed", "vertical narrow", "curved diagonal", "double arc"]
    for sample_id, index in enumerate(candidates):
        label = int(digits.target[index])
        image = (digits.images[index] / 16.0 * 255).astype(np.uint8)
        rows.append(
            {
                "mean_intensity": float(image.mean()),
                "ink_pixels": int((image > 80).sum()),
                "scanner": "scanner_a" if sample_id % 5 else "rare_scanner",
                "description": f"handwritten symbol {class_words[label]} sample quality {sample_id % 7}",
                "image": image,
                "audio": _audio_sample(label, sample_id),
                "label": label,
            }
        )
    frame = pd.DataFrame(rows)
    missing_rows = rng.choice(len(frame), max(1, len(frame) // 20), replace=False)
    for row in missing_rows[: len(missing_rows) // 2]:
        frame.at[row, "description"] = ""
    for row in missing_rows[len(missing_rows) // 2 :]:
        frame.at[row, "audio"] = None
    return frame, "label", {
        "name": "Mixed Digits (tabular + text + image + audio)",
        "origin": "generated from scikit-learn bundled digits",
        "task": "multiclass",
        "modalities": {"description": "text", "image": "image", "audio": "audio"},
    }


DATASETS: dict[str, Callable[[int | None], tuple[pd.DataFrame, str, dict[str, Any]]]] = {
    "tabular": tabular_dataset,
    "text": text_dataset,
    "image": image_dataset,
    "audio": audio_dataset,
    "video": video_dataset,
    "mixed": mixed_dataset,
}


def _scalar_metrics(metrics: dict[str, Any]) -> dict[str, float]:
    result = {}
    for key, value in metrics.items():
        if isinstance(value, (int, float, np.integer, np.floating)) and np.isfinite(value):
            result[key] = float(value)
    return result


def run_case(name: str, frame: pd.DataFrame, target: str, metadata: dict[str, Any], output: Path) -> dict[str, Any]:
    common = dict(
        target=target,
        task_type=metadata["task"],
        modalities=metadata["modalities"],
        model="auto",
        ensemble="none",
        ensemble_models_limit=3,
        optimize=False,
        n_trials=0,
        n_estimators=70,
        feature_budget=160 if name != "mixed" else 240,
        speed_accuracy=0.62,
        multimodal_backend="classical",
        supervised_adaptation="auto",
        adapter_features=12,
        device="cpu",
        cpu_threads=max(1, min(8, (os.cpu_count() or 2) // 2)),
        feature_workers=1,
        report=False,
        explain=False,
        static_charts=False,
        random_state=42,
        verbose=0,
        data_science={
            "quality_audit": True,
            "numeric_imputation": "median",
            "add_missing_indicators": True,
            "rare_category_min_frequency": 0.01,
            "numeric_transform": "auto",
            "winsorize": "auto",
            "calibration": "sigmoid",
            "conformal": True,
            "ood_detection": True,
            "drift_monitoring": True,
            "modality_dropout": "auto",
        },
    )
    gc.collect()
    started = time.perf_counter()
    with PeakRSS() as memory:
        model = smarttab.fit(frame, **common)
    fit_seconds = time.perf_counter() - started

    features = frame.drop(columns=[target])
    # Warm-up avoids reporting import and first-call overhead as steady-state inference.
    model.predict(features.head(min(3, len(features))))
    infer_started = time.perf_counter()
    model.predict(features)
    inference_seconds = time.perf_counter() - infer_started

    case_dir = output / name
    case_dir.mkdir(parents=True, exist_ok=True)
    report = model.report(case_dir / "model_report")
    bundle_path = case_dir / f"{name}.smarttab"
    model.save(bundle_path)

    primary_value = model.metrics.get(model.primary_metric)
    if not isinstance(primary_value, (int, float, np.integer, np.floating)):
        primary_value = None
    result = {
        "case": name,
        "dataset": metadata["name"],
        "origin": metadata["origin"],
        "task": metadata["task"],
        "modalities": sorted(set(metadata["modalities"].values())) or ["tabular"],
        "rows": int(len(frame)),
        "raw_feature_columns": int(frame.shape[1] - 1),
        "final_model_features": int(len(model.feature_names)),
        "model_name": model.model_name,
        "ensemble_selected": bool(model.ensemble_info and model.ensemble_info.get("selected")),
        "primary_metric": model.primary_metric,
        "primary_metric_value": float(primary_value) if primary_value is not None else None,
        "metrics": _scalar_metrics(model.metrics),
        "fit_seconds": float(fit_seconds),
        "inference_seconds_all_rows": float(inference_seconds),
        "inference_samples_per_second": float(len(frame) / max(inference_seconds, 1e-9)),
        "peak_rss_delta_mb": float(memory.delta_mb),
        "bundle_size_mb": float(bundle_path.stat().st_size / (1024 * 1024)),
        "data_quality_score": float(model.data_quality_report.get("quality_score", 0.0)),
        "data_quality_issue_count": int(len(model.data_quality_report.get("issues", []))),
        "cleaning": report.get("cleaning_report", {}),
        "uncertainty": report.get("uncertainty_info", {}),
        "feature_space": model.feature_space,
    }
    (case_dir / "result.json").write_text(json.dumps(result, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    return result


def write_outputs(results: list[dict[str, Any]], output: Path, elapsed: float) -> None:
    environment = {
        "smarttab": smarttab.__version__,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "processor": platform.processor(),
        "logical_cpu_count": psutil.cpu_count(logical=True),
        "physical_cpu_count": psutil.cpu_count(logical=False),
        "ram_total_gb": round(psutil.virtual_memory().total / 1024**3, 3),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scikit_learn": sklearn.__version__,
        "total_suite_seconds": elapsed,
    }
    payload = {"environment": environment, "results": results}
    (output / "cpu_benchmark_results.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=str), encoding="utf-8"
    )
    columns = [
        "case", "dataset", "task", "rows", "final_model_features", "model_name",
        "primary_metric", "primary_metric_value", "fit_seconds",
        "inference_samples_per_second", "peak_rss_delta_mb", "bundle_size_mb",
        "data_quality_score", "data_quality_issue_count",
    ]
    with (output / "cpu_benchmark_results.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(results)

    lines = [
        "# SmartTab CPU Benchmark Results",
        "",
        "These measurements describe this machine and this exact package build. They are not universal accuracy or latency guarantees.",
        "",
        "## Environment",
        "",
    ]
    lines.extend(f"- **{key}:** {value}" for key, value in environment.items())
    lines += [
        "",
        "## Results",
        "",
        "| Case | Dataset | Rows | Model | Primary metric | Value | Fit s | Samples/s | Peak Δ RSS MB | Features |",
        "|---|---|---:|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in results:
        value = row["primary_metric_value"]
        lines.append(
            f"| {row['case']} | {row['dataset']} | {row['rows']} | {row['model_name']} | "
            f"{row['primary_metric']} | {value:.4f} | {row['fit_seconds']:.3f} | "
            f"{row['inference_samples_per_second']:.1f} | {row['peak_rss_delta_mb']:.1f} | "
            f"{row['final_model_features']} |"
            if value is not None else
            f"| {row['case']} | {row['dataset']} | {row['rows']} | {row['model_name']} | "
            f"{row['primary_metric']} | n/a | {row['fit_seconds']:.3f} | "
            f"{row['inference_samples_per_second']:.1f} | {row['peak_rss_delta_mb']:.1f} | "
            f"{row['final_model_features']} |"
        )
    lines += [
        "",
        "## Interpretation",
        "",
        "- The tabular and image cases use bundled public scikit-learn datasets.",
        "- Text uses source snippets from the locally installed pandas and scikit-learn packages; no corpus is redistributed.",
        "- Audio and video cases are deterministic procedural/system benchmarks and should not be mistaken for broad real-world quality evidence.",
        "- Mixed Digits validates bounded multimodal fusion, missing-modality handling, calibration, conformal sets, and persistence.",
        "- Competitive claims require external benchmark datasets, repeated seeds, confidence intervals, and comparisons against task-specific baselines.",
    ]
    (output / "CPU_BENCHMARKS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("benchmark_results/cpu"))
    parser.add_argument("--quick", action="store_true", help="Use smaller datasets for CI/smoke runs")
    parser.add_argument("--cases", nargs="*", choices=list(DATASETS), default=list(DATASETS))
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    environment_payload = {
        "smarttab": smarttab.__version__,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "processor": platform.processor(),
        "logical_cpu_count": psutil.cpu_count(logical=True),
        "physical_cpu_count": psutil.cpu_count(logical=False),
        "ram_total_gb": round(psutil.virtual_memory().total / 1024**3, 3),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scikit_learn": sklearn.__version__,
    }
    environment_path = args.output / "environment.json"
    if not environment_path.exists():
        environment_path.write_text(
            json.dumps(environment_payload, indent=2, ensure_ascii=False), encoding="utf-8"
        )
    quick_limits = {
        "tabular": 320, "text": 220, "image": 350,
        "audio": 180, "video": 140, "mixed": 160,
    }
    full_limits = {
        "tabular": None, "text": 420, "image": 800,
        "audio": 360, "video": 180, "mixed": 240,
    }
    started = time.perf_counter()
    if not args.worker and len(args.cases) > 1:
        results = []
        for name in args.cases:
            command = [
                sys.executable, str(Path(__file__).resolve()),
                "--output", str(args.output), "--cases", name, "--worker",
            ]
            if args.quick:
                command.append("--quick")
            environment = os.environ.copy()
            source_root = str(Path(__file__).resolve().parents[1] / "src")
            environment["PYTHONPATH"] = source_root + os.pathsep + environment.get("PYTHONPATH", "")
            subprocess.run(command, check=True, env=environment)
            result_path = args.output / name / "result.json"
            results.append(json.loads(result_path.read_text(encoding="utf-8")))
        write_outputs(results, args.output, time.perf_counter() - started)
        return

    results = []
    for name in args.cases:
        limit = quick_limits[name] if args.quick else full_limits[name]
        frame, target, metadata = DATASETS[name](limit)
        print(f"[{name}] rows={len(frame)}", flush=True)
        result = run_case(name, frame, target, metadata, args.output)
        results.append(result)
        print(
            f"[{name}] {result['primary_metric']}={result['primary_metric_value']} "
            f"fit={result['fit_seconds']:.3f}s",
            flush=True,
        )
    if not args.worker:
        write_outputs(results, args.output, time.perf_counter() - started)


if __name__ == "__main__":
    main()
