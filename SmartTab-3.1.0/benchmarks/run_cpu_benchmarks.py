#!/usr/bin/env python3
"""Lightweight launcher for isolated SmartTab CPU modality benchmarks.

Each modality runs in a fresh Python process. This avoids cross-case OpenMP,
BLAS, CatBoost, LightGBM, decoder, and allocator state from contaminating later
measurements. Fresh-process execution prevents global native runtime state from
contaminating later cases.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import subprocess
import sys
import time

CASES = ["tabular", "text", "image", "audio", "video"]
ALL_CASES = [*CASES, "mixed"]


def _write_outputs(results: list[dict], output: Path, elapsed: float) -> None:
    environment_path = output / "environment.json"
    environment = json.loads(environment_path.read_text(encoding="utf-8")) if environment_path.exists() else {}
    environment["total_suite_seconds"] = elapsed
    payload = {"environment": environment, "results": results}
    (output / "cpu_benchmark_results.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
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
        value = row.get("primary_metric_value")
        value_text = f"{value:.4f}" if isinstance(value, (int, float)) else "n/a"
        lines.append(
            f"| {row['case']} | {row['dataset']} | {row['rows']} | {row['model_name']} | "
            f"{row['primary_metric']} | {value_text} | {row['fit_seconds']:.3f} | "
            f"{row['inference_samples_per_second']:.1f} | {row['peak_rss_delta_mb']:.1f} | "
            f"{row['final_model_features']} |"
        )
    lines += [
        "",
        "## Interpretation",
        "",
        "- The tabular and image cases use bundled public scikit-learn datasets.",
        "- Text uses source snippets from the locally installed pandas and scikit-learn packages; no corpus is redistributed.",
        "- Audio and video cases are deterministic procedural/system benchmarks and are not broad real-world quality evidence.",
        "- Mixed Digits validates bounded multimodal fusion, missing-modality handling, calibration, conformal sets, and persistence.",
        "- Competitive claims require external datasets, repeated seeds, confidence intervals, and task-specific baselines.",
    ]
    (output / "CPU_BENCHMARKS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("benchmark_results/cpu"))
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--cases", nargs="*", choices=ALL_CASES, default=CASES)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    worker = Path(__file__).with_name("cpu_worker.py")
    source_root = Path(__file__).resolve().parents[1] / "src"
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(source_root) + os.pathsep + environment.get("PYTHONPATH", "")
    # Stable CPU benchmark settings. The library remains free to use more threads
    # outside this harness.
    environment.setdefault("OMP_NUM_THREADS", "1")
    environment.setdefault("OPENBLAS_NUM_THREADS", "1")
    environment.setdefault("MKL_NUM_THREADS", "1")
    environment.setdefault("NUMEXPR_NUM_THREADS", "1")

    started = time.perf_counter()
    results: list[dict] = []
    for case in args.cases:
        command = [
            sys.executable, str(worker), "--output", str(args.output),
            "--cases", case, "--worker",
        ]
        if args.quick:
            command.append("--quick")
        subprocess.run(command, check=True, env=environment)
        results.append(json.loads((args.output / case / "result.json").read_text(encoding="utf-8")))
    _write_outputs(results, args.output, time.perf_counter() - started)


if __name__ == "__main__":
    main()
