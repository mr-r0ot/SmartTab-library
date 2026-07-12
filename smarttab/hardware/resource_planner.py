"""Stage 3 (part 2) — turns a HardwareProfile + dataset size into concrete resource decisions."""

from __future__ import annotations

from dataclasses import dataclass, field

from smarttab.analysis.dataset_analyzer import DatasetProfile
from smarttab.exceptions import ConfigurationError
from smarttab.hardware.profiler import HardwareProfile

MIN_FREE_RAM_FRACTION = 0.15  # always leave >=15% RAM free
MIN_GPU_FREE_VRAM_MB = 1024
MIN_SAMPLES_FOR_GPU_BENEFIT = 5000


@dataclass
class ResourcePlan:
    cpu_threads: int
    use_gpu: bool
    memory_budget_mb: float
    notes: list[str] = field(default_factory=list)


def resolve_resource_plan(
    hardware: HardwareProfile,
    dataset_profile: DatasetProfile,
    device: str = "auto",
    cpu_threads: str | int = "auto",
    gpu_memory: str = "auto",
    ram_limit: str | float = "auto",
) -> ResourcePlan:
    notes: list[str] = []

    threads = _resolve_cpu_threads(hardware, cpu_threads, notes)
    use_gpu = _resolve_device(hardware, dataset_profile, device, notes)
    memory_budget_mb = _resolve_ram_limit(hardware, ram_limit, notes)

    if gpu_memory != "auto":
        notes.append(f"gpu_memory override requested ({gpu_memory}) but is only advisory in this version")

    return ResourcePlan(cpu_threads=threads, use_gpu=use_gpu, memory_budget_mb=memory_budget_mb, notes=notes)


def _resolve_cpu_threads(hardware: HardwareProfile, cpu_threads: str | int, notes: list[str]) -> int:
    if cpu_threads == "auto":
        threads = max(1, hardware.cpu.physical_cores - 1)
        notes.append(f"cpu_threads=auto -> {threads} (physical_cores-1, leaving one core free)")
        return threads
    try:
        threads = int(cpu_threads)
    except (TypeError, ValueError) as exc:
        raise ConfigurationError(f"cpu_threads must be 'auto' or an integer, got {cpu_threads!r}") from exc
    if threads < 1:
        raise ConfigurationError("cpu_threads must be >= 1")
    return threads


def _resolve_device(
    hardware: HardwareProfile, dataset_profile: DatasetProfile, device: str, notes: list[str]
) -> bool:
    if device == "cpu":
        return False
    if device == "gpu":
        if not hardware.gpu.available:
            notes.append("device='gpu' requested but no GPU was detected; will fall back to CPU at train time")
        return True
    if device != "auto":
        raise ConfigurationError(f"device must be 'auto', 'cpu', or 'gpu', got {device!r}")

    if not hardware.gpu.available:
        notes.append("device=auto -> cpu (no GPU detected)")
        return False
    if hardware.gpu.vram_free_mb < MIN_GPU_FREE_VRAM_MB:
        notes.append(
            f"device=auto -> cpu (GPU free VRAM {hardware.gpu.vram_free_mb:.0f}MB < {MIN_GPU_FREE_VRAM_MB}MB)"
        )
        return False
    if dataset_profile.n_samples < MIN_SAMPLES_FOR_GPU_BENEFIT:
        notes.append(
            f"device=auto -> cpu (dataset has {dataset_profile.n_samples} rows, "
            f"too small to benefit from GPU overhead)"
        )
        return False
    notes.append(f"device=auto -> gpu ({hardware.gpu.name})")
    return True


def _resolve_ram_limit(hardware: HardwareProfile, ram_limit: str | float, notes: list[str]) -> float:
    if ram_limit == "auto":
        budget = hardware.ram.available_mb * (1 - MIN_FREE_RAM_FRACTION)
        notes.append(f"ram_limit=auto -> {budget:.0f}MB (keeping >={MIN_FREE_RAM_FRACTION:.0%} RAM free)")
        return budget
    try:
        value = float(ram_limit)
    except (TypeError, ValueError) as exc:
        raise ConfigurationError(f"ram_limit must be 'auto' or a number, got {ram_limit!r}") from exc
    if 0 < value <= 1:
        return hardware.ram.total_mb * value
    if value > 1:
        return value
    raise ConfigurationError("ram_limit must be a fraction in (0, 1] or an absolute MB value > 1")
