"""Stage 3 (part 2) — turns a HardwareProfile + dataset size into concrete resource decisions."""

from __future__ import annotations

from dataclasses import dataclass, field

from smarttab.analysis.dataset_analyzer import DatasetProfile
from smarttab.exceptions import ConfigurationError
from smarttab.hardware.profiler import HardwareProfile

MIN_FREE_RAM_FRACTION = 0.15  # always leave >=15% RAM free
MIN_GPU_FREE_VRAM_MB = 1024
MIN_MODEL_MEMORY_BUDGET_MB = 128
MIN_SAMPLES_FOR_GPU_BENEFIT = 5000


@dataclass
class ResourcePlan:
    cpu_threads: int
    use_gpu: bool
    memory_budget_mb: float
    gpu_memory_budget_mb: float = 0.0
    gpu_ram_part: float = 0.0
    notes: list[str] = field(default_factory=list)


def resolve_resource_plan(
    hardware: HardwareProfile,
    dataset_profile: DatasetProfile,
    device: str = "auto",
    cpu_threads: str | int = "auto",
    gpu_memory: str | float = "auto",
    ram_limit: str | float = "auto",
) -> ResourcePlan:
    notes: list[str] = []

    threads = _resolve_cpu_threads(hardware, cpu_threads, notes)
    use_gpu = _resolve_device(hardware, dataset_profile, device, notes)
    memory_budget_mb = _resolve_ram_limit(hardware, ram_limit, notes)
    gpu_memory_budget_mb, gpu_ram_part = _resolve_gpu_memory(
        hardware, use_gpu, gpu_memory, notes
    )

    return ResourcePlan(
        cpu_threads=threads,
        use_gpu=use_gpu,
        memory_budget_mb=memory_budget_mb,
        gpu_memory_budget_mb=gpu_memory_budget_mb,
        gpu_ram_part=gpu_ram_part,
        notes=notes,
    )


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
        if budget < MIN_MODEL_MEMORY_BUDGET_MB:
            raise ConfigurationError(
                f"only {hardware.ram.available_mb:.1f}MB RAM is available; "
                f"SmartTab requires at least {MIN_MODEL_MEMORY_BUDGET_MB}MB"
            )
        notes.append(
            f"ram_limit=auto -> estimator budget {budget:.0f}MB "
            f"(keeping >={MIN_FREE_RAM_FRACTION:.0%} RAM free)"
        )
        return budget
    try:
        value = float(ram_limit)
    except (TypeError, ValueError) as exc:
        raise ConfigurationError(f"ram_limit must be 'auto' or a number, got {ram_limit!r}") from exc
    if 0 < value <= 1:
        budget = hardware.ram.total_mb * value
    elif value > 1:
        budget = value
    else:
        raise ConfigurationError(
            "ram_limit must be a fraction in (0, 1] or an absolute MB value > 1"
        )
    if budget < MIN_MODEL_MEMORY_BUDGET_MB:
        raise ConfigurationError(
            f"resolved ram_limit is {budget:.1f}MB; at least {MIN_MODEL_MEMORY_BUDGET_MB}MB is required"
        )
    notes.append(f"ram_limit -> estimator budget {budget:.0f}MB")
    return budget


def _resolve_gpu_memory(
    hardware: HardwareProfile,
    use_gpu: bool,
    gpu_memory: str | float,
    notes: list[str],
) -> tuple[float, float]:
    if not use_gpu or not hardware.gpu.available:
        if gpu_memory != "auto":
            notes.append("gpu_memory was specified but GPU execution is disabled")
        return 0.0, 0.0

    total = hardware.gpu.vram_total_mb or hardware.gpu.vram_free_mb
    free = hardware.gpu.vram_free_mb
    if gpu_memory == "auto":
        budget = free * 0.85
    else:
        try:
            value = float(gpu_memory)
        except (TypeError, ValueError) as exc:
            raise ConfigurationError(
                "gpu_memory must be 'auto', a fraction in (0, 1], or an absolute MB value"
            ) from exc
        if 0 < value <= 1:
            budget = total * value
        elif value > 1:
            budget = value
        else:
            raise ConfigurationError(
                "gpu_memory must be 'auto', a fraction in (0, 1], or an absolute MB value"
            )
    if budget > free:
        raise ConfigurationError(
            f"gpu_memory resolves to {budget:.0f}MB but only {free:.0f}MB VRAM is free"
        )
    if budget < 256:
        raise ConfigurationError("GPU training requires a gpu_memory budget of at least 256MB")
    part = min(0.95, max(0.05, budget / max(total, 1.0)))
    notes.append(
        f"gpu_memory -> {budget:.0f}MB admission budget; CatBoost gpu_ram_part={part:.3f}"
    )
    return budget, part
