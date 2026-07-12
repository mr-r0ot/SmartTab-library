"""Stage 3 (part 1) — Hardware Analyzer.

Detects CPU/RAM/GPU/disk characteristics of the current machine. Every
detection call degrades gracefully: if a library or device isn't available,
the corresponding info object reports ``available=False`` / ``"unknown"``
instead of raising, since hardware profiling must never be the reason
``fit()`` fails.
"""

from __future__ import annotations

import subprocess
import sys
from dataclasses import dataclass, field

import psutil

from smarttab.logging_utils import get_logger

logger = get_logger()


@dataclass
class CPUInfo:
    physical_cores: int
    logical_cores: int
    brand: str = "unknown"
    l3_cache_bytes: int | None = None
    avx: bool = False
    avx2: bool = False
    avx512: bool = False


@dataclass
class RAMInfo:
    total_mb: float
    available_mb: float


@dataclass
class GPUInfo:
    available: bool = False
    name: str = "none"
    vram_total_mb: float = 0.0
    vram_free_mb: float = 0.0
    compute_capability: str | None = None
    driver_version: str | None = None


@dataclass
class DiskInfo:
    kind: str = "unknown"  # "ssd" | "hdd" | "unknown"


@dataclass
class HardwareProfile:
    cpu: CPUInfo
    ram: RAMInfo
    gpu: GPUInfo
    disk: DiskInfo = field(default_factory=DiskInfo)


def profile_hardware() -> HardwareProfile:
    return HardwareProfile(
        cpu=_profile_cpu(),
        ram=_profile_ram(),
        gpu=_profile_gpu(),
        disk=_profile_disk(),
    )


def _profile_cpu() -> CPUInfo:
    physical = psutil.cpu_count(logical=False) or 1
    logical = psutil.cpu_count(logical=True) or physical

    brand = "unknown"
    l3_cache = None
    avx = avx2 = avx512 = False
    try:
        import cpuinfo  # py-cpuinfo

        info = cpuinfo.get_cpu_info()
        brand = info.get("brand_raw", "unknown")
        l3_cache = info.get("l3_cache_size")
        flags = set(info.get("flags", []))
        avx = "avx" in flags
        avx2 = "avx2" in flags
        avx512 = any(f.startswith("avx512") for f in flags)
    except Exception as exc:  # pragma: no cover - defensive
        logger.debug("CPU flag detection unavailable: %s", exc)

    return CPUInfo(
        physical_cores=physical,
        logical_cores=logical,
        brand=brand,
        l3_cache_bytes=l3_cache,
        avx=avx,
        avx2=avx2,
        avx512=avx512,
    )


def _profile_ram() -> RAMInfo:
    vm = psutil.virtual_memory()
    return RAMInfo(total_mb=vm.total / (1024 * 1024), available_mb=vm.available / (1024 * 1024))


def _profile_gpu() -> GPUInfo:
    try:
        import pynvml

        pynvml.nvmlInit()
        try:
            if pynvml.nvmlDeviceGetCount() == 0:
                return GPUInfo(available=False)
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            name = pynvml.nvmlDeviceGetName(handle)
            if isinstance(name, bytes):
                name = name.decode()
            mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
            try:
                major, minor = pynvml.nvmlDeviceGetCudaComputeCapability(handle)
                compute_capability = f"{major}.{minor}"
            except Exception:
                compute_capability = None
            try:
                driver_version = pynvml.nvmlSystemGetDriverVersion()
                if isinstance(driver_version, bytes):
                    driver_version = driver_version.decode()
            except Exception:
                driver_version = None
            return GPUInfo(
                available=True,
                name=name,
                vram_total_mb=mem.total / (1024 * 1024),
                vram_free_mb=mem.free / (1024 * 1024),
                compute_capability=compute_capability,
                driver_version=driver_version,
            )
        finally:
            pynvml.nvmlShutdown()
    except Exception as exc:
        logger.debug("No usable GPU detected: %s", exc)
        return GPUInfo(available=False)


def _profile_disk() -> DiskInfo:
    try:
        if sys.platform.startswith("win"):
            out = subprocess.run(
                ["powershell", "-NoProfile", "-Command",
                 "(Get-PhysicalDisk | Select-Object -First 1 -ExpandProperty MediaType)"],
                capture_output=True, text=True, timeout=5,
            )
            media = out.stdout.strip().lower()
            if "ssd" in media:
                return DiskInfo(kind="ssd")
            if "hdd" in media:
                return DiskInfo(kind="hdd")
            return DiskInfo(kind="unknown")
        else:
            with open("/sys/block/sda/queue/rotational") as f:
                return DiskInfo(kind="hdd" if f.read().strip() == "1" else "ssd")
    except Exception as exc:
        logger.debug("Disk type detection unavailable: %s", exc)
        return DiskInfo(kind="unknown")
