"""Stage 3 (part 1) — Hardware Analyzer.

Detects CPU/RAM/GPU/disk characteristics of the current machine. Every
detection call degrades gracefully: if a library or device isn't available,
the corresponding info object reports ``available=False`` / ``"unknown"``
instead of raising, since hardware profiling must never be the reason
``fit()`` fails.
"""

from __future__ import annotations

import platform
import re
import subprocess
import sys
from pathlib import Path
from functools import lru_cache
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


@lru_cache(maxsize=1)
def _profile_cpu() -> CPUInfo:
    """Profile CPU without invoking py-cpuinfo's subprocess probe.

    Native learner thread pools and fork-based CPU probes are a poor combination.
    Linux exposes the required information directly; other platforms degrade to
    stable platform/psutil metadata rather than risking a blocked fit.
    """
    physical = psutil.cpu_count(logical=False) or 1
    logical = psutil.cpu_count(logical=True) or physical
    brand = platform.processor() or platform.machine() or "unknown"
    flags: set[str] = set()
    l3_cache: int | None = None
    try:
        if sys.platform.startswith("linux"):
            cpuinfo_path = Path("/proc/cpuinfo")
            if cpuinfo_path.exists():
                text = cpuinfo_path.read_text(encoding="utf-8", errors="ignore")
                brand_match = re.search(r"^(?:model name|Hardware)\s*:\s*(.+)$", text, re.MULTILINE)
                if brand_match:
                    brand = brand_match.group(1).strip()
                flag_match = re.search(r"^(?:flags|Features)\s*:\s*(.+)$", text, re.MULTILINE)
                if flag_match:
                    flags = set(flag_match.group(1).lower().split())
            cache_paths = [
                Path("/sys/devices/system/cpu/cpu0/cache/index3/size"),
                Path("/sys/devices/system/cpu/cpu0/cache/index2/size"),
            ]
            for cache_path in cache_paths:
                if cache_path.exists():
                    l3_cache = _parse_cache_size(cache_path.read_text().strip())
                    if l3_cache:
                        break
        elif sys.platform == "darwin":
            result = subprocess.run(
                ["sysctl", "-n", "machdep.cpu.brand_string"],
                capture_output=True, text=True, timeout=2, check=False,
            )
            if result.stdout.strip():
                brand = result.stdout.strip()
    except Exception as exc:  # pragma: no cover - platform defensive path
        logger.debug("CPU detail detection unavailable: %s", exc)

    return CPUInfo(
        physical_cores=physical,
        logical_cores=logical,
        brand=brand,
        l3_cache_bytes=l3_cache,
        avx="avx" in flags,
        avx2="avx2" in flags,
        avx512=any(flag.startswith("avx512") for flag in flags),
    )


def _parse_cache_size(value: str) -> int | None:
    match = re.fullmatch(r"\s*(\d+(?:\.\d+)?)\s*([KMG]?)B?\s*", value, re.IGNORECASE)
    if not match:
        return None
    number = float(match.group(1))
    multiplier = {"": 1, "K": 1024, "M": 1024**2, "G": 1024**3}[match.group(2).upper()]
    return int(number * multiplier)


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


@lru_cache(maxsize=1)
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
