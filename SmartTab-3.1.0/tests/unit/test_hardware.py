import pytest

from smarttab.analysis.dataset_analyzer import DatasetProfile, TaskType
from smarttab.exceptions import ConfigurationError
from smarttab.hardware.profiler import CPUInfo, DiskInfo, GPUInfo, HardwareProfile, RAMInfo, profile_hardware
from smarttab.hardware.resource_planner import resolve_resource_plan


def _profile(n_samples=1000):
    return DatasetProfile(
        task_type=TaskType.BINARY,
        target_column="target",
        n_samples=n_samples,
        n_features=5,
        feature_columns=["a", "b", "c", "d", "e"],
    )


def _hardware(gpu_available=False, vram_free_mb=0.0, physical_cores=4, available_ram_mb=8000.0):
    return HardwareProfile(
        cpu=CPUInfo(physical_cores=physical_cores, logical_cores=physical_cores * 2),
        ram=RAMInfo(total_mb=16000.0, available_mb=available_ram_mb),
        gpu=GPUInfo(available=gpu_available, name="fake-gpu" if gpu_available else "none", vram_free_mb=vram_free_mb),
        disk=DiskInfo(kind="unknown"),
    )


def test_profile_hardware_runs_on_real_machine():
    profile = profile_hardware()
    assert profile.cpu.physical_cores >= 1
    assert profile.cpu.logical_cores >= profile.cpu.physical_cores
    assert profile.ram.total_mb > 0
    assert isinstance(profile.gpu.available, bool)


def test_cpu_threads_auto_leaves_one_core_free():
    plan = resolve_resource_plan(_hardware(physical_cores=4), _profile(), cpu_threads="auto")
    assert plan.cpu_threads == 3


def test_cpu_threads_explicit_override():
    plan = resolve_resource_plan(_hardware(), _profile(), cpu_threads=2)
    assert plan.cpu_threads == 2


def test_cpu_threads_invalid_raises():
    with pytest.raises(ConfigurationError):
        resolve_resource_plan(_hardware(), _profile(), cpu_threads="not-a-number")


def test_device_auto_is_cpu_when_no_gpu():
    plan = resolve_resource_plan(_hardware(gpu_available=False), _profile(), device="auto")
    assert plan.use_gpu is False


def test_device_auto_is_cpu_when_dataset_too_small_for_gpu():
    plan = resolve_resource_plan(
        _hardware(gpu_available=True, vram_free_mb=4000), _profile(n_samples=100), device="auto"
    )
    assert plan.use_gpu is False


def test_device_auto_uses_gpu_when_available_and_dataset_large_enough():
    plan = resolve_resource_plan(
        _hardware(gpu_available=True, vram_free_mb=4000), _profile(n_samples=50000), device="auto"
    )
    assert plan.use_gpu is True


def test_device_explicit_cpu_forces_cpu_even_with_gpu_present():
    plan = resolve_resource_plan(_hardware(gpu_available=True, vram_free_mb=4000), _profile(n_samples=50000), device="cpu")
    assert plan.use_gpu is False


def test_ram_limit_auto_leaves_headroom():
    hw = _hardware(available_ram_mb=10000.0)
    plan = resolve_resource_plan(hw, _profile(), ram_limit="auto")
    assert plan.memory_budget_mb < 10000.0
    assert plan.memory_budget_mb == pytest.approx(8500.0)


def test_ram_limit_fraction_override():
    hw = _hardware()
    plan = resolve_resource_plan(hw, _profile(), ram_limit=0.5)
    assert plan.memory_budget_mb == pytest.approx(hw.ram.total_mb * 0.5)


def test_static_hardware_probes_are_cached():
    import smarttab.hardware.profiler as profiler

    profiler._profile_cpu.cache_clear()
    first = profiler._profile_cpu()
    second = profiler._profile_cpu()
    assert first is second
    assert first.physical_cores >= 1
    # CPU probing is deliberately subprocess-free; repeated fit() calls must not
    # reintroduce the py-cpuinfo deadlock that this cache was created to prevent.
    assert "cpuinfo" not in profiler._profile_cpu.__wrapped__.__code__.co_names
    profiler._profile_cpu.cache_clear()


def test_gpu_memory_budget_is_resolved_and_impossible_request_rejected():
    hw = _hardware(gpu_available=True, vram_free_mb=4000)
    hw.gpu.vram_total_mb = 8000
    plan = resolve_resource_plan(
        hw,
        _profile(n_samples=50000),
        device="gpu",
        gpu_memory=0.25,
    )
    assert plan.gpu_memory_budget_mb == pytest.approx(2000)
    assert plan.gpu_ram_part == pytest.approx(0.25)
    with pytest.raises(ConfigurationError, match="only 4000MB VRAM is free"):
        resolve_resource_plan(
            hw,
            _profile(n_samples=50000),
            device="gpu",
            gpu_memory=6000,
        )


def test_too_small_ram_budget_is_rejected():
    with pytest.raises(ConfigurationError, match="at least 128MB"):
        resolve_resource_plan(_hardware(), _profile(), ram_limit=64)
