"""Resource-plan settings must reach the native boosting estimators."""

import pytest

from smarttab.analysis.dataset_analyzer import TaskType
from smarttab.exceptions import ConfigurationError
from smarttab.hardware.resource_planner import ResourcePlan
from smarttab.training.trainer import build_estimator


def _plan(*, gpu: bool = False) -> ResourcePlan:
    return ResourcePlan(
        cpu_threads=2,
        use_gpu=gpu,
        memory_budget_mb=1024,
        gpu_memory_budget_mb=2048 if gpu else 0,
        gpu_ram_part=0.25 if gpu else 0,
    )


def test_catboost_receives_ram_budget():
    estimator = build_estimator(
        "catboost",
        {},
        TaskType.BINARY,
        n_estimators=20,
        cpu_threads=2,
        use_gpu=False,
        resource_plan=_plan(),
    )
    params = estimator.get_params()
    assert params["used_ram_limit"] == "1024mb"


def test_catboost_receives_gpu_fraction_without_training_gpu():
    estimator = build_estimator(
        "catboost",
        {},
        TaskType.BINARY,
        n_estimators=20,
        cpu_threads=2,
        use_gpu=True,
        resource_plan=_plan(gpu=True),
    )
    params = estimator.get_params()
    assert params["gpu_ram_part"] == 0.25
    assert params["task_type"] == "GPU"


def test_lightgbm_receives_histogram_cache_budget():
    estimator = build_estimator(
        "lightgbm",
        {},
        TaskType.BINARY,
        n_estimators=20,
        cpu_threads=2,
        use_gpu=False,
        resource_plan=_plan(),
    )
    params = estimator.get_params()
    assert params["histogram_pool_size"] == 512


def test_unknown_lightgbm_parameter_is_rejected():
    with pytest.raises(ConfigurationError, match="unknown LightGBM params"):
        build_estimator(
            "lightgbm",
            {"num_leave_typo": 31},
            TaskType.BINARY,
            n_estimators=20,
            cpu_threads=2,
            use_gpu=False,
            resource_plan=_plan(),
        )
