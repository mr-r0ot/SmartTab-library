import numpy as np
import pandas as pd

from smarttab.analysis.dataset_analyzer import TaskType
from smarttab.multimodal.adaptation import SupervisedFeatureAdapter
from smarttab.multimodal.config import FeatureSpaceConfig
from smarttab.multimodal.pipeline import MultiModalFeaturePipeline


def test_supervised_adapter_is_bounded_and_replayable():
    rng = np.random.default_rng(12)
    frame = pd.DataFrame(rng.normal(size=(180, 20)))
    y = pd.Series((frame[0] + frame[1] > 0).astype(int))
    adapter = SupervisedFeatureAdapter(TaskType.BINARY, n_components=7).fit(frame, y)
    assert adapter.fitted
    transformed = adapter.transform(frame.head(11))
    assert transformed.shape == (11, 7)
    assert np.isfinite(transformed.to_numpy()).all()


def test_multimodal_pipeline_reports_supervised_adaptation():
    values = pd.Series([
        f"space orbit satellite launch mission {i}" if i % 2 else f"baseball pitcher inning game score {i}"
        for i in range(140)
    ])
    y = pd.Series(np.arange(140) % 2)
    config = FeatureSpaceConfig(
        total_features=48,
        column_limits={"text": 48},
        backend="classical",
        speed_accuracy=0.8,
        supervised_adaptation="pls",
        adapter_features=6,
    )
    pipeline = MultiModalFeaturePipeline(config, {"text": "text"}, TaskType.BINARY)
    fitted = pipeline.fit_transform(pd.DataFrame({"text": values}), y)
    replay = pipeline.transform(pd.DataFrame({"text": values.head(5)}))
    assert fitted.shape[1] <= 48
    assert replay.shape[1] == fitted.shape[1]
    assert pipeline.report_.adapted_features["text"] > 0
