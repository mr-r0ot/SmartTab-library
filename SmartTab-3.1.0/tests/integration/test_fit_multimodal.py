from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import smarttab


COMMON = dict(
    model="lightgbm",
    optimize=False,
    n_trials=0,
    n_estimators=25,
    report=False,
    explain=False,
    verbose=0,
    duplicate_policy="keep",
    feature_budget=64,
    speed_accuracy=0.2,
    random_state=4,
)


def test_direct_text_api_prediction_feature_transform_and_roundtrip(tmp_path):
    texts = [
        ("excellent durable useful product " if i % 2 else "broken poor useless product ") + f"sample {i}"
        for i in range(48)
    ]
    labels = [i % 2 for i in range(48)]
    model = smarttab.fit_text(texts, labels, **COMMON)

    assert model.dataset_profile.text_columns == ["input"]
    assert model.feature_space["generated_features"] <= 64
    assert model.predict(["excellent durable", "broken useless"]).shape == (2,)
    assert model.predict("excellent useful").single
    assert model.transform_features(["one text"]).shape[1] == len(model.feature_names)

    bundle = model.save(tmp_path / "text.smarttab")
    loaded = smarttab.load(bundle, trusted=True)
    np.testing.assert_array_equal(model.predict(texts[:5]), loaded.predict(texts[:5]))


def test_mixed_tabular_text_image_dataframe():
    rows = 42
    images = []
    for i in range(rows):
        image = np.zeros((18, 18, 3), dtype=np.uint8)
        image[..., i % 2] = 150 + i
        images.append(image)
    frame = pd.DataFrame({
        "age": np.arange(rows) + 20,
        "review": [("positive detailed review " if i % 2 else "negative damaged review ") + str(i) for i in range(rows)],
        "photo": images,
        "label": [i % 2 for i in range(rows)],
    })
    model = smarttab.fit(
        frame,
        target="label",
        modalities={"review": "text", "photo": "image"},
        feature_budget={"total": 96, "review": 56, "photo": 40},
        **{key: value for key, value in COMMON.items() if key != "feature_budget"},
    )

    assert model.dataset_profile.column_modalities == {"review": "text", "photo": "image"}
    assert set(model.feature_space["modalities"]) == {"review", "photo"}
    prediction = model.predict(frame.drop(columns="label").iloc[:3])
    assert prediction.shape == (3,)


@pytest.mark.parametrize("task_type", ["regression", "multiclass", "multilabel", "multioutput_regression"])
def test_text_modality_supports_shared_task_engine(task_type):
    texts = [f"document {i} category {i % 3} signal {i * i}" for i in range(45)]
    if task_type == "regression":
        y = np.asarray([i * 0.25 for i in range(45)])
    elif task_type == "multiclass":
        y = np.asarray([f"class-{i % 3}" for i in range(45)])
    elif task_type == "multilabel":
        y = np.column_stack([[i % 2 for i in range(45)], [(i // 2) % 2 for i in range(45)]])
    else:
        y = np.column_stack([[i * 0.1 for i in range(45)], [np.sin(i) for i in range(45)]])
    model = smarttab.fit_text(texts, y, task_type=task_type, **COMMON)
    predictions = model.predict(texts[:4])
    expected_shape = (4, 2) if task_type in {"multilabel", "multioutput_regression"} else (4,)
    assert predictions.shape == expected_shape


def test_image_audio_and_video_direct_apis():
    count = 24
    images = []
    audio = []
    videos = []
    sample_rate = 4000
    for i in range(count):
        image = np.zeros((16, 16, 3), dtype=np.uint8)
        image[..., i % 2] = 100 + i
        images.append(image)
        time = np.arange(sample_rate // 8) / sample_rate
        audio.append((sample_rate, (0.25 * np.sin(2 * np.pi * (180 + 70 * (i % 2) + i) * time)).astype(np.float32)))
        frames = []
        for j in range(4):
            frame = np.zeros((14, 14, 3), dtype=np.uint8)
            frame[j : j + 2, :, i % 2] = 180 + i
            frames.append(frame)
        videos.append(np.stack(frames))
    labels = [i % 2 for i in range(count)]

    image_model = smarttab.fit_images(images, labels, **COMMON)
    audio_model = smarttab.fit_audio(audio, labels, **COMMON)
    video_model = smarttab.fit_videos(videos, labels, **COMMON)

    assert image_model.predict(images[:2]).shape == (2,)
    assert audio_model.predict(audio[:2]).shape == (2,)
    assert video_model.predict(videos[:2]).shape == (2,)
    assert audio_model.predict({"waveform": audio[0][1], "sample_rate": sample_rate}).single
    assert video_model.predict((12.0, videos[0])).single


def test_explicit_feature_space_object_controls_fit():
    config = smarttab.FeatureSpaceConfig(
        total_features=48,
        modality_limits={"text": 48},
        speed_accuracy=0.3,
        backend="classical",
        batch_size=8,
        workers=1,
    )
    texts = [f"bounded document {index} class {index % 2}" for index in range(36)]
    labels = [index % 2 for index in range(36)]
    model = smarttab.fit_text(texts, labels, feature_budget=config, **{
        key: value for key, value in COMMON.items() if key not in {"feature_budget", "speed_accuracy"}
    })

    assert model.feature_space["feature_budget"] == 48
    assert model.feature_space["generated_features"] <= 48


def _media_samples(modality: str, count: int = 30):
    if modality == "image":
        samples = []
        for index in range(count):
            image = np.zeros((12, 14, 3), dtype=np.uint8)
            image[..., index % 3] = 80 + index
            image[index % 6 : index % 6 + 2, :, :] += 20
            samples.append(image)
        return samples
    if modality == "audio":
        sample_rate = 3000
        samples = []
        for index in range(count):
            time = np.arange(sample_rate // 12) / sample_rate
            wave = 0.2 * np.sin(2 * np.pi * (120 + 35 * (index % 3) + index) * time)
            samples.append((sample_rate, wave.astype(np.float32)))
        return samples
    if modality == "video":
        samples = []
        for index in range(count):
            frames = []
            for step in range(3):
                frame = np.zeros((10, 12, 3), dtype=np.uint8)
                frame[:, (step + index) % 8 : (step + index) % 8 + 2, index % 3] = 120 + index
                frames.append(frame)
            samples.append(np.stack(frames))
        return samples
    raise AssertionError(modality)


@pytest.mark.parametrize("modality", ["image", "audio", "video"])
@pytest.mark.parametrize("task_type", ["regression", "multiclass", "multilabel", "multioutput_regression"])
def test_every_media_modality_uses_the_full_task_engine(modality, task_type):
    count = 30
    samples = _media_samples(modality, count)
    if task_type == "regression":
        y = np.asarray([index * 0.15 + (index % 3) for index in range(count)])
    elif task_type == "multiclass":
        y = np.asarray([f"class-{index % 3}" for index in range(count)])
    elif task_type == "multilabel":
        y = np.column_stack(
            [[index % 2 for index in range(count)], [(index // 2) % 2 for index in range(count)]]
        )
    else:
        y = np.column_stack(
            [[index * 0.1 for index in range(count)], [np.cos(index / 3) for index in range(count)]]
        )
    model = smarttab.fit(
        samples,
        y=y,
        modality=modality,
        task_type=task_type,
        **{
            **COMMON,
            "n_estimators": 10,
            "feature_budget": 40,
        },
    )
    predictions = model.predict(samples[:3])
    expected = (3, 2) if task_type in {"multilabel", "multioutput_regression"} else (3,)
    assert predictions.shape == expected


def test_text_can_participate_in_group_ranking():
    rows = 40
    frame = pd.DataFrame(
        {
            "query_id": np.repeat(np.arange(8), 5),
            "document": [f"query document relevance signal {index % 5} row {index}" for index in range(rows)],
            "relevance": np.tile([0, 1, 2, 1, 0], 8),
        }
    )
    model = smarttab.fit(
        frame,
        target="relevance",
        group_id="query_id",
        task_type="ranking",
        modalities={"document": "text"},
        model="lightgbm",
        optimize=False,
        n_trials=0,
        n_estimators=10,
        report=False,
        explain=False,
        verbose=0,
        feature_budget=40,
        speed_accuracy=0.1,
        duplicate_policy="keep",
    )
    assert model.predict(frame.drop(columns=["relevance", "query_id"]).iloc[:5]).shape == (5,)


def test_hybrid_fusion_plan_survives_bundle_roundtrip(tmp_path):
    rows = 48
    frame = pd.DataFrame(
        {
            "numeric": np.arange(rows) % 7,
            "review": [
                ("excellent reliable " if index % 2 else "broken unreliable ") + str(index)
                for index in range(rows)
            ],
            "label": [index % 2 for index in range(rows)],
        }
    )
    model = smarttab.fit(
        frame,
        target="label",
        modalities={"review": "text"},
        ensemble="voting",
        fusion="hybrid",
        ensemble_models_limit=3,
        model="auto",
        optimize=False,
        n_trials=0,
        n_estimators=8,
        cv=2,
        feature_budget=36,
        speed_accuracy=0.1,
        report=False,
        explain=False,
        verbose=0,
        duplicate_policy="keep",
    )

    assert model.ensemble_info["fusion_strategy"] == "hybrid"
    assert any(
        candidate.get("feature_group") == "modality:text"
        for candidate in model.ensemble_info["candidates"]
    )
    path = model.save(tmp_path / "hybrid.smarttab")
    loaded = smarttab.load(path, trusted=True)
    assert loaded.ensemble_info["fusion_strategy"] == "hybrid"
    assert any(
        candidate.get("feature_group") == "modality:text"
        for candidate in loaded.ensemble_info["candidates"]
    )
    expected = model.predict(frame.drop(columns="label").iloc[:6])
    actual = loaded.predict(frame.drop(columns="label").iloc[:6])
    np.testing.assert_array_equal(expected, actual)
