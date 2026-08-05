from __future__ import annotations

import numpy as np
import pandas as pd

from smarttab.multimodal.text import TextFeatureExtractor


def test_text_features_are_bounded_deterministic_and_richer_than_length():
    values = pd.Series([
        "سلام دنیا! این یک متن فارسی است.",
        "Hello world! Visit https://example.com now.",
        "aaaaa 123 123 test@example.com",
        "短い日本語の文章です。",
        "",
        "Mixed فارسی English 42!!!",
    ])
    extractor = TextFeatureExtractor(max_features=48, backend="classical", random_state=7)
    first = extractor.fit_transform(values)
    second = extractor.transform(values)

    assert first.shape[1] <= 48
    assert first.shape == second.shape
    np.testing.assert_allclose(first.to_numpy(), second.to_numpy(), rtol=1e-5, atol=1e-6)
    assert any("char_entropy" in name for name in first.columns)
    assert any("arabic_ratio" in name for name in first.columns)
    assert any("lsa_" in name for name in first.columns)


def test_text_vocabulary_is_bounded_by_budget_multiplier():
    values = pd.Series([f"token_{i} unique_{i} shared phrase" for i in range(120)])
    extractor = TextFeatureExtractor(
        max_features=40,
        backend="classical",
        max_vocabulary_multiplier=3,
    )
    frame = extractor.fit_transform(values)

    assert frame.shape[1] <= 40
    assert extractor.word_vectorizer_ is not None
    assert extractor.char_vectorizer_ is not None
    total_vocab = len(extractor.word_vectorizer_.vocabulary_) + len(extractor.char_vectorizer_.vocabulary_)
    assert total_vocab <= 240


def test_text_hashing_is_explicitly_bounded_and_replayable():
    values = pd.Series([f"document {i} repeated token class {i % 3}" for i in range(80)])
    extractor = TextFeatureExtractor(
        max_features=52,
        backend="classical",
        vectorizer="hashing",
        random_state=11,
    )
    fitted = extractor.fit_transform(values)
    replayed = extractor.transform(values.iloc[:5])

    assert extractor.lexical_mode_ == "hashing"
    assert fitted.shape[1] <= 52
    assert any(name.startswith("hash_lsa_") for name in fitted.columns)
    np.testing.assert_allclose(fitted.iloc[:5], replayed, rtol=1e-5, atol=1e-6)


def test_text_bytes_paths_and_very_long_documents_are_bounded(tmp_path):
    path = tmp_path / "sample.txt"
    path.write_text("آغاز " + ("middle-token " * 5000) + " پایان", encoding="utf-8")
    values = pd.Series([b"hello bytes", path, "x" * 100_000], dtype=object)
    extractor = TextFeatureExtractor(
        max_features=40,
        backend="classical",
        max_chars=1200,
        input_mode="auto",
    )
    frame = extractor.fit_transform(values)

    normalized = extractor._normalize(values)
    assert frame.shape[1] <= 40
    assert all(len(text) <= 1200 for text in normalized)
    assert normalized[0] == "hello bytes"
    assert "آغاز" in normalized[1] and "پایان" in normalized[1]


def test_hybrid_text_without_download_falls_back_to_classical():
    values = pd.Series(["alpha beta gamma", "delta epsilon", "سلام دنیا", "more text"])
    extractor = TextFeatureExtractor(
        max_features=96,
        backend="hybrid",
        allow_model_download=False,
        random_state=3,
    )
    fitted = extractor.fit_transform(values)
    replayed = extractor.transform(values.iloc[:2])

    assert extractor.backend_used_ == "classical"
    assert extractor.embedding_names_ == []
    assert any("unavailable" in note for note in extractor.notes_)
    assert fitted.shape[1] <= 96
    np.testing.assert_allclose(fitted.iloc[:2], replayed, rtol=1e-5, atol=1e-6)
