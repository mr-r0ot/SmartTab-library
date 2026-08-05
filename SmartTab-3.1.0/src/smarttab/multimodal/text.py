"""Bounded linguistic, lexical-semantic, and optional neural text features."""

from __future__ import annotations

import importlib.util
import math
import re
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.feature_extraction.text import HashingVectorizer, TfidfVectorizer

from smarttab.multimodal.base import BaseFeatureExtractor
from smarttab.multimodal.common import safe_float32_frame

_URL_RE = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
_EMAIL_RE = re.compile(r"\b[^\s@]+@[^\s@]+\.[^\s@]+\b")
_WORD_RE = re.compile(r"\b\w+\b", re.UNICODE)
_SENTENCE_RE = re.compile(r"[.!?؟。！？]+")
_REPEATED_RE = re.compile(r"(.)\1{2,}", re.UNICODE)


class TextFeatureExtractor(BaseFeatureExtractor):
    modality = "text"

    def __init__(
        self,
        *,
        max_features: int,
        backend: str = "auto",
        speed_accuracy: float = 0.5,
        allow_model_download: bool = False,
        batch_size: int = 32,
        random_state: int = 42,
        error_policy: str = "warn",
        device: str = "auto",
        model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        lowercase: bool = True,
        word_ngram_range: tuple[int, int] = (1, 2),
        char_ngram_range: tuple[int, int] = (3, 5),
        min_df: int | float | str = "auto",
        max_df: float = 0.995,
        sublinear_tf: bool = True,
        max_vocabulary_multiplier: int = 6,
        vectorizer: str = "auto",
        max_svd_fit_rows: int | str = "auto",
        max_chars: int | str = "auto",
        input_mode: str = "auto",
        encoding: str = "utf-8",
        **_: Any,
    ) -> None:
        super().__init__(max_features=max_features, error_policy=error_policy)
        self.backend = backend
        self.speed_accuracy = float(speed_accuracy)
        self.allow_model_download = allow_model_download
        self.batch_size = int(batch_size)
        self.random_state = int(random_state)
        self.device = device
        self.model_name = model_name
        self.lowercase = lowercase
        self.word_ngram_range = tuple(word_ngram_range)
        self.char_ngram_range = tuple(char_ngram_range)
        self.min_df = min_df
        self.max_df = float(max_df)
        self.sublinear_tf = bool(sublinear_tf)
        self.max_vocabulary_multiplier = int(max_vocabulary_multiplier)
        if vectorizer not in {"auto", "tfidf", "hashing"}:
            raise ValueError("text vectorizer must be 'auto', 'tfidf', or 'hashing'")
        self.vectorizer = vectorizer
        self.max_svd_fit_rows = int(20_000 + 80_000 * self.speed_accuracy) if max_svd_fit_rows == "auto" else max(100, int(max_svd_fit_rows))
        self.max_chars = int(20_000 + 180_000 * self.speed_accuracy) if max_chars == "auto" else max(256, int(max_chars))
        if input_mode not in {"auto", "text", "path"}:
            raise ValueError("text input_mode must be 'auto', 'text', or 'path'")
        self.input_mode = input_mode
        self.encoding = str(encoding)
        self.lexical_mode_: str | None = None

        self.word_vectorizer_: TfidfVectorizer | None = None
        self.char_vectorizer_: TfidfVectorizer | None = None
        self.svd_: TruncatedSVD | None = None
        self.embedding_reducer_: PCA | None = None
        self._embedding_model: Any = None
        self.backend_used_: str = "classical"
        self.handcrafted_names_: list[str] = []
        self.lexical_names_: list[str] = []
        self.embedding_names_: list[str] = []
        self.notes_: list[str] = []

    def fit_transform(self, values: pd.Series, y: Any = None) -> pd.DataFrame:
        texts = self._normalize(values)
        handcrafted, handcrafted_names = self._handcrafted_matrix(texts)
        self.handcrafted_names_ = handcrafted_names[: self.max_features]
        remaining = max(0, self.max_features - len(self.handcrafted_names_))

        use_embeddings = self._should_use_embeddings()
        deep_budget = 0
        if use_embeddings and remaining >= 16:
            deep_fraction = 0.55 if self.backend == "hybrid" else 0.85
            deep_budget = min(remaining, max(16, int(round(remaining * deep_fraction))))
        lexical_budget = max(0, remaining - deep_budget)
        if self.backend == "pretrained" and deep_budget:
            lexical_budget = 0
            deep_budget = remaining

        parts = [handcrafted[:, : len(self.handcrafted_names_)]]
        names = list(self.handcrafted_names_)

        lexical = self._fit_lexical(texts, lexical_budget)
        if lexical is not None:
            parts.append(lexical)
            names.extend(self.lexical_names_)

        embeddings = self._fit_embeddings(texts, deep_budget) if deep_budget else None
        if embeddings is not None:
            parts.append(embeddings)
            names.extend(self.embedding_names_)
            self.backend_used_ = "hybrid" if lexical is not None else "pretrained"
        else:
            self.backend_used_ = "classical"
            if use_embeddings and deep_budget:
                self.notes_.append("pretrained text embeddings unavailable; classical features used")
                fallback = self._fit_lexical(texts, remaining) if lexical is None else None
                if fallback is not None:
                    parts.append(fallback)
                    names.extend(self.lexical_names_)

        matrix = np.column_stack(parts) if parts else np.empty((len(texts), 0), dtype=np.float32)
        matrix = matrix[:, : self.max_features]
        self.feature_names_ = names[: matrix.shape[1]]
        self._fitted = True
        return safe_float32_frame(matrix, self.feature_names_, values.index)

    def transform(self, values: pd.Series) -> pd.DataFrame:
        if not self._fitted:
            raise RuntimeError("TextFeatureExtractor must be fitted before transform()")
        texts = self._normalize(values)
        handcrafted, _ = self._handcrafted_matrix(texts)
        parts = [handcrafted[:, : len(self.handcrafted_names_)]]
        if self.svd_ is not None and self.word_vectorizer_ is not None and self.char_vectorizer_ is not None:
            word = self.word_vectorizer_.transform(texts)
            char = self.char_vectorizer_.transform(texts)
            parts.append(self.svd_.transform(sparse.hstack([word, char], format="csr")))
        if self.embedding_names_:
            embeddings = self._encode_embeddings(texts)
            if embeddings is None:
                raise RuntimeError(
                    f"pretrained text encoder {self.model_name!r} is unavailable during transform"
                )
            if self.embedding_reducer_ is not None:
                embeddings = self.embedding_reducer_.transform(embeddings)
            parts.append(embeddings[:, : len(self.embedding_names_)])
        matrix = np.column_stack(parts)[:, : len(self.feature_names_)]
        return safe_float32_frame(matrix, self.feature_names_, values.index)

    def _fit_lexical(self, texts: list[str], budget: int) -> np.ndarray | None:
        if budget < 2 or len(texts) < 2 or not any(text.strip() for text in texts):
            return None
        min_df: int | float
        if self.min_df == "auto":
            min_df = 1 if len(texts) < 50 else 2
        else:
            min_df = self.min_df
        vocabulary_cap = max(64, budget * max(2, self.max_vocabulary_multiplier))
        mode = self.vectorizer
        if mode == "auto":
            mode = "hashing" if len(texts) >= 50_000 or self.speed_accuracy <= 0.2 else "tfidf"
        self.lexical_mode_ = mode

        if mode == "hashing":
            word_features = max(32, int(vocabulary_cap * 0.6))
            char_features = max(32, int(vocabulary_cap * 0.4))
            self.word_vectorizer_ = HashingVectorizer(
                lowercase=self.lowercase,
                analyzer="word",
                ngram_range=self.word_ngram_range,
                n_features=word_features,
                alternate_sign=False,
                norm="l2",
                strip_accents="unicode",
                dtype=np.float32,
            )
            self.char_vectorizer_ = HashingVectorizer(
                lowercase=self.lowercase,
                analyzer="char_wb",
                ngram_range=self.char_ngram_range,
                n_features=char_features,
                alternate_sign=False,
                norm="l2",
                dtype=np.float32,
            )
            word = self.word_vectorizer_.transform(texts)
            char = self.char_vectorizer_.transform(texts)
            combined = sparse.hstack([word, char], format="csr")
            self.notes_.append(
                f"bounded hashing vectorization used ({combined.shape[1]} intermediate dimensions)"
            )
        else:
            self.word_vectorizer_ = TfidfVectorizer(
                lowercase=self.lowercase,
                analyzer="word",
                ngram_range=self.word_ngram_range,
                min_df=min_df,
                max_df=self.max_df,
                max_features=max(32, int(vocabulary_cap * 0.6)),
                sublinear_tf=self.sublinear_tf,
                strip_accents="unicode",
                dtype=np.float32,
            )
            self.char_vectorizer_ = TfidfVectorizer(
                lowercase=self.lowercase,
                analyzer="char_wb",
                ngram_range=self.char_ngram_range,
                min_df=min_df,
                max_df=self.max_df,
                max_features=max(32, int(vocabulary_cap * 0.4)),
                sublinear_tf=self.sublinear_tf,
                dtype=np.float32,
            )
            try:
                word = self.word_vectorizer_.fit_transform(texts)
                char = self.char_vectorizer_.fit_transform(texts)
                combined = sparse.hstack([word, char], format="csr")
            except ValueError as exc:
                if min_df != 1:
                    self.word_vectorizer_.set_params(min_df=1)
                    self.char_vectorizer_.set_params(min_df=1)
                    try:
                        word = self.word_vectorizer_.fit_transform(texts)
                        char = self.char_vectorizer_.fit_transform(texts)
                        combined = sparse.hstack([word, char], format="csr")
                        self.notes_.append(
                            "text min_df automatically relaxed to 1 after sparse-vocabulary fallback"
                        )
                    except ValueError:
                        self._record_error(f"text vectorization failed: {exc}")
                        self.word_vectorizer_ = None
                        self.char_vectorizer_ = None
                        return None
                else:
                    self._record_error(f"text vectorization failed: {exc}")
                    self.word_vectorizer_ = None
                    self.char_vectorizer_ = None
                    return None

        fit_matrix = combined
        if combined.shape[0] > self.max_svd_fit_rows:
            rng = np.random.default_rng(self.random_state)
            sample_indices = np.sort(
                rng.choice(combined.shape[0], size=self.max_svd_fit_rows, replace=False)
            )
            fit_matrix = combined[sample_indices]
            self.notes_.append(
                f"LSA fitted on a bounded {len(sample_indices)}-row sample and replayed on all rows"
            )
        components = min(
            budget,
            max(1, fit_matrix.shape[0] - 1),
            max(1, fit_matrix.shape[1] - 1),
        )
        if components < 1:
            return None
        self.svd_ = TruncatedSVD(
            n_components=components,
            n_iter=5 if self.speed_accuracy < 0.7 else 8,
            random_state=self.random_state,
        )
        self.svd_.fit(fit_matrix)
        transformed = self.svd_.transform(combined)
        prefix = "hash_lsa" if mode == "hashing" else "lsa"
        self.lexical_names_ = [f"{prefix}_{i:04d}" for i in range(transformed.shape[1])]
        return transformed.astype(np.float32, copy=False)

    def _should_use_embeddings(self) -> bool:
        if self.backend == "classical":
            return False
        if self.backend in {"pretrained", "hybrid"}:
            return True
        return bool(
            self.speed_accuracy >= 0.75
            and self.allow_model_download
            and importlib.util.find_spec("sentence_transformers") is not None
        )

    def _load_embedding_model(self) -> Any | None:
        if self._embedding_model is not None:
            return self._embedding_model
        if importlib.util.find_spec("sentence_transformers") is None:
            return None
        try:
            from sentence_transformers import SentenceTransformer

            kwargs: dict[str, Any] = {}
            if not self.allow_model_download:
                kwargs["local_files_only"] = True
            try:
                requested_device = None if self.device == "auto" else ("cuda" if self.device in {"gpu", "cuda"} else "cpu")
                self._embedding_model = SentenceTransformer(self.model_name, device=requested_device, **kwargs)
            except TypeError:
                if not self.allow_model_download:
                    return None
                self._embedding_model = SentenceTransformer(self.model_name, device=("cuda" if self.device in {"gpu", "cuda"} else None))
            return self._embedding_model
        except Exception as exc:
            self._record_error(f"could not load text embedding model {self.model_name!r}: {exc}")
            return None

    def _encode_embeddings(self, texts: list[str]) -> np.ndarray | None:
        model = self._load_embedding_model()
        if model is None:
            return None
        try:
            return np.asarray(
                model.encode(
                    texts,
                    batch_size=self.batch_size,
                    show_progress_bar=False,
                    convert_to_numpy=True,
                    normalize_embeddings=True,
                ),
                dtype=np.float32,
            )
        except Exception as exc:
            self._record_error(f"text embedding failed: {exc}")
            return None

    def _fit_embeddings(self, texts: list[str], budget: int) -> np.ndarray | None:
        embeddings = self._encode_embeddings(texts)
        if embeddings is None or budget < 1:
            return None
        components = min(budget, embeddings.shape[1], max(1, embeddings.shape[0] - 1))
        if components < embeddings.shape[1] and components >= 2:
            self.embedding_reducer_ = PCA(
                n_components=components,
                svd_solver="randomized",
                random_state=self.random_state,
            )
            embeddings = self.embedding_reducer_.fit_transform(embeddings)
        else:
            embeddings = embeddings[:, :components]
        self.embedding_names_ = [f"embedding_{i:04d}" for i in range(embeddings.shape[1])]
        return embeddings.astype(np.float32, copy=False)

    def _normalize(self, values: pd.Series) -> list[str]:
        texts: list[str] = []
        for value in values.tolist():
            if value is None or value is pd.NA or (isinstance(value, float) and np.isnan(value)):
                text = ""
            elif isinstance(value, (bytes, bytearray, memoryview)):
                text = bytes(value).decode(self.encoding, errors="replace")
            elif isinstance(value, Path):
                text = self._read_text_path(value)
            elif isinstance(value, str) and self.input_mode != "text":
                path = Path(value).expanduser()
                try:
                    is_file = path.is_file()
                except OSError:
                    is_file = False
                if self.input_mode == "path" or (self.input_mode == "auto" and is_file):
                    text = self._read_text_path(path)
                else:
                    text = value
            else:
                text = str(value)
            texts.append(self._bounded_text(text))
        return texts

    def _read_text_path(self, value: str | Path) -> str:
        path = Path(value).expanduser()
        if not path.exists() or not path.is_file():
            raise FileNotFoundError(path)
        # Read a bounded prefix/middle/suffix in bytes. This prevents a single
        # huge document from dominating RAM while retaining document structure.
        byte_budget = max(1024, self.max_chars * 4)
        size = path.stat().st_size
        with path.open("rb") as handle:
            if size <= byte_budget:
                payload = handle.read(byte_budget)
            else:
                third = max(1, byte_budget // 3)
                head = handle.read(third)
                handle.seek(max(0, size // 2 - third // 2))
                middle = handle.read(third)
                handle.seek(max(0, size - (byte_budget - 2 * third)))
                tail = handle.read(byte_budget - 2 * third)
                payload = head + b"\n...\n" + middle + b"\n...\n" + tail
        return payload.decode(self.encoding, errors="replace")

    def _bounded_text(self, text: str) -> str:
        if len(text) <= self.max_chars:
            return text
        separator = "\n…\n"
        content_budget = max(3, self.max_chars - 2 * len(separator))
        third = max(1, content_budget // 3)
        middle_start = max(0, len(text) // 2 - third // 2)
        tail_size = max(1, content_budget - 2 * third)
        return separator.join(
            [
                text[:third],
                text[middle_start : middle_start + third],
                text[-tail_size:],
            ]
        )

    @staticmethod
    def _handcrafted_matrix(texts: list[str]) -> tuple[np.ndarray, list[str]]:
        rows: list[list[float]] = []
        names = [
            "char_count", "word_count", "sentence_count", "line_count", "unique_word_ratio",
            "avg_word_length", "std_word_length", "max_word_length", "char_entropy", "token_entropy",
            "digit_ratio", "alpha_ratio", "uppercase_ratio", "lowercase_ratio", "whitespace_ratio",
            "punctuation_ratio", "symbol_ratio", "non_ascii_ratio", "url_count", "email_count",
            "repeated_run_count", "emoji_symbol_count", "latin_ratio", "arabic_ratio", "cyrillic_ratio",
            "cjk_ratio", "newline_ratio", "question_count", "exclamation_count", "numeric_token_ratio",
            "hapax_ratio", "type_token_ratio_log", "compression_proxy", "empty_flag",
        ]
        for text in texts:
            chars = list(text)
            length = len(chars)
            words = _WORD_RE.findall(text)
            word_lengths = np.asarray([len(word) for word in words], dtype=float)
            word_counts = Counter(word.casefold() for word in words)
            unique_words = len(word_counts)
            char_counts = Counter(chars)
            char_entropy = _counter_entropy(char_counts)
            token_entropy = _counter_entropy(word_counts)
            alpha = sum(char.isalpha() for char in chars)
            upper = sum(char.isupper() for char in chars)
            lower = sum(char.islower() for char in chars)
            digits = sum(char.isdigit() for char in chars)
            spaces = sum(char.isspace() for char in chars)
            punctuation = sum(unicodedata.category(char).startswith("P") for char in chars)
            symbols = sum(unicodedata.category(char).startswith("S") for char in chars)
            non_ascii = sum(ord(char) > 127 for char in chars)
            latin = sum("LATIN" in unicodedata.name(char, "") for char in chars if char.isalpha())
            arabic = sum("ARABIC" in unicodedata.name(char, "") for char in chars if char.isalpha())
            cyrillic = sum("CYRILLIC" in unicodedata.name(char, "") for char in chars if char.isalpha())
            cjk = sum(
                any(marker in unicodedata.name(char, "") for marker in ("CJK", "HIRAGANA", "KATAKANA", "HANGUL"))
                for char in chars
            )
            numeric_tokens = sum(token.isnumeric() for token in words)
            hapax = sum(count == 1 for count in word_counts.values())
            denominator = max(1, length)
            word_denominator = max(1, len(words))
            type_token = unique_words / word_denominator
            rows.append([
                float(length), float(len(words)), float(len(_SENTENCE_RE.findall(text))), float(text.count("\n") + 1),
                float(type_token), float(word_lengths.mean()) if len(word_lengths) else 0.0,
                float(word_lengths.std()) if len(word_lengths) else 0.0,
                float(word_lengths.max()) if len(word_lengths) else 0.0,
                char_entropy, token_entropy, digits / denominator, alpha / denominator,
                upper / max(1, alpha), lower / max(1, alpha), spaces / denominator,
                punctuation / denominator, symbols / denominator, non_ascii / denominator,
                float(len(_URL_RE.findall(text))), float(len(_EMAIL_RE.findall(text))),
                float(len(_REPEATED_RE.findall(text))), float(symbols), latin / max(1, alpha),
                arabic / max(1, alpha), cyrillic / max(1, alpha), cjk / max(1, alpha),
                text.count("\n") / denominator, float(text.count("?") + text.count("؟")),
                float(text.count("!")), numeric_tokens / word_denominator, hapax / max(1, unique_words),
                math.log1p(unique_words) / math.log1p(max(1, len(words))),
                unique_words / max(1.0, math.sqrt(max(1, length))), float(length == 0),
            ])
        return np.asarray(rows, dtype=np.float32), names

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        state["_embedding_model"] = None
        return state


def _counter_entropy(counter: Counter) -> float:
    total = sum(counter.values())
    if total <= 0:
        return 0.0
    probabilities = np.asarray(list(counter.values()), dtype=float) / total
    return float(-(probabilities * np.log2(probabilities)).sum())
