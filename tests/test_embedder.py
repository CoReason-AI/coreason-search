# Copyright (c) 2025 CoReason, Inc.
#
# This software is proprietary and dual-licensed.
# Licensed under the Prosperity Public License 3.0 (the "License").
# A copy of the license is available at https://prosperitylicense.com/versions/3.0.0
# For details, see the LICENSE file.
# Commercial use beyond a 30-day trial requires a separate license.
#
# Source Code: https://github.com/CoReason-AI/coreason_search

from typing import Generator

import numpy as np
import pytest

from coreason_search.config import EmbeddingConfig
from coreason_search.embedder import MockEmbedder, get_embedder, reset_embedder


class TestEmbedder:
    @pytest.fixture(autouse=True)  # type: ignore[misc]
    def setup_teardown(self) -> Generator[None, None, None]:
        reset_embedder()
        yield
        reset_embedder()

    def test_singleton_behavior(self) -> None:
        """Test that get_embedder returns the same instance for same config."""
        embedder1 = get_embedder()
        embedder2 = get_embedder()
        assert embedder1 is embedder2
        assert isinstance(embedder1, MockEmbedder)

    def test_mock_embed_string(self) -> None:
        """Test embedding a single string."""
        embedder = get_embedder()
        text = "Hello world"
        vector = embedder.embed(text)
        assert isinstance(vector, np.ndarray)
        assert vector.shape == (1, 1024)
        assert vector.dtype == np.float32

    def test_mock_embed_list(self) -> None:
        """Test embedding a list of strings."""
        embedder = get_embedder()
        texts = ["Hello", "World", "Test"]
        vectors = embedder.embed(texts)
        assert isinstance(vectors, np.ndarray)
        assert vectors.shape == (3, 1024)

    def test_mock_embed_empty_list(self) -> None:
        """Test embedding an empty list."""
        embedder = get_embedder()
        vectors = embedder.embed([])
        assert isinstance(vectors, np.ndarray)
        assert vectors.shape == (0, 1024)

    def test_mock_embed_empty_string(self) -> None:
        """Test embedding an empty string."""
        embedder = get_embedder()
        vector = embedder.embed("")
        assert vector.shape == (1, 1024)

    def test_custom_config(self) -> None:
        """Test initializing with custom config."""
        config = EmbeddingConfig(model_name="test-model")
        embedder = get_embedder(config)
        assert isinstance(embedder, MockEmbedder)
        assert embedder.config.model_name == "test-model"

    def test_lru_cache_distinct_configs(self) -> None:
        """Test that different configs yield different instances (lru_cache behavior)."""
        config1 = EmbeddingConfig(model_name="model1")
        embedder1 = get_embedder(config1)
        assert isinstance(embedder1, MockEmbedder)  # mypy check
        assert embedder1.config.model_name == "model1"

        config2 = EmbeddingConfig(model_name="model2")
        embedder2 = get_embedder(config2)

        # With lru_cache, different arguments -> new result
        assert embedder2 is not embedder1
        assert isinstance(embedder2, MockEmbedder)
        assert embedder2.config.model_name == "model2"

        # Verify idempotency for same config
        embedder1_again = get_embedder(config1)
        assert embedder1_again is embedder1

    def test_embed_large_batch(self) -> None:
        """Test embedding a large batch of strings."""
        embedder = get_embedder()
        batch_size = 1000
        texts = [f"Item {i}" for i in range(batch_size)]
        vectors = embedder.embed(texts)
        assert vectors.shape == (batch_size, 1024)

    def test_embed_special_characters_unicode(self) -> None:
        """Test embedding strings with special characters and unicode."""
        embedder = get_embedder()
        texts = ["Hello \n\t World", "🎉 Emoji Test 🚀", "Müller"]
        vectors = embedder.embed(texts)
        assert vectors.shape == (3, 1024)

    def test_embed_long_string(self) -> None:
        """Test embedding a very long string."""
        embedder = get_embedder()
        text = "A" * 100000
        vector = embedder.embed(text)
        assert vector.shape == (1, 1024)

    def test_embed_whitespace_string(self) -> None:
        """Test embedding a string with only whitespace."""
        embedder = get_embedder()
        text = "   \n   "
        vector = embedder.embed(text)
        assert vector.shape == (1, 1024)
