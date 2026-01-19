# Copyright (c) 2025 CoReason, Inc.
#
# This software is proprietary and dual-licensed.
# Licensed under the Prosperity Public License 3.0 (the "License").
# A copy of the license is available at https://prosperitylicense.com/versions/3.0.0
# For details, see the LICENSE file.
# Commercial use beyond a 30-day trial requires a separate license.
#
# Source Code: https://github.com/CoReason-AI/coreason_search

from abc import ABC, abstractmethod
from functools import lru_cache
from typing import List, Optional, Union

import numpy as np

from coreason_search.schemas import EmbeddingConfig


class BaseEmbedder(ABC):
    """Abstract base class for all embedders."""

    @abstractmethod
    def embed(self, text: Union[str, List[str]]) -> np.ndarray:
        """
        Embed a string or list of strings into a numpy array.

        Args:
            text: Single string or list of strings to embed.

        Returns:
            np.ndarray: Array of shape (1, dim) or (n, dim).
        """
        pass  # pragma: no cover


class MockEmbedder(BaseEmbedder):
    """
    Mock embedder that generates random vectors.
    Used for testing and environments where heavy models cannot be loaded.
    """

    def __init__(self, config: EmbeddingConfig):
        self.config = config
        # Qwen2-7B-instruct is 3584 dims, but we can default to 1024 or whatever
        # Since this is a mock, we'll use a fixed dimension for consistency unless specified.
        # Let's say 768 as a common default, or maybe we should add `embedding_dim` to config?
        # The config schema doesn't have it. I'll hardcode a "mock" dimension or infer.
        self.embedding_dim = 1024

    def embed(self, text: Union[str, List[str]]) -> np.ndarray:
        if isinstance(text, str):
            text = [text]

        if not text:
            return np.empty((0, self.embedding_dim), dtype=np.float32)

        # Generate random embeddings
        # Seed based on text length to have deterministic-ish behavior for same input length?
        # Or just random. For a mock, random is fine, but deterministic is better for tests.
        # I'll use a fixed seed per call to ensure reproducibility if needed,
        # but pure random is also acceptable for "Mock".
        # Let's use a simple deterministic generation based on string hash to be fancy?
        # No, "generates random numpy arrays" was the instruction.
        rng = np.random.default_rng(42)
        count = len(text)
        embeddings = rng.random((count, self.embedding_dim), dtype=np.float32)

        if self.config.model_name:
            # Just accessing config to ensure it's used
            pass

        return embeddings


@lru_cache(maxsize=32)
def get_embedder(config: Optional[EmbeddingConfig] = None) -> BaseEmbedder:
    """
    Singleton factory for the Embedder.
    """
    if config is None:
        config = EmbeddingConfig()
    return MockEmbedder(config)


def reset_embedder() -> None:
    """Reset the singleton instance (clear cache)."""
    get_embedder.cache_clear()
