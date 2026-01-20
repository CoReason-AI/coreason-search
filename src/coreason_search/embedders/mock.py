# Copyright (c) 2025 CoReason, Inc.
#
# This software is proprietary and dual-licensed.
# Licensed under the Prosperity Public License 3.0 (the "License").
# A copy of the license is available at https://prosperitylicense.com/versions/3.0.0
# For details, see the LICENSE file.
# Commercial use beyond a 30-day trial requires a separate license.
#
# Source Code: https://github.com/CoReason-AI/coreason_search

from typing import List, Union

import numpy as np

from coreason_search.config import EmbeddingConfig
from coreason_search.interfaces import BaseEmbedder


class MockEmbedder(BaseEmbedder):
    """Mock embedder that generates random vectors.

    Used for testing and environments where heavy models cannot be loaded.
    """

    def __init__(self, config: EmbeddingConfig):
        """Initialize the Mock Embedder.

        Args:
            config: Configuration for the embedder.
        """
        self.config = config
        self.embedding_dim = 1024

    def embed(self, text: Union[str, List[str]]) -> np.ndarray:
        """Generate random embeddings for the input text.

        Args:
            text: Single string or list of strings.

        Returns:
            np.ndarray: Random embeddings array.
        """
        if isinstance(text, str):
            text = [text]

        if not text:
            return np.empty((0, self.embedding_dim), dtype=np.float32)

        rng = np.random.default_rng(42)
        count = len(text)
        embeddings = rng.random((count, self.embedding_dim), dtype=np.float32)

        return embeddings
