# Copyright (c) 2025 CoReason, Inc.
#
# This software is proprietary and dual-licensed.
# Licensed under the Prosperity Public License 3.0 (the "License").
# A copy of the license is available at https://prosperitylicense.com/versions/3.0.0
# For details, see the LICENSE file.
# Commercial use beyond a 30-day trial requires a separate license.
#
# Source Code: https://github.com/CoReason-AI/coreason_search

from typing import TYPE_CHECKING, List, Optional, Union

import numpy as np

from coreason_search.config import EmbeddingConfig
from coreason_search.embedder import BaseEmbedder
from coreason_search.utils.logger import logger

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer  # pragma: no cover


class HuggingFaceEmbedder(BaseEmbedder):
    """
    Embedder implementation using Sentence Transformers or HuggingFace models.
    Supports local sovereign execution.
    """

    def __init__(self, config: EmbeddingConfig):
        """
        Initialize the HuggingFace Embedder.
        Attempts to import sentence_transformers.
        """
        self.config = config
        self._model: Optional["SentenceTransformer"] = None

        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as e:
            raise ImportError(
                "sentence-transformers is not installed. "
                "Install it with `pip install sentence-transformers` "
                "to use HuggingFaceEmbedder."
            ) from e

        logger.info(f"Loading embedding model: {config.model_name}")
        # Initialize model (this might download weights)
        # We assume standard SentenceTransformer compatible models
        # For Qwen, strict trust_remote_code might be needed if it's custom.
        # But `gte-Qwen2` typically works with sentence-transformers >= 3.0
        # We'll use default device selection (cuda if available)
        self._model = SentenceTransformer(
            config.model_name,
            trust_remote_code=True,
        )

        # Set max sequence length if specified
        if config.context_length:
            self._model.max_seq_length = config.context_length

    def embed(self, text: Union[str, List[str]]) -> np.ndarray:
        """
        Embed text using the loaded model.
        """
        if self._model is None:
            raise RuntimeError("Model not initialized")  # pragma: no cover

        if isinstance(text, str):
            text = [text]

        if not text:
            # Handle empty input consistent with MockEmbedder
            # Get dimension from model
            dim = self._model.get_sentence_embedding_dimension()
            return np.empty((0, dim), dtype=np.float32)

        # Encode
        # normalize_embeddings=True is usually desired for cosine similarity
        embeddings = self._model.encode(
            text,
            batch_size=self.config.batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )

        return embeddings
