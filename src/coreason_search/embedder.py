# Copyright (c) 2025 CoReason, Inc.
#
# This software is proprietary and dual-licensed.
# Licensed under the Prosperity Public License 3.0 (the "License").
# A copy of the license is available at https://prosperitylicense.com/versions/3.0.0
# For details, see the LICENSE file.
# Commercial use beyond a 30-day trial requires a separate license.
#
# Source Code: https://github.com/CoReason-AI/coreason_search

from functools import lru_cache
from typing import Optional

from coreason_search.config import EmbeddingConfig
from coreason_search.interfaces import BaseEmbedder


@lru_cache(maxsize=32)
def get_embedder(config: Optional[EmbeddingConfig] = None) -> BaseEmbedder:
    """
    Singleton factory for the Embedder.
    Selects implementation based on config.provider.
    """
    if config is None:
        config = EmbeddingConfig()

    from coreason_search.embedders.mock import MockEmbedder
    from coreason_search.utils.logger import logger

    # Explicit Mock
    if config.provider == "mock":
        return MockEmbedder(config)

    # Explicit HF
    if config.provider == "hf":
        from coreason_search.embedders.hf import HuggingFaceEmbedder

        return HuggingFaceEmbedder(config)

    # Auto: Try HF, fall back to Mock
    if config.provider == "auto":
        try:
            from coreason_search.embedders.hf import HuggingFaceEmbedder

            return HuggingFaceEmbedder(config)
        except ImportError:
            logger.warning("Could not load HuggingFaceEmbedder (missing dependencies). Falling back to MockEmbedder.")
            return MockEmbedder(config)
        except Exception as e:  # pragma: no cover
            logger.error(f"Failed to initialize HuggingFaceEmbedder: {e}. Falling back to MockEmbedder.")
            return MockEmbedder(config)

    # Default fallback
    return MockEmbedder(config)  # pragma: no cover


def reset_embedder() -> None:
    """Reset the singleton instance (clear cache)."""
    get_embedder.cache_clear()
