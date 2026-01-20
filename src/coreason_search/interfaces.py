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
from typing import Dict, List, Union

import numpy as np

from coreason_search.schemas import Hit, SearchRequest


class BaseEmbedder(ABC):
    """Abstract base class for all embedders."""

    @abstractmethod
    def embed(self, text: Union[str, List[str]]) -> np.ndarray:
        """Embed a string or list of strings into a numpy array.

        Args:
            text: Single string or list of strings to embed.

        Returns:
            np.ndarray: Array of shape (1, dim) or (n, dim).
        """
        pass  # pragma: no cover


class BaseRetriever(ABC):
    """Abstract base class for all retrievers (strategies)."""

    @abstractmethod
    def retrieve(self, request: SearchRequest) -> List[Hit]:
        """Execute the retrieval strategy.

        Args:
            request: The full search request object.

        Returns:
            List[Hit]: A list of raw hits from the backend.
        """
        pass  # pragma: no cover


class BaseReranker(ABC):
    """Abstract base class for re-rankers."""

    @abstractmethod
    def rerank(self, query: Union[str, Dict[str, str]], hits: List[Hit], top_k: int) -> List[Hit]:
        """Re-rank the hits using a cross-encoder or other logic.

        Args:
            query: The user query.
            hits: The list of hits to re-rank.
            top_k: The number of top results to return.

        Returns:
            List[Hit]: The re-ranked list of hits.
        """
        pass  # pragma: no cover


class BaseScout(ABC):
    """Abstract base class for The Scout (Context Distiller)."""

    @abstractmethod
    def distill(self, query: Union[str, Dict[str, str]], hits: List[Hit]) -> List[Hit]:
        """Distill the content of the hits, removing irrelevant parts.

        Args:
            query: The user query.
            hits: The list of hits to process.

        Returns:
            List[Hit]: The list of hits with 'distilled_text' populated/updated.
        """
        pass  # pragma: no cover
