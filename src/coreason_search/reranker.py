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
from typing import Dict, List, Union

from coreason_search.schemas import Hit


class BaseReranker(ABC):
    """Abstract base class for re-rankers."""

    @abstractmethod
    def rerank(self, query: Union[str, Dict[str, str]], hits: List[Hit], top_k: int) -> List[Hit]:
        """
        Re-rank the hits using a cross-encoder or other logic.

        Args:
            query: The user query.
            hits: The list of hits to re-rank.
            top_k: The number of top results to return.

        Returns:
            List[Hit]: The re-ranked list of hits.
        """
        pass  # pragma: no cover


class MockReranker(BaseReranker):
    """
    Mock Re-Ranker that simulates Cross-Encoder behavior.
    """

    def rerank(self, query: Union[str, Dict[str, str]], hits: List[Hit], top_k: int) -> List[Hit]:
        """
        Mock re-ranking.

        Args:
            query: The user query.
            hits: The list of hits to re-rank.
            top_k: The number of top results to return.

        Returns:
            List[Hit]: The re-ranked list of hits.
        """
        if not hits:
            return []

        # Simulate scoring based on some logic, or just reverse order to prove it does something?
        # Or random?
        # Let's use string length of content + query length as a deterministic "score" for testing stability.
        # Or just assign a random score.
        # Ideally, we want to see the order CHANGE from the input.
        # If input is already sorted by something, let's sort by content length descending.

        scored_hits = []
        for hit in hits:
            # Mock score: length of content. longer content = more relevant (just for mock)
            # In real life, cross encoder gives a float.
            new_score = len(hit.content) * 0.01

            new_hit = hit.model_copy()
            new_hit.score = new_score
            scored_hits.append(new_hit)

        # Sort by new score
        scored_hits.sort(key=lambda x: x.score, reverse=True)

        return scored_hits[:top_k]


@lru_cache(maxsize=32)
def get_reranker() -> BaseReranker:
    """Singleton factory for Reranker."""
    return MockReranker()


def reset_reranker() -> None:
    get_reranker.cache_clear()
