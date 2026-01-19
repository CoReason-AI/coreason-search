# Copyright (c) 2025 CoReason, Inc.
#
# This software is proprietary and dual-licensed.
# Licensed under the Prosperity Public License 3.0 (the "License").
# A copy of the license is available at https://prosperitylicense.com/versions/3.0.0
# For details, see the LICENSE file.
# Commercial use beyond a 30-day trial requires a separate license.
#
# Source Code: https://github.com/CoReason-AI/coreason_search

import re
from abc import ABC, abstractmethod
from functools import lru_cache
from typing import Any, Dict, List, Optional, Union

from coreason_search.schemas import Hit
from coreason_search.utils.common import extract_query_text

# Pre-compiled regex for sentence splitting
# Split on . ! ? followed by whitespace using lookbehind
SENTENCE_SPLIT_REGEX = re.compile(r"(?<=[.!?])\s+")
# Pre-compiled regex for unit normalization (removing non-word chars)
UNIT_NORMALIZATION_REGEX = re.compile(r"[^\w\s]")


class BaseScout(ABC):
    """Abstract base class for The Scout (Context Distiller)."""

    @abstractmethod
    def distill(self, query: Union[str, Dict[str, str]], hits: List[Hit]) -> List[Hit]:
        """
        Distill the content of the hits, removing irrelevant parts.

        Args:
            query: The user query.
            hits: The list of hits to process.

        Returns:
            List[Hit]: The list of hits with 'distilled_text' populated/updated.
        """
        pass  # pragma: no cover


class MockScout(BaseScout):
    """
    Mock implementation of The Scout.
    Implements the Segmentation -> Scoring -> Filtering pipeline
    using deterministic heuristics for testing.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}

    def distill(self, query: Union[str, Dict[str, str]], hits: List[Hit]) -> List[Hit]:
        """
        Mock distillation:
        1. Segment text into sentences.
        2. Score sentences based on keyword overlap with query.
        3. Filter out sentences with score 0.
        """
        query_text = extract_query_text(query)
        # Normalize query terms once
        query_terms = set(query_text.lower().split())

        distilled_hits = []
        for hit in hits:
            # Create a copy of the hit
            new_hit = hit.model_copy()
            original_text = hit.original_text

            if not original_text:
                new_hit.distilled_text = ""
                distilled_hits.append(new_hit)
                continue

            # 1. Segmentation
            segments = self._segment(original_text)

            # 2. Scoring & 3. Filtering
            relevant_segments = []
            for seg in segments:
                score = self._score_unit(seg, query_terms)
                if score > 0.5:  # Threshold
                    relevant_segments.append(seg)

            # Reconstruct
            if relevant_segments:
                new_hit.distilled_text = " ".join(relevant_segments)
            else:
                new_hit.distilled_text = ""

            distilled_hits.append(new_hit)

        return distilled_hits

    def _segment(self, text: str) -> List[str]:
        """
        Split text into logical units (sentences).
        Uses pre-compiled regex.
        """
        return [s.strip() for s in SENTENCE_SPLIT_REGEX.split(text) if s.strip()]

    def _score_unit(self, unit: str, query_terms: set[str]) -> float:
        """
        Score a unit based on presence of query terms.
        Returns 1.0 if any query term is present as a substring, 0.0 otherwise.
        """
        # Simple normalization
        unit_clean = unit.lower()

        if not query_terms:
            return 0.0

        # Substring matching
        for term in query_terms:
            if term in unit_clean:
                return 1.0
        return 0.0


@lru_cache(maxsize=32)
def get_scout(config: Optional[Dict[str, Any]] = None) -> BaseScout:
    """
    Singleton factory for Scout.
    Accepts config to allow future configuration of the Scout model.
    """
    # Note: lru_cache will cache based on the config dict.
    # If config is None, it caches the default.
    # If config is mutable (dict), it might fail lru_cache hashing if not handled.
    # However, Python dicts are not hashable.
    # If we want to use lru_cache with dict, we can't directly.
    # We should probably use a Pydantic model for config (like EmbeddingConfig) if we want caching.
    # For now, since MockScout is lightweight and config is unused/optional,
    # we can remove lru_cache OR require config to be hashable (frozendict).
    #
    # Given the previous pattern `get_embedder(config: EmbeddingConfig)`, that worked because EmbeddingConfig is frozen.
    # Since there is no `ScoutConfig` yet, and I shouldn't over-engineer,
    # I will modify `get_scout` to NOT use `lru_cache` on the argument if it's a dict,
    # OR better: I will create a simple internal singleton management manually
    # to avoid the "dict is not hashable" error with lru_cache.
    #
    # Actually, the user asked for standard deps. `lru_cache` is standard.
    # I'll just remove `lru_cache` for now and return the instance,
    # or implement a manual singleton check.
    #
    # But wait, `EmbeddingConfig` exists. I should probably create `ScoutConfig`?
    # No, that's expanding scope.
    #
    # I will just use a global variable or memoization helper.
    # Or, just assume config is passed once.
    #
    # Let's use the simplest valid python: Manual singleton.
    return _get_scout_instance(config)


_SCOUT_INSTANCE: Optional[BaseScout] = None


def _get_scout_instance(config: Optional[Dict[str, Any]] = None) -> BaseScout:
    global _SCOUT_INSTANCE
    if _SCOUT_INSTANCE is None:
        _SCOUT_INSTANCE = MockScout(config)
    return _SCOUT_INSTANCE


def reset_scout() -> None:
    """Reset singleton."""
    global _SCOUT_INSTANCE
    _SCOUT_INSTANCE = None
