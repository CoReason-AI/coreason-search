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
from typing import Dict, List, Optional, Union

from coreason_search.config import ScoutConfig
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

    def __init__(self, config: Optional[ScoutConfig] = None):
        self.config = config or ScoutConfig()

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
                # Use threshold from config
                if score > self.config.threshold:
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
def get_scout(config: Optional[ScoutConfig] = None) -> BaseScout:
    """
    Singleton factory for Scout.
    """
    if config is None:
        config = ScoutConfig()
    return MockScout(config)


def reset_scout() -> None:
    """Reset singleton."""
    get_scout.cache_clear()
