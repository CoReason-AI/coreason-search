# Copyright (c) 2025 CoReason, Inc.
#
# This software is proprietary and dual-licensed.
# Licensed under the Prosperity Public License 3.0 (the "License").
# A copy of the license is available at https://prosperitylicense.com/versions/3.0.0
# For details, see the LICENSE file.
# Commercial use beyond a 30-day trial requires a separate license.
#
# Source Code: https://github.com/CoReason-AI/coreason_search

from typing import Dict, List

from coreason_search.schemas import Hit


class FusionEngine:
    """Fusion Engine using Reciprocal Rank Fusion (RRF).

    Combines results from multiple retrieval strategies into a single ranked list.
    """

    def __init__(self, k: int = 60):
        """Initialize Fusion Engine.

        Args:
            k: The constant k for RRF. Defaults to 60.
        """
        self.k = k

    def fuse(self, results: List[List[Hit]]) -> List[Hit]:
        """Fuse multiple lists of Hits into a single ranked list.

        Args:
            results: A list of lists of Hit objects from different strategies.

        Returns:
            List[Hit]: A single list of fused, unique Hits sorted by RRF score.
        """
        if not results:
            return []

        # Map doc_id -> RRF score
        doc_scores: Dict[str, float] = {}
        # Map doc_id -> Hit object (keep the first occurrence or merge?)
        # PRD says "deduped". We can keep the first one we encounter or try to merge metadata?
        # Let's keep the one from the highest priority strategy?
        # Actually, RRF doesn't care about "priority", it treats ranks.
        # We need a reference to the Hit object.
        doc_map: Dict[str, Hit] = {}

        for hit_list in results:
            for rank, hit in enumerate(hit_list):
                # RRF Formula: score += 1 / (k + rank)
                # rank is 0-indexed here? RRF usually uses 1-based rank?
                # "1 / (k + rank_vector)" where rank_vector starts at 1 usually.
                # If rank is 0-indexed, use rank + 1.
                score = 1.0 / (self.k + rank + 1)

                if hit.doc_id in doc_scores:
                    doc_scores[hit.doc_id] += score
                    # Optional: Merge source strategies? e.g. "dense,sparse"
                    # hit.source_strategy is str.
                    # We can append if different?
                    # Let's simple keep the first one but update score?
                    # Or keep the one with better original score? No, RRF replaces score.
                else:
                    doc_scores[hit.doc_id] = score
                    doc_map[hit.doc_id] = hit

        # Create new hits with updated scores
        fused_hits = []
        for doc_id, rrf_score in doc_scores.items():
            original_hit = doc_map[doc_id]
            # Create a copy or modify? Pydantic models -> copy.
            new_hit = original_hit.model_copy()
            new_hit.score = rrf_score
            # We might want to note it was fused?
            # new_hit.source_strategy = "fused"?
            # Or keep original for provenance.
            # PRD doesn't specify changing source_strategy.
            fused_hits.append(new_hit)

        # Sort by score descending
        fused_hits.sort(key=lambda x: x.score, reverse=True)

        return fused_hits
