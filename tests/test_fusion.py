# Copyright (c) 2025 CoReason, Inc.
#
# This software is proprietary and dual-licensed.
# Licensed under the Prosperity Public License 3.0 (the "License").
# A copy of the license is available at https://prosperitylicense.com/versions/3.0.0
# For details, see the LICENSE file.
# Commercial use beyond a 30-day trial requires a separate license.
#
# Source Code: https://github.com/CoReason-AI/coreason_search


from coreason_search.fusion import FusionEngine
from coreason_search.schemas import Hit


def create_hit(doc_id: str, score: float = 1.0) -> Hit:
    return Hit(
        doc_id=doc_id,
        content="content",
        original_text="text",
        distilled_text="",
        score=score,
        source_strategy="test",
        metadata={},
    )


class TestFusionEngine:
    def test_rrf_simple(self) -> None:
        """Test simple RRF fusion of two lists."""
        # List A: [1, 2, 3]
        list_a = [create_hit("1"), create_hit("2"), create_hit("3")]
        # List B: [3, 2, 4]
        list_b = [create_hit("3"), create_hit("2"), create_hit("4")]

        fusion = FusionEngine(k=1)
        # RRF for doc 1: 1/(1+1) = 0.5
        # RRF for doc 2: 1/(1+2) + 1/(1+2) = 0.33 + 0.33 = 0.66
        # RRF for doc 3: 1/(1+3) + 1/(1+1) = 0.25 + 0.5 = 0.75
        # RRF for doc 4: 1/(1+3) = 0.25

        # Expected order: 3, 2, 1, 4

        results = fusion.fuse([list_a, list_b])

        assert len(results) == 4
        assert results[0].doc_id == "3"
        assert results[1].doc_id == "2"
        assert results[2].doc_id == "1"
        assert results[3].doc_id == "4"

    def test_empty_lists(self) -> None:
        """Test fusion with empty lists."""
        fusion = FusionEngine()
        results = fusion.fuse([])
        assert results == []

        results = fusion.fuse([[], []])
        assert results == []

    def test_single_list(self) -> None:
        """Test fusion with a single list (should preserve order but change scores)."""
        list_a = [create_hit("1"), create_hit("2")]
        fusion = FusionEngine(k=60)
        results = fusion.fuse([list_a])

        assert len(results) == 2
        assert results[0].doc_id == "1"
        assert results[1].doc_id == "2"
        # Score check: 1/61 vs 1/62
        assert results[0].score > results[1].score

    def test_deduplication(self) -> None:
        """Test that duplicates are merged."""
        list_a = [create_hit("1")]
        list_b = [create_hit("1")]
        fusion = FusionEngine()
        results = fusion.fuse([list_a, list_b])
        assert len(results) == 1
        assert results[0].doc_id == "1"
        # Score should be sum
        # k=60. 1/61 + 1/61 = 2/61 approx 0.032
        assert results[0].score > 0.03

    def test_stability(self) -> None:
        """Test that objects are copied correctly."""
        h1 = create_hit("1")
        list_a = [h1]
        fusion = FusionEngine()
        results = fusion.fuse([list_a])

        assert results[0] is not h1  # Should be a copy
        assert results[0].doc_id == h1.doc_id
