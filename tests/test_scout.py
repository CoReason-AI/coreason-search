# Copyright (c) 2025 CoReason, Inc.
#
# This software is proprietary and dual-licensed.
# Licensed under the Prosperity Public License 3.0 (the "License").
# A copy of the license is available at https://prosperitylicense.com/versions/3.0.0
# For details, see the LICENSE file.
# Commercial use beyond a 30-day trial requires a separate license.
#
# Source Code: https://github.com/CoReason-AI/coreason_search

from typing import Dict, Generator

import pytest

from coreason_search.schemas import Hit
from coreason_search.scout import MockScout, get_scout, reset_scout


class TestScout:
    @pytest.fixture(autouse=True)  # type: ignore[misc]
    def setup_teardown(self) -> Generator[None, None, None]:
        reset_scout()
        yield
        reset_scout()

    def test_singleton(self) -> None:
        """Test singleton behavior."""
        s1 = get_scout()
        s2 = get_scout()
        assert s1 is s2
        assert isinstance(s1, MockScout)

    def test_mock_scout_distillation(self) -> None:
        """Test that MockScout correctly 'distills' text."""
        scout = get_scout()

        original_text = "This is a very long sentence that needs to be distilled."
        hit = Hit(
            doc_id="1",
            content="preview",
            original_text=original_text,
            distilled_text="",  # Initially empty or same as original
            score=1.0,
            source_strategy="test",
            metadata={},
        )

        hits = [hit]
        result_hits = scout.distill(query="test", hits=hits)

        assert len(result_hits) == 1
        distilled = result_hits[0].distilled_text

        # Verify it changed (mock logic halves the string)
        assert len(distilled) < len(original_text)
        assert distilled.endswith("...")

        # Verify original text is untouched
        assert result_hits[0].original_text == original_text

    def test_mock_scout_empty_hits(self) -> None:
        """Test MockScout with empty list."""
        scout = get_scout()
        result_hits = scout.distill(query="test", hits=[])
        assert result_hits == []

    def test_mock_scout_edge_cases(self) -> None:
        """Test MockScout with edge cases (empty strings, short strings)."""
        scout = get_scout()

        # Case 1: Empty string
        hit_empty = Hit(
            doc_id="empty",
            content="",
            original_text="",
            distilled_text="something",  # Should be overwritten
            score=1.0,
            source_strategy="test",
            metadata={},
        )

        # Case 2: Short string (len 1) -> max(1, 1//2) = 1. So it keeps 1 char + "..."
        hit_short = Hit(
            doc_id="short",
            content="a",
            original_text="a",
            distilled_text="",
            score=1.0,
            source_strategy="test",
            metadata={},
        )

        hits = [hit_empty, hit_short]
        results = scout.distill(query="test", hits=hits)

        assert len(results) == 2

        # Check empty
        # original_len = 0. We added check for 0.
        # "..."
        assert results[0].distilled_text == "..."

        # Check short
        # original_len = 1. keep_len = max(1, 0) = 1.
        # "a"[:1] -> "a".
        # Adds "..." -> "a..."
        assert results[1].distilled_text == "a..."

    def test_mock_scout_boolean_query(self) -> None:
        """Test MockScout with a Dict (boolean) query."""
        scout = get_scout()

        hit = Hit(
            doc_id="1",
            content="text",
            original_text="some text content",
            distilled_text="",
            score=1.0,
            source_strategy="test",
            metadata={},
        )

        # The mock doesn't use the query, but we verify the interface accepts Dict
        bool_query: Dict[str, str] = {"title": "some term"}
        results = scout.distill(query=bool_query, hits=[hit])

        assert len(results) == 1
        assert results[0].doc_id == "1"
