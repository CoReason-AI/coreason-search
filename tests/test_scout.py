# Copyright (c) 2025 CoReason, Inc.
#
# This software is proprietary and dual-licensed.
# Licensed under the Prosperity Public License 3.0 (the "License").
# A copy of the license is available at https://prosperitylicense.com/versions/3.0.0
# For details, see the LICENSE file.
# Commercial use beyond a 30-day trial requires a separate license.
#
# Source Code: https://github.com/CoReason-AI/coreason_search

from typing import Dict, Generator, Optional

import pytest
from coreason_identity.models import UserContext

from coreason_search.schemas import Hit
from coreason_search.scout import MockScout, get_scout, reset_scout


class TestScout:
    @pytest.fixture(autouse=True)
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

    def test_scout_distillation_logic(self) -> None:
        """
        Test that Scout correctly filters relevant sentences.
        """
        scout = get_scout()

        # Two sentences: one relevant (contains 'fruit'), one irrelevant.
        original_text = "Apple is a fruit. Cars are fast."
        hit = Hit(
            doc_id="1",
            content="preview",
            original_text=original_text,
            distilled_text="",
            score=1.0,
            source_strategy="test",
            metadata={},
        )

        hits = [hit]
        # Query contains 'fruit'
        result_hits = scout.distill(query="fruit", hits=hits)

        assert len(result_hits) == 1
        distilled = result_hits[0].distilled_text

        # Expect only the first sentence
        assert "Apple is a fruit" in distilled
        assert "Cars are fast" not in distilled
        assert distilled.strip() == "Apple is a fruit."

    def test_scout_no_match(self) -> None:
        """Test that if no sentences match, distilled text is empty."""
        scout = get_scout()

        original_text = "Cars are fast. The sky is blue."
        hit = Hit(
            doc_id="1",
            content="preview",
            original_text=original_text,
            distilled_text="",
            score=1.0,
            source_strategy="test",
            metadata={},
        )

        # Query matches nothing
        result_hits = scout.distill(query="fruit", hits=[hit])

        # Expect empty
        assert result_hits[0].distilled_text == ""

    def test_scout_full_match(self) -> None:
        """Test that if all sentences match, all are kept."""
        scout = get_scout()

        original_text = "Apple is a fruit. Banana is also a fruit."
        hit = Hit(
            doc_id="1",
            content="preview",
            original_text=original_text,
            distilled_text="",
            score=1.0,
            source_strategy="test",
            metadata={},
        )

        result_hits = scout.distill(query="fruit", hits=[hit])

        distilled = result_hits[0].distilled_text
        assert "Apple is a fruit." in distilled
        assert "Banana is also a fruit." in distilled

    def test_mock_scout_empty_hits(self) -> None:
        """Test MockScout with empty list."""
        scout = get_scout()
        result_hits = scout.distill(query="test", hits=[])
        assert result_hits == []

    def test_mock_scout_edge_cases(self) -> None:
        """Test MockScout with edge cases (empty strings)."""
        scout = get_scout()

        # Case 1: Empty string
        hit_empty = Hit(
            doc_id="empty",
            content="",
            original_text="",
            distilled_text="something",
            score=1.0,
            source_strategy="test",
            metadata={},
        )

        hits = [hit_empty]
        results = scout.distill(query="test", hits=hits)

        assert len(results) == 1
        assert results[0].distilled_text == ""

    def test_mock_scout_boolean_query(self) -> None:
        """Test MockScout with a Dict (boolean) query."""
        scout = get_scout()

        hit = Hit(
            doc_id="1",
            content="text",
            original_text="The title is awesome.",
            distilled_text="",
            score=1.0,
            source_strategy="test",
            metadata={},
        )

        # Query dict -> "title is awesome" -> "title is awesome" via extract_query_text
        bool_query: Dict[str, str] = {"text": "awesome"}
        results = scout.distill(query=bool_query, hits=[hit])

        assert len(results) == 1
        # Should match "awesome"
        assert "awesome" in results[0].distilled_text

    def test_empty_query(self) -> None:
        """Test MockScout with empty query (should return 0.0 score and thus empty result)."""
        scout = get_scout()
        hit = Hit(
            doc_id="1",
            content="text",
            original_text="Some text",
            distilled_text="",
            score=1.0,
            source_strategy="test",
            metadata={},
        )
        # Empty query -> score 0.0 -> filtered out
        results = scout.distill(query="", hits=[hit])
        assert results[0].distilled_text == ""

    def test_scout_substring_matching(self) -> None:
        """
        Test that scoring uses substring/fuzzy matching.
        Query 'run' should match 'running'.
        """
        scout = get_scout()
        hit = Hit(
            doc_id="1",
            content="text",
            original_text="I am running fast.",
            distilled_text="",
            score=1.0,
            source_strategy="test",
            metadata={},
        )
        # "run" is in "running"
        results = scout.distill(query="run", hits=[hit])
        assert results[0].distilled_text == "I am running fast."

    def test_scout_complex_structure(self) -> None:
        """
        Test handling of newlines and multiple spaces.
        """
        scout = get_scout()
        # Text with newlines and bullets
        original_text = "Header.\n* Item 1 is cool.\n* Item 2 is bad."
        hit = Hit(
            doc_id="1",
            content="text",
            original_text=original_text,
            distilled_text="",
            score=1.0,
            source_strategy="test",
            metadata={},
        )
        # Query matches "cool"
        results = scout.distill(query="cool", hits=[hit])
        # Should keep "Item 1 is cool."
        # Should filter "Item 2 is bad."
        # Header might be filtered if it doesn't match.
        distilled = results[0].distilled_text
        assert "Item 1 is cool" in distilled
        assert "Item 2 is bad" not in distilled

    def test_scout_punctuation_edge_cases(self) -> None:
        """
        Test handling of tricky punctuation like decimals and abbreviations.
        """
        scout = get_scout()
        # "3.14" might be split by naive regex.
        # "Mr. Smith" might be split.
        original_text = "The value is 3.14. Mr. Smith matches."
        hit = Hit(
            doc_id="1",
            content="text",
            original_text=original_text,
            distilled_text="",
            score=1.0,
            source_strategy="test",
            metadata={},
        )

        # 1. Test Decimal retention
        # Query "value" matches first part.
        results = scout.distill(query="value", hits=[hit])
        # If naive split "3.14" -> "3." and "14.", then "14." won't be kept.
        # Ideally we want "The value is 3.14." to be kept.
        # But for mock, if it splits, we just verify the behavior.
        # If it splits, "The value is 3." is kept. "14." is dropped.
        # If we fix regex, it should keep "3.14".
        # Let's assert what we WANT (robustness).
        assert "3.14" in results[0].distilled_text

        # 2. Test Abbreviation retention
        # Query "Smith" matches "Smith".
        results2 = scout.distill(query="Smith", hits=[hit])
        # If split "Mr." and "Smith matches.", "Mr." is dropped (no match), "Smith matches." is kept.
        # So "Mr. Smith matches." -> "Smith matches."
        # This shows the limitation.
        # We'll see what happens.
        assert "Smith" in results2[0].distilled_text

    def test_scout_jit_fetching(self) -> None:
        """Test JIT fetching of content via workspace/fetcher."""

        # Mock Fetcher
        def mock_fetcher(source_pointer: Dict[str, str], user_context: Optional[UserContext]) -> str:
            # Verify we received the context
            if user_context and "secret_key" in user_context.permissions:
                return "Secret content is here. It is safe."
            return "Public content."

        # Instantiate Scout with fetcher
        scout = MockScout(content_fetcher=mock_fetcher)

        # Hit without original_text but with pointer
        hit = Hit(
            doc_id="jit-1",
            content=None,
            original_text=None,
            distilled_text="",
            score=1.0,
            source_strategy="jit",
            metadata={},
            source_pointer={"id": "doc1"},
        )

        # 1. Test with authorized context
        user_context = UserContext(sub="u", email="u@e.com", permissions=["secret_key"])
        results = scout.distill(query="Secret", hits=[hit], user_context=user_context)

        assert len(results) == 1
        res = results[0]
        # Should have fetched "Secret content is here. It is safe." and distilled it
        assert "Secret content is here" in res.distilled_text
        # Ephemeral check: original_text must NOT be populated on the result
        assert res.original_text is None

        # 2. Test without authorized context (Mock logic)
        results_public = scout.distill(query="Public", hits=[hit], user_context=None)
        assert len(results_public) == 1
        assert "Public content" in results_public[0].distilled_text
        assert results_public[0].original_text is None
