# Copyright (c) 2025 CoReason, Inc.
#
# This software is proprietary and dual-licensed.
# Licensed under the Prosperity Public License 3.0 (the "License").
# A copy of the license is available at https://prosperitylicense.com/versions/3.0.0
# For details, see the LICENSE file.
# Commercial use beyond a 30-day trial requires a separate license.
#
# Source Code: https://github.com/CoReason-AI/coreason_search

from typing import Any, Dict, Optional

import pytest

from coreason_search.schemas import Hit
from coreason_search.scout import MockScout, get_scout, reset_scout


class TestZeroCopyEdgeCases:
    """Tests for edge cases in Zero-Copy / JIT Fetching logic."""

    def setup_method(self) -> None:
        reset_scout()

    def test_fetcher_exception_propagation(self) -> None:
        """Test that exceptions from the fetcher are propagated."""

        def exploding_fetcher(ptr: Dict[str, str], ctx: Optional[Dict[str, Any]]) -> str:
            raise ValueError("Fetcher exploded")

        scout = MockScout(content_fetcher=exploding_fetcher)
        hit = Hit(
            doc_id="1",
            content=None,
            original_text=None,
            distilled_text="",
            score=1.0,
            source_strategy="s",
            metadata={},
            source_pointer={"id": "1"}
        )

        with pytest.raises(ValueError, match="Fetcher exploded"):
            scout.distill(query="test", hits=[hit])

    def test_fetcher_returns_none(self) -> None:
        """Test behavior when fetcher returns None (simulation of not found)."""

        def none_fetcher(ptr: Dict[str, str], ctx: Optional[Dict[str, Any]]) -> Any:
            return None  # Type hint says str, but runtime might return None

        scout = MockScout(content_fetcher=none_fetcher) # type: ignore
        hit = Hit(
            doc_id="1",
            content=None,
            original_text=None,
            distilled_text="",
            score=1.0,
            source_strategy="s",
            metadata={},
            source_pointer={"id": "1"}
        )

        results = scout.distill(query="test", hits=[hit])
        assert len(results) == 1
        assert results[0].distilled_text == ""
        assert results[0].original_text is None

    def test_fetcher_returns_empty_string(self) -> None:
        """Test behavior when fetcher returns empty string."""

        def empty_fetcher(ptr: Dict[str, str], ctx: Optional[Dict[str, Any]]) -> str:
            return ""

        scout = MockScout(content_fetcher=empty_fetcher)
        hit = Hit(
            doc_id="1",
            content=None,
            original_text=None,
            distilled_text="",
            score=1.0,
            source_strategy="s",
            metadata={},
            source_pointer={"id": "1"}
        )

        results = scout.distill(query="test", hits=[hit])
        assert len(results) == 1
        assert results[0].distilled_text == ""

    def test_missing_pointer_and_text(self) -> None:
        """Test hit with no text AND no pointer."""
        scout = MockScout(content_fetcher=lambda p, c: "content")
        hit = Hit(
            doc_id="1",
            content=None,
            original_text=None,
            distilled_text="",
            score=1.0,
            source_strategy="s",
            metadata={},
            source_pointer=None # No pointer
        )

        results = scout.distill(query="test", hits=[hit])
        assert len(results) == 1
        # Should not have called fetcher (if it did, it would get content)
        assert results[0].distilled_text == ""

    def test_mixed_batch_processing(self) -> None:
        """Test a batch of mixed Hit types (Pointer, Full Text, Broken Pointer)."""

        def hybrid_fetcher(ptr: Dict[str, str], ctx: Optional[Dict[str, Any]]) -> str:
            if ptr.get("id") == "pointer":
                return "This is fetched content."
            return ""

        scout = MockScout(content_fetcher=hybrid_fetcher)

        hit_text = Hit(
            doc_id="text",
            content="Existing content",
            original_text="Existing content",
            distilled_text="",
            score=1.0,
            source_strategy="s",
            metadata={}
        )

        hit_pointer = Hit(
            doc_id="pointer",
            content=None,
            original_text=None,
            distilled_text="",
            score=1.0,
            source_strategy="s",
            metadata={},
            source_pointer={"id": "pointer"}
        )

        hit_broken = Hit(
            doc_id="broken",
            content=None,
            original_text=None,
            distilled_text="",
            score=1.0,
            source_strategy="s",
            metadata={},
            source_pointer={"id": "broken"} # returns empty
        )

        # Query matches "content"
        results = scout.distill(query="content", hits=[hit_text, hit_pointer, hit_broken])

        assert len(results) == 3

        # 1. Text Hit: processed normally
        res_text = next(h for h in results if h.doc_id == "text")
        assert "Existing content" in res_text.distilled_text

        # 2. Pointer Hit: fetched and processed
        res_pointer = next(h for h in results if h.doc_id == "pointer")
        assert "fetched content" in res_pointer.distilled_text
        assert res_pointer.original_text is None

        # 3. Broken Hit: fetched empty, result empty
        res_broken = next(h for h in results if h.doc_id == "broken")
        assert res_broken.distilled_text == ""
