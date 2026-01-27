# Copyright (c) 2025 CoReason, Inc.
#
# This software is proprietary and dual-licensed.
# Licensed under the Prosperity Public License 3.0 (the "License").
# A copy of the license is available at https://prosperitylicense.com/versions/3.0.0
# For details, see the LICENSE file.
# Commercial use beyond a 30-day trial requires a separate license.
#
# Source Code: https://github.com/CoReason-AI/coreason_search

from typing import Generator

import pytest

from coreason_search.reranker import MockReranker, get_reranker, reset_reranker
from coreason_search.schemas import Hit


class TestReranker:
    @pytest.fixture(autouse=True)
    def setup_teardown(self) -> Generator[None, None, None]:
        reset_reranker()
        yield
        reset_reranker()

    def test_singleton(self) -> None:
        r1 = get_reranker()
        r2 = get_reranker()
        assert r1 is r2
        assert isinstance(r1, MockReranker)

    def test_rerank_logic(self) -> None:
        """Test that reranking changes order based on our mock logic (content length)."""
        reranker = get_reranker()

        # Hit 1: Short content
        h1 = Hit(
            doc_id="1",
            content="short",
            original_text="short",
            distilled_text="",
            score=0.9,
            source_strategy="test",
            metadata={},
        )
        # Hit 2: Long content
        h2 = Hit(
            doc_id="2",
            content="very long content indeed",
            original_text="very long content indeed",
            distilled_text="",
            score=0.8,
            source_strategy="test",
            metadata={},
        )

        # Input: [h1, h2] (h1 higher score initially)
        hits = [h1, h2]

        # Rerank with top_k=2
        results = reranker.rerank("query", hits, top_k=2)

        # Expect h2 first because it's longer (mock logic)
        assert results[0].doc_id == "2"
        assert results[1].doc_id == "1"
        assert results[0].score > results[1].score

    def test_top_k_truncation(self) -> None:
        """Test that reranker respects top_k."""
        reranker = get_reranker()
        hits = [
            Hit(
                doc_id=str(i),
                content="a" * i,
                original_text="",
                distilled_text="",
                score=0,
                source_strategy="",
                metadata={},
            )
            for i in range(1, 6)
        ]
        # Lengths: 1, 2, 3, 4, 5. Order after rerank should be 5, 4, 3, 2, 1

        results = reranker.rerank("q", hits, top_k=3)
        assert len(results) == 3
        assert results[0].doc_id == "5"
        assert results[1].doc_id == "4"
        assert results[2].doc_id == "3"

    def test_empty_hits(self) -> None:
        """Test with empty hits."""
        reranker = get_reranker()
        results = reranker.rerank("q", [], top_k=5)
        assert results == []

    def test_query_dict_handling(self) -> None:
        """Test passing a dict as query."""
        reranker = get_reranker()
        h1 = Hit(doc_id="1", content="a", original_text="", distilled_text="", score=0, source_strategy="", metadata={})
        results = reranker.rerank({"q": "v"}, [h1], top_k=1)
        assert len(results) == 1
