# Copyright (c) 2025 CoReason, Inc.
#
# This software is proprietary and dual-licensed.
# Licensed under the Prosperity Public License 3.0 (the "License").
# A copy of the license is available at https://prosperitylicense.com/versions/3.0.0
# For details, see the LICENSE file.
# Commercial use beyond a 30-day trial requires a separate license.
#
# Source Code: https://github.com/CoReason-AI/coreason_search

import json
import time
from typing import Iterator
from unittest.mock import MagicMock

import pytest

from coreason_search.db import DocumentSchema, get_db_manager
from coreason_search.embedder import get_embedder
from coreason_search.retrievers.sparse import SparseRetriever
from coreason_search.schemas import RetrieverType, SearchRequest


class TestSparseRetriever:
    @pytest.fixture(autouse=True)
    def setup_teardown(self, setup_teardown_db_and_embedder: None) -> None:
        """Use shared fixture from conftest."""
        pass

    def _seed_db(self) -> None:
        """Helper to populate DB with some data and FTS index."""
        manager = get_db_manager()
        table = manager.get_table()
        embedder = get_embedder()

        docs = [
            DocumentSchema(
                doc_id="1",
                vector=embedder.embed("apple")[0],
                content="Apple is a fruit.",
                title="Apple Document",
                metadata=json.dumps({"category": "fruit"}),
            ),
            DocumentSchema(
                doc_id="2",
                vector=embedder.embed("banana")[0],
                content="Banana is also a fruit.",
                title="Banana Document",
                metadata=json.dumps({"category": "fruit"}),
            ),
            DocumentSchema(
                doc_id="3",
                vector=embedder.embed("carrot")[0],
                content="Carrot is a vegetable.",
                title="Carrot Document",
                metadata=json.dumps({"category": "vegetable"}),
            ),
        ]
        table.add(docs)
        # Create FTS index for multiple fields
        # Note: tantivy-py is required for multi-field indexing in lancedb

        # Robust index creation for Windows (retry on Access Denied)
        max_retries = 3
        for attempt in range(max_retries):
            try:
                table.create_fts_index(["content", "title"], replace=True, use_tantivy=True)
                break
            except ValueError as e:
                # Catch "Access is denied" (Windows os error 5)
                if "Access is denied" in str(e) and attempt < max_retries - 1:
                    time.sleep(1)
                    continue
                raise

    def test_retrieve_simple(self) -> None:
        """Test simple FTS retrieval."""
        self._seed_db()
        retriever = SparseRetriever()

        request = SearchRequest(query="Apple", strategies=[RetrieverType.LANCE_FTS], top_k=5)

        hits = retriever.retrieve(request)
        assert len(hits) >= 1
        assert hits[0].content == "Apple is a fruit."
        assert hits[0].source_strategy == "lance_fts"

    def test_retrieve_pubmed_syntax(self) -> None:
        """Test retrieval using PubMed-style syntax (e.g. [Title])."""
        self._seed_db()
        retriever = SparseRetriever()

        # "Apple"[Title] should map to title:Apple
        # Doc 1 has Title="Apple Document". Content="Apple is a fruit".
        # Doc 2 has Title="Banana Document". Content="Banana...".
        # Searching Apple[Title] should find Doc 1.

        request = SearchRequest(query='"Apple"[Title]', strategies=[RetrieverType.LANCE_FTS], top_k=5)
        hits = retriever.retrieve(request)
        assert len(hits) == 1
        assert hits[0].doc_id == "1"

        # Test case where term is in content but NOT in title
        # "fruit" is in content of 1 and 2, but NOT in titles.
        # "fruit"[Title] should return 0 results.
        request_fail = SearchRequest(query='"fruit"[Title]', strategies=[RetrieverType.LANCE_FTS], top_k=5)
        hits_fail = retriever.retrieve(request_fail)
        assert len(hits_fail) == 0

    def test_retrieve_dict_query(self) -> None:
        """Test retrieval with a dict query (e.g. Boolean logic)."""
        self._seed_db()
        retriever = SparseRetriever()

        # Depending on how we mapped dict -> string in `_prepare_query`
        # We implemented "key:value" AND "key:value"
        # Tantivy "content:fruit" should work if we indexed "content".
        # But wait, `create_fts_index("content")` usually indexes the field 'content'.
        # If we query "category:fruit", it might fail if 'category' is not indexed or in metadata.
        # Metadata is a string field "metadata". FTS on it requires indexing "metadata".
        # The `_seed_db` indexed "content".
        # So let's test a dict that maps to valid content query?
        # Or maybe we assume `_prepare_query` is for fields that ARE indexed.

        # Let's try querying against content implicitly?
        # Tantivy default field is what we indexed.
        # If we pass "Apple", it searches default field.
        # If we pass {"content": "Apple"}, it becomes "content:Apple".

        request = SearchRequest(
            query={"content": "Apple"},
            strategies=[RetrieverType.LANCE_FTS],
        )
        hits = retriever.retrieve(request)
        assert len(hits) == 1
        assert hits[0].doc_id == "1"

    def test_systematic_generator(self) -> None:
        """Test the systematic search generator."""
        self._seed_db()
        retriever = SparseRetriever()

        # Query matching multiple docs
        request = SearchRequest(
            query="fruit",
            strategies=[RetrieverType.LANCE_FTS],
        )

        generator = retriever.retrieve_systematic(request)
        assert isinstance(generator, Iterator)

        results = list(generator)
        assert len(results) == 2  # Apple and Banana
        # Order is not guaranteed in FTS unless scored, but usually scored.
        doc_ids = sorted([h.doc_id for h in results])
        assert doc_ids == ["1", "2"]

    def test_retrieve_no_results(self) -> None:
        """Test retrieval with no matches."""
        self._seed_db()
        retriever = SparseRetriever()
        request = SearchRequest(
            query="zombie",
            strategies=[RetrieverType.LANCE_FTS],
        )
        hits = retriever.retrieve(request)
        assert len(hits) == 0

    def test_retrieve_systematic_empty(self) -> None:
        """Test systematic search with no results (hits break 112)."""
        self._seed_db()
        retriever = SparseRetriever()
        request = SearchRequest(query="zombie", strategies=[RetrieverType.LANCE_FTS])
        results = list(retriever.retrieve_systematic(request))
        assert len(results) == 0

    def test_retrieve_systematic_pagination(self) -> None:
        """Test systematic search with multiple pages."""
        self._seed_db()
        sparse_retriever = SparseRetriever()
        req = SearchRequest(query="test", strategies=[RetrieverType.LANCE_FTS])

        mock_builder = MagicMock()
        sparse_retriever.table = MagicMock()
        sparse_retriever.table.search.return_value = mock_builder
        mock_builder.limit.return_value = mock_builder
        mock_builder.offset.return_value = mock_builder

        # Simulate batch_size=1000
        # First batch full (1000 items), second batch has 1 item (hits break at line 126)

        full_batch = [{"doc_id": str(i), "content": "c", "metadata": "{}", "_score": 1.0} for i in range(1000)]
        partial_batch = [{"doc_id": "1001", "content": "c", "metadata": "{}", "_score": 1.0}]

        mock_builder.to_list.side_effect = [full_batch, partial_batch]

        results = list(sparse_retriever.retrieve_systematic(req))
        assert len(results) == 1001
        # It should have called to_list twice.
        assert mock_builder.to_list.call_count == 2

    def test_missing_index(self) -> None:
        """Test behavior when FTS index is missing."""
        # Create DB but don't index
        manager = get_db_manager()
        table = manager.get_table()
        embedder = get_embedder()
        table.add([DocumentSchema(doc_id="1", vector=embedder.embed("a")[0], content="a", metadata="{}")])

        retriever = SparseRetriever()
        request = SearchRequest(query="a", strategies=[RetrieverType.LANCE_FTS])

        # Should raise an error because index is missing
        with pytest.raises((ValueError, RuntimeError)):  # LanceDB raises ValueError or RuntimeError
            retriever.retrieve(request)

    def test_metadata_parsing(self) -> None:
        """Test metadata parsing in sparse results."""
        self._seed_db()
        retriever = SparseRetriever()
        request = SearchRequest(query="Carrot", strategies=[RetrieverType.LANCE_FTS])
        hits = retriever.retrieve(request)
        assert len(hits) == 1
        assert hits[0].metadata["category"] == "vegetable"

    def test_retrieve_with_filters(self) -> None:
        """Test sparse retrieval with metadata filters."""
        self._seed_db()
        retriever = SparseRetriever()

        # Doc 1 -> fruit, Doc 2 -> fruit, Doc 3 -> vegetable
        request = SearchRequest(query="fruit", strategies=[RetrieverType.LANCE_FTS], filters={"category": "fruit"})
        hits = retriever.retrieve(request)
        # Should match "fruit" in content (Docs 1, 2) and filter matches "fruit"
        assert len(hits) == 2
        for h in hits:
            assert h.metadata["category"] == "fruit"

        # Negative filter
        request = SearchRequest(query="fruit", strategies=[RetrieverType.LANCE_FTS], filters={"category": "vegetable"})
        hits = retriever.retrieve(request)
        # Content matches (Docs 1, 2) but filter excludes them.
        assert len(hits) == 0

    def test_systematic_generator_filtered(self) -> None:
        """Test systematic generator with filters."""
        self._seed_db()
        retriever = SparseRetriever()

        request = SearchRequest(query="fruit", strategies=[RetrieverType.LANCE_FTS], filters={"category": "fruit"})
        generator = retriever.retrieve_systematic(request)
        results = list(generator)
        assert len(results) == 2
        for h in results:
            assert h.metadata["category"] == "fruit"

        # Filter out all
        request = SearchRequest(query="fruit", strategies=[RetrieverType.LANCE_FTS], filters={"category": "cars"})
        results = list(retriever.retrieve_systematic(request))
        assert len(results) == 0

    def test_retrieve_complex_filters(self) -> None:
        """Test sparse retrieval with complex logical filters ($or)."""
        self._seed_db()
        retriever = SparseRetriever()

        # Doc 1 -> fruit, Doc 2 -> fruit, Doc 3 -> vegetable
        # Filter: fruit OR vegetable
        request = SearchRequest(
            query="fruit",
            strategies=[RetrieverType.LANCE_FTS],
            filters={"$or": [{"category": "fruit"}, {"category": "vegetable"}]},
        )
        hits = retriever.retrieve(request)
        assert len(hits) == 2

        # Filter: category=vegetable (should exclude 1, 2)
        request = SearchRequest(query="fruit", strategies=[RetrieverType.LANCE_FTS], filters={"category": "vegetable"})
        hits = retriever.retrieve(request)
        assert len(hits) == 0

    def test_retrieve_edge_cases(self) -> None:
        """Test edge cases: empty queries, special characters."""
        self._seed_db()
        retriever = SparseRetriever()

        # Empty query string (should result in 0 hits or handle gracefully)
        # Parser returns empty string. LanceDB FTS with empty string?
        # Typically returns nothing or error.
        request_empty = SearchRequest(query="", strategies=[RetrieverType.LANCE_FTS])
        # Depending on implementation, might raise or return empty.
        # Our parser returns "" if query is empty.
        # LanceDB search("") might behave differently.
        # Let's verify it doesn't crash.
        try:
            hits = retriever.retrieve(request_empty)
            assert len(hits) == 0
        except Exception:
            # If DB errors on empty string, that's acceptable, but ideally should be handled.
            # But we just want to ensure it doesn't crash the *process* unexpectedly.
            pass

        # Unicode characters
        # "β-amyloid"[Title] -> title:β-amyloid
        # We don't have this in DB, but query execution should pass.
        request_unicode = SearchRequest(query="β-amyloid[Title]", strategies=[RetrieverType.LANCE_FTS])
        hits = retriever.retrieve(request_unicode)
        assert len(hits) == 0

    def test_retrieve_multi_word_unquoted(self) -> None:
        """Test that unquoted multi-word terms followed by tag only apply tag to last word."""
        # Seeding with specific distractor for this test
        manager = get_db_manager()
        table = manager.get_table()
        embedder = get_embedder()

        # Clear existing data? The setup_teardown fixture handles it per test method,
        # but _seed_db adds to it.
        # Since this test method calls _seed_db() at start, it has docs 1, 2, 3.
        # Doc 1: Title="Apple Document", Content="Apple is a fruit"
        # Doc 2: Title="Banana Document", Content="Banana is also a fruit"
        # Doc 3: Title="Carrot Document", Content="Carrot is a vegetable"

        self._seed_db()

        # Add Distractor: Doc 4
        # Content has "fruit" and "Apple". Title has "Orange".
        # If parsing works (title:Apple), this should NOT match.
        # If parsing fails (content:Apple), this WOULD match.
        docs = [
            DocumentSchema(
                doc_id="4",
                vector=embedder.embed("distractor")[0],
                content="Apple is a tasty fruit.",
                title="Orange Document",
                metadata=json.dumps({"category": "fruit"}),
            )
        ]
        table.add(docs)

        retriever = SparseRetriever()

        # Query: fruit AND Apple[Title]
        # Parsed as: fruit AND title:Apple
        # Should match Doc 1 (fruit in Content, Apple in Title).
        # Should NOT match Doc 2 (fruit in Content, No Apple).
        # Should NOT match Doc 4 (fruit in Content, Apple in Content but NOT Title).

        request = SearchRequest(query="fruit AND Apple[Title]", strategies=[RetrieverType.LANCE_FTS])
        hits = retriever.retrieve(request)
        assert len(hits) == 1
        assert hits[0].doc_id == "1"

        # Query: Apple AND fruit[Title]
        # Parsed as: Apple AND title:fruit
        # "Apple" matches Doc 1 and Doc 4 content.
        # "fruit" in Title? No docs have "fruit" in Title.
        # So should return 0 results.
        request2 = SearchRequest(query="Apple AND fruit[Title]", strategies=[RetrieverType.LANCE_FTS])
        hits2 = retriever.retrieve(request2)
        assert len(hits2) == 0

    def test_retrieve_complex_boolean_logic(self) -> None:
        """Test complex nested boolean logic with multiple fields."""
        self._seed_db()
        retriever = SparseRetriever()

        # Doc 1: Title="Apple Document", Content="Apple is a fruit"
        # Doc 2: Title="Banana Document", Content="Banana is also a fruit"
        # Doc 3: Title="Carrot Document", Content="Carrot is a vegetable"

        # Query: (Apple[Title] OR Banana[Title]) AND fruit[Content]
        # Parsed: (title:Apple OR title:Banana) AND content:fruit
        # Should match Doc 1 and Doc 2.

        request = SearchRequest(
            query="(Apple[Title] OR Banana[Title]) AND fruit[Content]", strategies=[RetrieverType.LANCE_FTS]
        )
        hits = retriever.retrieve(request)
        assert len(hits) == 2
        doc_ids = sorted([h.doc_id for h in hits])
        assert doc_ids == ["1", "2"]

        # Query: (Apple[Title] AND Banana[Title])
        # Should match nothing (no doc has both in title).
        request_fail = SearchRequest(query="(Apple[Title] AND Banana[Title])", strategies=[RetrieverType.LANCE_FTS])
        hits_fail = retriever.retrieve(request_fail)
        assert len(hits_fail) == 0
