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
from typing import Generator, Iterator
from unittest.mock import MagicMock

import pytest

from coreason_search.db import DocumentSchema, LanceDBManager, get_db_manager
from coreason_search.embedder import get_embedder, reset_embedder
from coreason_search.retrievers.sparse import SparseRetriever
from coreason_search.schemas import RetrieverType, SearchRequest


class TestSparseRetriever:
    @pytest.fixture(autouse=True)  # type: ignore[misc]
    def setup_teardown(self, tmp_path: str) -> Generator[None, None, None]:
        db_path = str(tmp_path) + "/lancedb_sparse"
        manager = get_db_manager(db_path)
        manager.reset()
        get_db_manager(db_path)
        reset_embedder()
        yield
        if LanceDBManager._instance:
            LanceDBManager._instance.reset()
        reset_embedder()

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
                metadata=json.dumps({"category": "fruit"}),
            ),
            DocumentSchema(
                doc_id="2",
                vector=embedder.embed("banana")[0],
                content="Banana is also a fruit.",
                metadata=json.dumps({"category": "fruit"}),
            ),
            DocumentSchema(
                doc_id="3",
                vector=embedder.embed("carrot")[0],
                content="Carrot is a vegetable.",
                metadata=json.dumps({"category": "vegetable"}),
            ),
        ]
        table.add(docs)
        # Create FTS index
        table.create_fts_index("content")

    def test_retrieve_simple(self) -> None:
        """Test simple FTS retrieval."""
        self._seed_db()
        retriever = SparseRetriever()

        request = SearchRequest(query="Apple", strategies=[RetrieverType.LANCE_FTS], top_k=5)

        hits = retriever.retrieve(request)
        assert len(hits) >= 1
        assert hits[0].content == "Apple is a fruit."
        assert hits[0].source_strategy == "lance_fts"

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
        # We query for "is" which is in "Apple is a fruit", "Banana is also...", "Carrot is..."
        # If "is" is a stopword, maybe query "fruit" or "vegetable" in text?
        # But text query matches hits first.
        # Let's use "*" or something broad? LanceDB FTS might not support *.
        # Use "fruit" -> matches Doc 1, 2.
        # Use "vegetable" -> matches Doc 3.
        # If we query "fruit", we get Doc 1, 2. Filter: fruit OR vegetable.
        # Doc 3 (vegetable) is NOT retrieved by FTS query "fruit".
        # So filter logic only applies to retrieved items.
        # Query "fruit OR vegetable" in FTS?
        # Let's query "a" (in "Apple is a fruit", "Banana is also a fruit", "Carrot is a vegetable").
        # "a" might be stopword.
        # Let's query "fruit" and check filter.

        # Query: "fruit" (matches 1, 2)
        # Filter: category=fruit OR category=vegetable
        # Matches 1, 2.
        # Doc 3 is not retrieved, so filter doesn't matter for it.

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
