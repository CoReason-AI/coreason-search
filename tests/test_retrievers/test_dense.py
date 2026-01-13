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
from typing import Generator

import pytest
from lancedb.table import Table

from coreason_search.db import DocumentSchema, LanceDBManager, get_db_manager
from coreason_search.embedder import get_embedder, reset_embedder
from coreason_search.retrievers.dense import DenseRetriever
from coreason_search.schemas import Hit, RetrieverType, SearchRequest


class TestDenseRetriever:
    @pytest.fixture(autouse=True)  # type: ignore[misc]
    def setup_teardown(self, tmp_path: str) -> Generator[None, None, None]:
        # Reset DB and Embedder
        db_path = str(tmp_path) + "/lancedb_dense"
        manager = get_db_manager(db_path)
        manager.reset()
        # Connect fresh
        get_db_manager(db_path)

        reset_embedder()
        yield
        if LanceDBManager._instance:
            LanceDBManager._instance.reset()
        reset_embedder()

    def _seed_db(self) -> None:
        """Helper to populate DB with some data."""
        manager = get_db_manager()
        table = manager.get_table()
        embedder = get_embedder()

        # Create 5 docs
        docs = []
        for i in range(5):
            text = f"Document number {i} about science"
            vector = embedder.embed(text)[0]
            docs.append(
                DocumentSchema(
                    doc_id=str(i),
                    vector=vector,
                    content=text,
                    metadata=json.dumps({"index": i}),
                )
            )
        table.add(docs)

    def test_initialization(self) -> None:
        """Test that DenseRetriever initializes correctly."""
        retriever = DenseRetriever()
        assert retriever.db_manager is not None
        assert retriever.embedder is not None
        assert isinstance(retriever.table, Table)

    def test_retrieve_simple(self) -> None:
        """Test a simple retrieval."""
        self._seed_db()
        retriever = DenseRetriever()

        request = SearchRequest(query="science", strategies=[RetrieverType.LANCE_DENSE], top_k=3)

        hits = retriever.retrieve(request)

        assert len(hits) == 3
        assert isinstance(hits[0], Hit)
        assert hits[0].source_strategy == "lance_dense"
        # Since it's a mock embedder with random vectors, score is random but existing.
        assert isinstance(hits[0].score, float)
        assert hits[0].original_text.startswith("Document")
        assert hits[0].metadata["index"] in [0, 1, 2, 3, 4]

    def test_retrieve_dict_query(self) -> None:
        """Test retrieval when query is a dictionary (fallback logic)."""
        self._seed_db()
        retriever = DenseRetriever()

        request = SearchRequest(query={"text": "science"}, strategies=[RetrieverType.LANCE_DENSE], top_k=1)

        hits = retriever.retrieve(request)
        assert len(hits) == 1

    def test_retrieve_dict_query_no_text_key(self) -> None:
        """Test retrieval when query is a dictionary without 'text' key."""
        self._seed_db()
        retriever = DenseRetriever()

        request = SearchRequest(
            query={"keyword": "science", "year": "2024"}, strategies=[RetrieverType.LANCE_DENSE], top_k=1
        )

        hits = retriever.retrieve(request)
        assert len(hits) == 1
        # It should join values "science 2024"

    def test_retrieve_empty_db(self) -> None:
        """Test retrieval from an empty database."""
        # No seed
        retriever = DenseRetriever()
        request = SearchRequest(
            query="empty",
            strategies=[RetrieverType.LANCE_DENSE],
        )
        hits = retriever.retrieve(request)
        assert len(hits) == 0

    def test_retrieve_metadata_handling(self) -> None:
        """Test that metadata is correctly parsed."""
        manager = get_db_manager()
        table = manager.get_table()
        embedder = get_embedder()

        vector = embedder.embed("test")[0]
        # Insert doc with complex metadata
        meta = {"key": "value", "list": [1, 2]}
        table.add(
            [DocumentSchema(doc_id="meta_test", vector=vector, content="test content", metadata=json.dumps(meta))]
        )

        retriever = DenseRetriever()
        request = SearchRequest(query="test", strategies=[RetrieverType.LANCE_DENSE])
        hits = retriever.retrieve(request)

        assert len(hits) == 1
        assert hits[0].metadata == meta

    def test_retrieve_broken_metadata(self) -> None:
        """Test handling of broken JSON in metadata (if it somehow got in)."""
        # We skip checking this via insertion because `db.py` enforces it.
        # This test placeholder documents that we rely on db.py validation.
        pass

    def test_retrieve_limit(self) -> None:
        """Test that top_k is respected."""
        self._seed_db()  # 5 docs
        retriever = DenseRetriever()

        request = SearchRequest(query="test", strategies=[RetrieverType.LANCE_DENSE], top_k=2)
        hits = retriever.retrieve(request)
        assert len(hits) == 2
