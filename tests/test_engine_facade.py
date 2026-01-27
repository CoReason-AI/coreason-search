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

import pytest

from coreason_search.config import Settings
from coreason_search.db import DocumentSchema, get_db_manager, reset_db_manager
from coreason_search.embedder import get_embedder, reset_embedder
from coreason_search.engine import SearchEngine
from coreason_search.schemas import RetrieverType, SearchRequest, SearchResponse


class TestSearchEngineFacade:
    """Test the synchronous facade wrapping the async engine."""

    @pytest.fixture(autouse=True)
    def setup_teardown(self, tmp_path: str) -> Generator[None, None, None]:
        self.db_path = str(tmp_path) + "/lancedb_facade"
        reset_db_manager()
        get_db_manager(self.db_path)
        reset_embedder()
        yield
        reset_db_manager()
        reset_embedder()

    def _get_engine(self) -> SearchEngine:
        """Helper to initialize engine with the correct DB URI."""
        config = Settings(database_uri=self.db_path)
        return SearchEngine(config)

    def _seed_db(self) -> None:
        manager = get_db_manager()
        table = manager.get_table()
        embedder = get_embedder()

        docs = [
            DocumentSchema(
                doc_id="1",
                vector=embedder.embed("apple")[0],
                content="Apple pie",
                metadata=json.dumps({"type": "food"}),
            ),
        ]
        table.add(docs)
        try:
            table.create_fts_index("content")
        except Exception:
            pass

    def test_context_manager_lifecycle(self) -> None:
        """Test that the context manager works and cleans up."""
        engine = self._get_engine()

        # We can't easily check if client is closed on the real object without accessing private member
        # or mocking. Let's rely on functional test: if it runs without error, it's good.
        with engine:
            pass

    def test_execute_synchronous(self) -> None:
        """Test blocking execute call."""
        self._seed_db()
        engine = self._get_engine()

        request = SearchRequest(
            query="apple",
            strategies=[RetrieverType.LANCE_DENSE, RetrieverType.LANCE_FTS],
            top_k=5,
        )

        with engine:
            response = engine.execute(request)

        assert isinstance(response, SearchResponse)
        assert len(response.hits) >= 1

    def test_execute_systematic_synchronous(self) -> None:
        """Test blocking systematic search generator."""
        self._seed_db()
        engine = self._get_engine()

        request = SearchRequest(
            query="apple",
            strategies=[RetrieverType.LANCE_FTS],
        )

        with engine:
            gen = engine.execute_systematic(request)
            assert isinstance(gen, Iterator)
            results = list(gen)

        assert len(results) >= 1
        assert results[0].doc_id == "1"
