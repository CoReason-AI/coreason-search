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
from unittest.mock import MagicMock, patch

import pytest
from coreason_identity.models import UserContext

from coreason_search.config import Settings
from coreason_search.db import DocumentSchema, get_db_manager, reset_db_manager
from coreason_search.embedder import get_embedder, reset_embedder
from coreason_search.engine import SearchEngineAsync
from coreason_search.schemas import Hit, RetrieverType, SearchRequest, SearchResponse


class TestSearchEngineAsync:
    @pytest.fixture(autouse=True)
    def setup_teardown(self, tmp_path: str) -> Generator[None, None, None]:
        self.db_path = str(tmp_path) + "/lancedb_engine"
        reset_db_manager()
        get_db_manager(self.db_path)
        reset_embedder()
        yield
        reset_db_manager()
        reset_embedder()

    def _get_engine(self) -> SearchEngineAsync:
        """Helper to initialize engine with the correct DB URI."""
        config = Settings(database_uri=self.db_path)
        return SearchEngineAsync(config)

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
            DocumentSchema(
                doc_id="2",
                vector=embedder.embed("banana")[0],
                content="Banana bread",
                metadata=json.dumps({"type": "food"}),
            ),
        ]
        table.add(docs)
        try:
            table.create_fts_index("content")
        except Exception:
            pass

    @pytest.mark.asyncio
    async def test_execute_standard_flow(self) -> None:
        """Test standard RAG execution flow."""
        self._seed_db()
        engine = self._get_engine()
        async with engine:
            request = SearchRequest(
                query="apple",
                strategies=[RetrieverType.LANCE_DENSE, RetrieverType.LANCE_FTS],
                top_k=5,
                fusion_enabled=True,
                rerank_enabled=True,
                distill_enabled=True,
            )

            response = await engine.execute(request)

            assert isinstance(response, SearchResponse)
            assert len(response.hits) >= 1

            matched_hit = next((h for h in response.hits if h.original_text == "Apple pie"), None)
            if matched_hit:
                assert matched_hit.distilled_text == "Apple pie"

    @pytest.mark.asyncio
    async def test_execute_systematic_flow(self) -> None:
        """Test systematic search generator."""
        self._seed_db()
        engine = self._get_engine()
        async with engine:
            request = SearchRequest(
                query="apple",
                strategies=[RetrieverType.LANCE_FTS],
            )

            results = []
            async for hit in engine.execute_systematic(request):
                results.append(hit)

            assert len(results) >= 1
            assert results[0].doc_id == "1"

    @pytest.mark.asyncio
    async def test_execute_systematic_audit(self) -> None:
        """Test systematic search execution with audit logging."""
        self._seed_db()
        engine = self._get_engine()

        req = SearchRequest(
            query="test",
            strategies=[RetrieverType.LANCE_FTS],
            top_k=5,
        )

        engine.sparse_retriever = MagicMock()
        mock_hit = Hit(
            doc_id="1",
            content="c",
            original_text="c",
            distilled_text="",
            score=1.0,
            source_strategy="sparse",
            metadata={},
        )
        engine.sparse_retriever.retrieve_systematic.return_value = iter([mock_hit])
        engine.sparse_retriever.get_table_version.return_value = 123

        with patch.object(engine.veritas, "log_audit") as mock_audit:
            results = []
            async with engine:
                async for hit in engine.execute_systematic(req):
                    results.append(hit)

            assert len(results) == 1
            assert results[0].doc_id == "1"

            assert mock_audit.call_count == 2
            start_call = mock_audit.call_args_list[0]
            assert start_call[0][0] == "SYSTEMATIC_SEARCH_START"
            assert start_call[0][1]["snapshot_id"] == 123

            complete_call = mock_audit.call_args_list[1]
            assert complete_call[0][0] == "SYSTEMATIC_SEARCH_COMPLETE"
            assert complete_call[0][1]["total_found"] == 1

    @pytest.mark.asyncio
    async def test_execute_systematic_audit_exception(self) -> None:
        """Test systematic search audit fallback when DB version fails."""
        self._seed_db()
        engine = self._get_engine()
        req = SearchRequest(query="test", strategies=[RetrieverType.LANCE_FTS])

        engine.sparse_retriever = MagicMock()
        engine.sparse_retriever.retrieve_systematic.return_value = iter([])
        engine.sparse_retriever.get_table_version.side_effect = Exception("DB Error")

        with patch.object(engine.veritas, "log_audit") as mock_audit:
            async with engine:
                async for _ in engine.execute_systematic(req):
                    pass

            start_call = mock_audit.call_args_list[0]
            assert start_call[0][0] == "SYSTEMATIC_SEARCH_START"
            assert start_call[0][1]["snapshot_id"] == -1

    @pytest.mark.asyncio
    async def test_unknown_strategy_and_error_handling(self) -> None:
        """Test that unknown strategies are handled gracefully."""
        self._seed_db()
        engine = self._get_engine()
        async with engine:
            request = SearchRequest(
                query="Protein X",
                strategies=[RetrieverType.GRAPH_NEIGHBOR, RetrieverType.LANCE_DENSE],
                top_k=5,
            )

            response = await engine.execute(request)
            assert len(response.hits) >= 1
            sources = {h.source_strategy for h in response.hits}
            assert "graph_neighbor" in sources

    @pytest.mark.asyncio
    async def test_fusion_disabled(self) -> None:
        """Test execution with fusion disabled."""
        self._seed_db()
        engine = self._get_engine()
        async with engine:
            request = SearchRequest(
                query="apple",
                strategies=[RetrieverType.LANCE_DENSE, RetrieverType.LANCE_FTS],
                top_k=5,
                fusion_enabled=False,
            )

            response = await engine.execute(request)
            assert len(response.hits) >= 1

    @pytest.mark.asyncio
    async def test_rerank_distill_disabled(self) -> None:
        """Test disabling rerank and distill."""
        self._seed_db()
        engine = self._get_engine()
        async with engine:
            request = SearchRequest(
                query="apple", strategies=[RetrieverType.LANCE_DENSE], rerank_enabled=False, distill_enabled=False
            )

            response = await engine.execute(request)
            assert len(response.hits) >= 1
            assert response.hits[0].distilled_text == ""

    @pytest.mark.asyncio
    async def test_systematic_dense_warning(self) -> None:
        """Test systematic search with dense strategy (should work but fallback)."""
        self._seed_db()
        engine = self._get_engine()
        async with engine:
            request = SearchRequest(
                query="apple",
                strategies=[RetrieverType.LANCE_DENSE],
            )
            results = []
            async for hit in engine.execute_systematic(request):
                results.append(hit)
            assert len(results) >= 1

    @pytest.mark.asyncio
    async def test_strategy_exception_handling(self) -> None:
        """Test that exceptions in strategies are caught and logged."""
        self._seed_db()
        engine = self._get_engine()

        # Mock dense retriever to raise exception
        class BrokenRetriever:
            def retrieve(self, request: SearchRequest) -> list:  # type: ignore[type-arg]
                raise ValueError("Broken")

        engine.dense_retriever = BrokenRetriever()  # type: ignore

        request = SearchRequest(
            query="apple",
            strategies=[RetrieverType.LANCE_DENSE, RetrieverType.LANCE_FTS],
            top_k=5,
        )

        async with engine:
            response = await engine.execute(request)
            assert len(response.hits) >= 1

    @pytest.mark.asyncio
    async def test_execute_passes_user_context(self) -> None:
        """Test that user_context is passed to Scout.distill."""
        self._seed_db()
        engine = self._get_engine()

        user_context = UserContext(user_id="admin", email="admin@example.com", scopes=["admin"])
        request = SearchRequest(
            query="apple", strategies=[RetrieverType.LANCE_DENSE], user_context=user_context, distill_enabled=True
        )

        # Mock the scout to verify call args
        with patch.object(engine.scout, "distill", wraps=engine.scout.distill) as mock_distill:
            async with engine:
                await engine.execute(request)

            # Verify distill was called with user_context
            assert mock_distill.call_count == 1
            _, kwargs = mock_distill.call_args
            assert kwargs.get("user_context") == user_context
