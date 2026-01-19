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
from unittest.mock import MagicMock, patch

import pytest

from coreason_search.db import DocumentSchema, get_db_manager, reset_db_manager
from coreason_search.embedder import get_embedder, reset_embedder
from coreason_search.engine import SearchEngine
from coreason_search.schemas import Hit, RetrieverType, SearchRequest, SearchResponse


class TestSearchEngine:
    @pytest.fixture(autouse=True)  # type: ignore[misc]
    def setup_teardown(self, tmp_path: str) -> Generator[None, None, None]:
        db_path = str(tmp_path) + "/lancedb_engine"
        reset_db_manager()
        get_db_manager(db_path)
        reset_embedder()
        yield
        reset_db_manager()
        reset_embedder()

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

    def test_execute_standard_flow(self) -> None:
        """Test standard RAG execution flow."""
        self._seed_db()
        engine = SearchEngine()

        request = SearchRequest(
            query="apple",
            strategies=[RetrieverType.LANCE_DENSE, RetrieverType.LANCE_FTS],
            top_k=5,
            fusion_enabled=True,
            rerank_enabled=True,
            distill_enabled=True,
        )

        response = engine.execute(request)

        assert isinstance(response, SearchResponse)
        assert len(response.hits) >= 1
        # Check hits have distilled text (Scout ran)
        assert response.hits[0].distilled_text.endswith("...")

    def test_execute_systematic_flow(self) -> None:
        """Test systematic search generator."""
        self._seed_db()
        engine = SearchEngine()

        request = SearchRequest(
            query="apple",
            strategies=[RetrieverType.LANCE_FTS],
        )

        gen = engine.execute_systematic(request)
        assert isinstance(gen, Iterator)

        results = list(gen)
        assert len(results) >= 1
        assert results[0].doc_id == "1"

    def test_execute_systematic_audit(self) -> None:
        """Test systematic search execution with audit logging."""
        self._seed_db()
        engine = SearchEngine()

        req = SearchRequest(
            query="test",
            strategies=[RetrieverType.LANCE_FTS],
            top_k=5,
        )

        # Mock sparse retriever generator and version
        # We need to mock the sparse_retriever on the engine instance we created
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

        # Spy on veritas
        with patch.object(engine.veritas, "log_audit") as mock_audit:
            gen = engine.execute_systematic(req)
            results = list(gen)

            assert len(results) == 1
            assert results[0].doc_id == "1"

            # Verify Audit Calls
            assert mock_audit.call_count == 2

            # Check Start
            start_call = mock_audit.call_args_list[0]
            assert start_call[0][0] == "SYSTEMATIC_SEARCH_START"
            assert start_call[0][1]["snapshot_id"] == 123

            # Check Complete
            complete_call = mock_audit.call_args_list[1]
            assert complete_call[0][0] == "SYSTEMATIC_SEARCH_COMPLETE"
            assert complete_call[0][1]["total_found"] == 1

    def test_execute_systematic_audit_exception(self) -> None:
        """Test systematic search audit fallback when DB version fails."""
        self._seed_db()
        engine = SearchEngine()
        req = SearchRequest(query="test", strategies=[RetrieverType.LANCE_FTS])

        engine.sparse_retriever = MagicMock()
        engine.sparse_retriever.retrieve_systematic.return_value = iter([])
        engine.sparse_retriever.get_table_version.side_effect = Exception("DB Error")

        with patch.object(engine.veritas, "log_audit") as mock_audit:
            list(engine.execute_systematic(req))

            # Check Start log has snapshot_id = -1
            start_call = mock_audit.call_args_list[0]
            assert start_call[0][0] == "SYSTEMATIC_SEARCH_START"
            assert start_call[0][1]["snapshot_id"] == -1

    def test_unknown_strategy_and_error_handling(self) -> None:
        """Test that unknown strategies are handled gracefully."""
        self._seed_db()
        engine = SearchEngine()

        # Mixed valid and invalid (Graph IS implemented now, but we can verify it doesn't crash)
        # Let's use a query that Graph mock recognizes ("Protein X") to verify Graph works mixed with Dense.
        request = SearchRequest(
            query="Protein X",
            strategies=[RetrieverType.GRAPH_NEIGHBOR, RetrieverType.LANCE_DENSE],
            top_k=5,
        )

        response = engine.execute(request)
        # Should succeed with Dense results AND Graph results
        # "Protein X" might not be in the vector DB (seeded with apples), but Graph should return hits.
        assert len(response.hits) >= 1
        sources = {h.source_strategy for h in response.hits}
        assert "graph_neighbor" in sources

        # Test truly unknown strategy?
        # Enum validation prevents passing strings not in Enum.
        # But we can try to force it if we bypass validation or if we just want to test exception handling.
        # The `execute` loop handles exceptions.

    def test_fusion_disabled(self) -> None:
        """Test execution with fusion disabled."""
        self._seed_db()
        engine = SearchEngine()

        request = SearchRequest(
            query="apple",
            strategies=[RetrieverType.LANCE_DENSE, RetrieverType.LANCE_FTS],
            top_k=5,
            fusion_enabled=False,
        )

        response = engine.execute(request)
        assert len(response.hits) >= 1

    def test_rerank_distill_disabled(self) -> None:
        """Test disabling rerank and distill."""
        self._seed_db()
        engine = SearchEngine()

        request = SearchRequest(
            query="apple", strategies=[RetrieverType.LANCE_DENSE], rerank_enabled=False, distill_enabled=False
        )

        response = engine.execute(request)
        assert len(response.hits) >= 1
        # Distilled text should be empty string (default) because Scout didn't run
        assert response.hits[0].distilled_text == ""

    def test_systematic_dense_warning(self) -> None:
        """Test systematic search with dense strategy (should work but fallback)."""
        self._seed_db()
        engine = SearchEngine()
        request = SearchRequest(
            query="apple",
            strategies=[RetrieverType.LANCE_DENSE],
        )
        gen = engine.execute_systematic(request)
        results = list(gen)
        assert len(results) >= 1

    def test_strategy_exception_handling(self) -> None:
        """Test that exceptions in strategies are caught and logged."""
        self._seed_db()
        engine = SearchEngine()

        # Mock dense retriever to raise exception
        class BrokenRetriever:
            def retrieve(self, request: SearchRequest) -> list:  # type: ignore[type-arg]
                raise ValueError("Broken")

        engine.dense_retriever = BrokenRetriever()  # type: ignore

        request = SearchRequest(
            query="apple",
            strategies=[RetrieverType.LANCE_DENSE, RetrieverType.LANCE_FTS],  # One broken, one works
            top_k=5,
        )

        # Should NOT raise exception, but log error and continue with FTS
        response = engine.execute(request)
        assert len(response.hits) >= 1
