# Copyright (c) 2025 CoReason, Inc.
#
# This software is proprietary and dual-licensed.
# Licensed under the Prosperity Public License 3.0 (the "License").
# A copy of the license is available at https://prosperitylicense.com/versions/3.0.0
# For details, see the LICENSE file.
# Commercial use beyond a 30-day trial requires a separate license.
#
# Source Code: https://github.com/CoReason-AI/coreason_search

from typing import Any, Generator, List
from unittest.mock import MagicMock, patch

import pytest

from coreason_search.db import get_db_manager, reset_db_manager
from coreason_search.embedder import reset_embedder
from coreason_search.engine import SearchEngine
from coreason_search.schemas import RetrieverType, SearchRequest
from coreason_search.veritas import reset_veritas_client


class TestComplexSystematic:
    @pytest.fixture(autouse=True)  # type: ignore[misc]
    def setup_teardown(self, tmp_path: str) -> Generator[None, None, None]:
        db_path = str(tmp_path) + "/lancedb_complex"
        reset_db_manager()
        get_db_manager(db_path)
        reset_embedder()
        reset_veritas_client()
        yield
        reset_db_manager()
        reset_embedder()
        reset_veritas_client()

    def _mock_sparse_batches(self, engine: SearchEngine, batches: List[List[Any]]) -> MagicMock:
        """Helper to mock sparse retriever batches."""
        mock_builder = MagicMock()
        engine.sparse_retriever.table = MagicMock()
        engine.sparse_retriever.table.search.return_value = mock_builder
        mock_builder.limit.return_value = mock_builder
        mock_builder.offset.return_value = mock_builder
        mock_builder.to_list.side_effect = batches
        return mock_builder

    def test_pagination_exact_multiple(self) -> None:
        """
        Test streaming when results are an exact multiple of batch size (1000).
        Scenario: 2000 items -> Batch 1 (1000), Batch 2 (1000), Batch 3 (Empty).
        """
        engine = SearchEngine()
        req = SearchRequest(query="test", strategies=[RetrieverType.LANCE_FTS])

        # Batch size is hardcoded to 1000 in sparse.py
        batch_1 = [{"doc_id": f"a_{i}", "content": "c", "metadata": "{}", "_score": 1.0} for i in range(1000)]
        batch_2 = [{"doc_id": f"b_{i}", "content": "c", "metadata": "{}", "_score": 1.0} for i in range(1000)]
        batch_3: List[Any] = []

        self._mock_sparse_batches(engine, [batch_1, batch_2, batch_3])

        results = list(engine.execute_systematic(req))
        assert len(results) == 2000
        assert results[0].doc_id == "a_0"
        assert results[1999].doc_id == "b_999"

    def test_generator_interruption_logging(self) -> None:
        """
        Test that Veritas logs correct partial count when generator is closed early.
        """
        engine = SearchEngine()
        req = SearchRequest(query="test", strategies=[RetrieverType.LANCE_FTS])

        # Mock 100 items
        batch_1 = [{"doc_id": str(i), "content": "c", "metadata": "{}", "_score": 1.0} for i in range(100)]
        self._mock_sparse_batches(engine, [batch_1, []])

        with patch.object(engine.veritas, "log_audit") as mock_audit:
            gen = engine.execute_systematic(req)

            # Consume only 5 items
            for _ in range(5):
                next(gen)

            # Close generator explicitly (simulating client disconnect/stop)
            if hasattr(gen, "close"):
                gen.close()

            # Verify 'SYSTEMATIC_SEARCH_COMPLETE' was logged.
            # Count logic: Generator yields item, then control returns to caller.
            # Caller calls next() -> yield resumes -> count incremented -> yields next.
            # If closed at yield, the increment loop doesn't happen for the item at yield.
            # So if we consumed 5 items, the counter might be 4 depending on where 'count += 1' is.
            # With 'yield hit; count += 1', the 5th item is yielded but loop closed before increment.
            # So count = 4.

            complete_calls = [call for call in mock_audit.call_args_list if call[0][0] == "SYSTEMATIC_SEARCH_COMPLETE"]
            assert len(complete_calls) == 1
            data = complete_calls[0][0][1]
            assert data["total_found"] == 4

    def test_audit_failure_propagates(self) -> None:
        """
        Test that failure in audit logging raises exception (Auditing is critical).
        """
        engine = SearchEngine()
        req = SearchRequest(query="test", strategies=[RetrieverType.LANCE_FTS])

        engine.sparse_retriever.table = MagicMock()  # Ensure no DB error

        with patch.object(engine.veritas, "log_audit", side_effect=ValueError("Audit Service Down")):
            with pytest.raises(ValueError, match="Audit Service Down"):
                next(engine.execute_systematic(req))

    def test_complex_filter_streaming(self) -> None:
        """
        Test filtering across multiple batches.
        Batch 1: 3 items (keep 1)
        Batch 2: 3 items (keep 0 - empty filtered batch)
        Batch 3: 3 items (keep 2)
        Batch 4: Empty (End)
        """
        engine = SearchEngine()

        # Override batch size to 3 for this test
        engine.sparse_retriever.systematic_batch_size = 3

        # Filter: val > 10
        req = SearchRequest(query="test", strategies=[RetrieverType.LANCE_FTS], filters={"val": {"$gt": 10}})

        def item(i: int, val: int) -> dict[str, Any]:
            return {"doc_id": str(i), "content": "c", "metadata": f'{{"val": {val}}}', "_score": 1.0}

        # Batch 1: [5, 15, 5] -> Keep 15
        b1 = [item(1, 5), item(2, 15), item(3, 5)]
        # Batch 2: [5, 5, 5] -> Keep None
        b2 = [item(4, 5), item(5, 5), item(6, 5)]
        # Batch 3: [20, 30, 5] -> Keep 20, 30
        b3 = [item(7, 20), item(8, 30), item(9, 5)]
        # Batch 4: End
        b4: List[Any] = []

        self._mock_sparse_batches(engine, [b1, b2, b3, b4])

        results = list(engine.execute_systematic(req))

        assert len(results) == 3
        ids = [r.doc_id for r in results]
        assert ids == ["2", "7", "8"]
