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
    @pytest.fixture(autouse=True)
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
        # Access internal async component
        engine._async.sparse_retriever.table = MagicMock()
        engine._async.sparse_retriever.table.search.return_value = mock_builder
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

        with engine:
            results = list(engine.execute_systematic(req))

        assert len(results) == 2000
        assert results[0].doc_id == "a_0"
        assert results[1999].doc_id == "b_999"

    def test_audit_failure_propagates(self) -> None:
        """
        Test that failure in audit logging raises exception (Auditing is critical).
        """
        engine = SearchEngine()
        req = SearchRequest(query="test", strategies=[RetrieverType.LANCE_FTS])

        engine._async.sparse_retriever.table = MagicMock()

        # We can test this on sync facade (even if it collects).
        # Just mock veritas on the async instance.
        with patch.object(engine._async.veritas, "log_audit", side_effect=ValueError("Audit Service Down")):
            with engine:
                with pytest.raises(ValueError, match="Audit Service Down"):
                    # This will fail during collection inside execute_systematic
                    list(engine.execute_systematic(req))

    def test_complex_filter_streaming(self) -> None:
        """
        Test filtering across multiple batches.
        """
        engine = SearchEngine()

        engine._async.sparse_retriever.systematic_batch_size = 3

        req = SearchRequest(query="test", strategies=[RetrieverType.LANCE_FTS], filters={"val": {"$gt": 10}})

        def item(i: int, val: int) -> dict[str, Any]:
            return {"doc_id": str(i), "content": "c", "metadata": f'{{"val": {val}}}', "_score": 1.0}

        b1 = [item(1, 5), item(2, 15), item(3, 5)]
        b2 = [item(4, 5), item(5, 5), item(6, 5)]
        b3 = [item(7, 20), item(8, 30), item(9, 5)]
        b4: List[Any] = []

        self._mock_sparse_batches(engine, [b1, b2, b3, b4])

        with engine:
            results = list(engine.execute_systematic(req))

        assert len(results) == 3
        ids = [r.doc_id for r in results]
        assert ids == ["2", "7", "8"]
