from typing import Iterator
from unittest.mock import MagicMock

import pytest

from coreason_search.engine import SearchEngine
from coreason_search.schemas import ExecutionMode, Hit, RetrieverType, SearchRequest, SearchResponse


@pytest.fixture  # type: ignore[misc]
def search_engine() -> SearchEngine:
    engine = SearchEngine()
    # Mock components to avoid external dependencies or complex setup
    engine.dense_retriever = MagicMock()
    engine.sparse_retriever = MagicMock()
    engine.graph_retriever = MagicMock()
    engine.fusion_engine = MagicMock()
    engine.reranker = MagicMock()
    engine.scout = MagicMock()
    engine.veritas = MagicMock()
    return engine


def test_execute_standard_mode(search_engine: SearchEngine) -> None:
    """Test that STANDARD execution mode returns a SearchResponse."""
    request = SearchRequest(query="test", strategies=[RetrieverType.LANCE_DENSE], execution_mode=ExecutionMode.STANDARD)

    # Setup mock returns
    search_engine.dense_retriever.retrieve = MagicMock(return_value=[])  # type: ignore[method-assign]
    search_engine.fusion_engine.fuse = MagicMock(return_value=[])  # type: ignore[method-assign]
    search_engine.reranker.rerank = MagicMock(return_value=[])  # type: ignore[method-assign]
    search_engine.scout.distill = MagicMock(return_value=[])  # type: ignore[method-assign]

    response = search_engine.execute(request)

    assert isinstance(response, SearchResponse)
    search_engine.dense_retriever.retrieve.assert_called_once()
    search_engine.sparse_retriever.retrieve_systematic.assert_not_called()  # type: ignore[attr-defined]


def test_execute_systematic_mode(search_engine: SearchEngine) -> None:
    """Test that SYSTEMATIC execution mode returns an Iterator."""
    request = SearchRequest(
        query="test[Title]", strategies=[RetrieverType.LANCE_FTS], execution_mode=ExecutionMode.SYSTEMATIC
    )

    # Mock sparse retriever to return a generator
    def mock_generator(req: SearchRequest) -> Iterator[Hit]:
        yield Hit(
            doc_id="1",
            content="test",
            original_text="test",
            distilled_text="",
            score=1.0,
            source_strategy="sparse",
            metadata={},
        )

    search_engine.sparse_retriever.retrieve_systematic = MagicMock(side_effect=mock_generator)  # type: ignore[method-assign]
    search_engine.sparse_retriever.get_table_version = MagicMock(return_value=1)  # type: ignore[method-assign]

    response = search_engine.execute(request)

    assert isinstance(response, Iterator)

    # Consume the iterator to verify content
    results = list(response)
    assert len(results) == 1
    assert results[0].doc_id == "1"

    search_engine.sparse_retriever.retrieve_systematic.assert_called_once()
    search_engine.dense_retriever.retrieve.assert_not_called()  # type: ignore[attr-defined]
    search_engine.veritas.log_audit.assert_called()  # type: ignore[attr-defined]


def test_default_execution_mode_is_standard(search_engine: SearchEngine) -> None:
    """Test that default execution mode is STANDARD."""
    request = SearchRequest(query="test", strategies=[RetrieverType.LANCE_DENSE])

    assert request.execution_mode == ExecutionMode.STANDARD

    # Setup mock returns
    search_engine.dense_retriever.retrieve = MagicMock(return_value=[])  # type: ignore[method-assign]
    search_engine.fusion_engine.fuse = MagicMock(return_value=[])  # type: ignore[method-assign]
    search_engine.reranker.rerank = MagicMock(return_value=[])  # type: ignore[method-assign]
    search_engine.scout.distill = MagicMock(return_value=[])  # type: ignore[method-assign]

    response = search_engine.execute(request)
    assert isinstance(response, SearchResponse)
