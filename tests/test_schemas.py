# Copyright (c) 2025 CoReason, Inc.
#
# This software is proprietary and dual-licensed.
# Licensed under the Prosperity Public License 3.0 (the "License").
# A copy of the license is available at https://prosperitylicense.com/versions/3.0.0
# For details, see the LICENSE file.
# Commercial use beyond a 30-day trial requires a separate license.
#
# Source Code: https://github.com/CoReason-AI/coreason_search

import pytest
from pydantic import ValidationError

from coreason_search.schemas import (
    EmbeddingConfig,
    Hit,
    RetrieverType,
    SearchRequest,
    SearchResponse,
)


class TestSchemas:
    def test_retriever_type_enum(self) -> None:
        """Test RetrieverType enum values."""
        assert RetrieverType.LANCE_DENSE.value == "lance_dense"
        assert RetrieverType.LANCE_FTS.value == "lance_fts"
        assert RetrieverType.GRAPH_NEIGHBOR.value == "graph_neighbor"

    def test_embedding_config_defaults(self) -> None:
        """Test EmbeddingConfig default values."""
        config = EmbeddingConfig()
        assert config.model_name == "Alibaba-NLP/gte-Qwen2-7B-instruct"
        assert config.context_length == 32768
        assert config.batch_size == 1

    def test_embedding_config_custom(self) -> None:
        """Test EmbeddingConfig with custom values."""
        config = EmbeddingConfig(model_name="custom-model", context_length=1024, batch_size=32)
        assert config.model_name == "custom-model"
        assert config.context_length == 1024
        assert config.batch_size == 32

    def test_search_request_valid_string_query(self) -> None:
        """Test SearchRequest with a string query."""
        request = SearchRequest(
            query="test query",
            strategies=[RetrieverType.LANCE_DENSE],
        )
        assert request.query == "test query"
        assert request.strategies == [RetrieverType.LANCE_DENSE]
        assert request.fusion_enabled is True  # default
        assert request.rerank_enabled is True  # default
        assert request.top_k == 5  # default
        assert request.filters is None  # default

    def test_search_request_valid_dict_query(self) -> None:
        """Test SearchRequest with a dictionary query."""
        query_dict = {"field": "value"}
        request = SearchRequest(
            query=query_dict,
            strategies=[RetrieverType.LANCE_FTS],
            top_k=10,
            filters={"year": 2024},
        )
        assert request.query == query_dict
        assert request.strategies == [RetrieverType.LANCE_FTS]
        assert request.top_k == 10
        assert request.filters == {"year": 2024}

    def test_search_request_invalid(self) -> None:
        """Test SearchRequest validation failure."""
        with pytest.raises(ValidationError):
            # strategies is required
            SearchRequest(query="test")  # type: ignore

    def test_hit_model(self) -> None:
        """Test Hit model instantiation."""
        hit = Hit(
            doc_id="doc1",
            content="content",
            score=0.95,
            source_strategy="dense",
            metadata={"key": "value"},
        )
        assert hit.doc_id == "doc1"
        assert hit.content == "content"
        assert hit.score == 0.95
        assert hit.source_strategy == "dense"
        assert hit.metadata == {"key": "value"}

    def test_search_response_model(self) -> None:
        """Test SearchResponse model instantiation."""
        hit1 = Hit(
            doc_id="doc1",
            content="c1",
            score=0.9,
            source_strategy="dense",
            metadata={},
        )
        response = SearchResponse(
            hits=[hit1],
            total_found=1,
            execution_time_ms=100.5,
            provenance_hash="hash123",
        )
        assert len(response.hits) == 1
        assert response.hits[0].doc_id == "doc1"
        assert response.total_found == 1
        assert response.execution_time_ms == 100.5
        assert response.provenance_hash == "hash123"
