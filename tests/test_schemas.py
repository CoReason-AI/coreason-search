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
from coreason_search.schemas import (
    EmbeddingConfig,
    Hit,
    RetrieverType,
    SearchRequest,
    SearchResponse,
)
from pydantic import ValidationError


def test_retriever_type_enum() -> None:
    """Test the RetrieverType enum values."""
    assert RetrieverType.LANCE_DENSE.value == "lance_dense"
    assert RetrieverType.LANCE_FTS.value == "lance_fts"
    assert RetrieverType.GRAPH_NEIGHBOR.value == "graph_neighbor"


def test_embedding_config_defaults() -> None:
    """Test EmbeddingConfig default values."""
    config = EmbeddingConfig()
    assert config.model_name == "Alibaba-NLP/gte-Qwen2-7B-instruct"
    assert config.context_length == 32768
    assert config.batch_size == 1


def test_embedding_config_custom() -> None:
    """Test EmbeddingConfig with custom values."""
    config = EmbeddingConfig(model_name="custom-model", context_length=1024, batch_size=32)
    assert config.model_name == "custom-model"
    assert config.context_length == 1024
    assert config.batch_size == 32


def test_search_request_valid_string_query() -> None:
    """Test SearchRequest with a string query."""
    req = SearchRequest(
        query="test query",
        strategies=[RetrieverType.LANCE_DENSE],
    )
    assert req.query == "test query"
    assert req.strategies == [RetrieverType.LANCE_DENSE]
    assert req.fusion_enabled is True
    assert req.rerank_enabled is True
    assert req.distill_enabled is True
    assert req.top_k == 5
    assert req.filters is None


def test_search_request_valid_dict_query() -> None:
    """Test SearchRequest with a dictionary query (boolean)."""
    req = SearchRequest(
        query={"title": "test"},
        strategies=[RetrieverType.LANCE_FTS],
    )
    assert req.query == {"title": "test"}


def test_search_request_validation_error() -> None:
    """Test SearchRequest validation failure."""
    with pytest.raises(ValidationError):
        SearchRequest(
            query=123,  # type: ignore
            strategies=[RetrieverType.LANCE_DENSE],
        )


def test_hit_creation() -> None:
    """Test Hit model creation."""
    hit = Hit(
        doc_id="doc1",
        content="some content",
        original_text="some content",
        distilled_text="distilled",
        score=0.95,
        source_strategy="dense",
        metadata={"year": 2024},
    )
    assert hit.doc_id == "doc1"
    assert hit.metadata["year"] == 2024


def test_search_response_creation() -> None:
    """Test SearchResponse model creation."""
    hit = Hit(
        doc_id="doc1",
        content="c",
        original_text="o",
        distilled_text="d",
        score=0.9,
        source_strategy="dense",
        metadata={},
    )
    resp = SearchResponse(
        hits=[hit],
        total_found=100,
        execution_time_ms=150.5,
        provenance_hash="abc123hash",
    )
    assert len(resp.hits) == 1
    assert resp.total_found == 100
    assert resp.execution_time_ms == 150.5
    assert resp.provenance_hash == "abc123hash"


def test_serialization() -> None:
    """Test JSON serialization."""
    req = SearchRequest(
        query="test",
        strategies=[RetrieverType.LANCE_DENSE],
    )
    json_str = req.model_dump_json()
    assert "lance_dense" in json_str
