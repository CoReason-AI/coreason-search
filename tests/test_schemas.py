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


def test_retriever_type_enum() -> None:
    """Test RetrieverType enum values."""
    # Compare value to string
    assert RetrieverType.LANCE_DENSE.value == "lance_dense"
    assert RetrieverType.LANCE_FTS.value == "lance_fts"
    assert RetrieverType.GRAPH_NEIGHBOR.value == "graph_neighbor"


def test_embedding_config_defaults() -> None:
    """Test EmbeddingConfig defaults."""
    config = EmbeddingConfig()
    assert config.model_name == "Alibaba-NLP/gte-Qwen2-7B-instruct"
    assert config.context_length == 32768
    assert config.batch_size == 1


def test_embedding_config_validation() -> None:
    """Test EmbeddingConfig validation."""
    with pytest.raises(ValidationError):
        EmbeddingConfig(context_length=0)
    with pytest.raises(ValidationError):
        EmbeddingConfig(batch_size=0)


def test_search_request_defaults() -> None:
    """Test default values for SearchRequest."""
    req = SearchRequest(query="test query", strategies=[RetrieverType.LANCE_DENSE])
    assert req.fusion_enabled is True
    assert req.rerank_enabled is True
    assert req.distill_enabled is True
    assert req.top_k == 5
    assert req.filters is None


def test_search_request_validation() -> None:
    """Test validation constraints."""
    # Test min_length for strategies
    with pytest.raises(ValidationError):
        SearchRequest(query="test", strategies=[])

    # Test top_k gt 0
    with pytest.raises(ValidationError):
        SearchRequest(query="test", strategies=[RetrieverType.LANCE_DENSE], top_k=0)


def test_search_request_complex_types() -> None:
    """Test SearchRequest with complex/edge case types."""
    # Boolean query (Dict)
    # The schema specifies Dict[str, str], so values must be strings.
    # Recursive boolean logic (lists/nested dicts) is not supported by this specific schema field yet.
    boolean_query = {"title": "Aspirin", "abstract": "Headache"}
    req_bool = SearchRequest(query=boolean_query, strategies=[RetrieverType.LANCE_FTS])
    assert req_bool.query == boolean_query

    # Complex Filters
    complex_filters = {
        "year": {"$gt": 2020},
        "tags": ["medical", "urgent"],
        "is_active": True,
        "nested": {"key": "value"},
    }
    req_filters = SearchRequest(query="test", strategies=[RetrieverType.LANCE_DENSE], filters=complex_filters)
    assert req_filters.filters == complex_filters


def test_hit_schema() -> None:
    """Test Hit schema structure."""
    hit = Hit(
        doc_id="123",
        content="some content",
        original_text="full text original",
        distilled_text="distilled",
        score=0.95,
        source_strategy="dense",
        metadata={"year": 2024},
    )
    assert hit.doc_id == "123"
    assert hit.original_text == "full text original"
    assert hit.distilled_text == "distilled"
    assert hit.metadata["year"] == 2024


def test_hit_edge_cases() -> None:
    """Test Hit schema with edge cases."""
    # Unicode and Emoji
    unicode_text = "こんにちは 🌍"
    hit = Hit(
        doc_id="uni-1",
        content=unicode_text,
        original_text=unicode_text,
        distilled_text=unicode_text,
        score=0.0,
        source_strategy="test",
        metadata={"author": "José"},
    )
    assert hit.original_text == unicode_text

    # Large metadata
    large_meta = {f"k{i}": i for i in range(100)}
    hit_large = Hit(
        doc_id="large-1",
        content="c",
        original_text="o",
        distilled_text="d",
        score=1.0,
        source_strategy="s",
        metadata=large_meta,
    )
    assert len(hit_large.metadata) == 100


def test_search_response_schema() -> None:
    """Test SearchResponse structure."""
    hit = Hit(
        doc_id="1", content="c", original_text="o", distilled_text="d", score=1.0, source_strategy="s", metadata={}
    )
    res = SearchResponse(hits=[hit], total_found=10, execution_time_ms=100.5, provenance_hash="abc")
    assert len(res.hits) == 1
    assert res.total_found == 10
