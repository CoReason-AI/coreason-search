# Copyright (c) 2025 CoReason, Inc.
#
# This software is proprietary and dual-licensed.
# Licensed under the Prosperity Public License 3.0 (the "License").
# A copy of the license is available at https://prosperitylicense.com/versions/3.0.0
# For details, see the LICENSE file.
# Commercial use beyond a 30-day trial requires a separate license.
#
# Source Code: https://github.com/CoReason-AI/coreason_search

from enum import Enum
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel


class RetrieverType(str, Enum):
    """Enumeration of supported retriever strategies."""

    LANCE_DENSE = "lance_dense"
    LANCE_FTS = "lance_fts"
    GRAPH_NEIGHBOR = "graph_neighbor"


class EmbeddingConfig(BaseModel):
    """Configuration for the embedding model."""

    model_name: str = "Alibaba-NLP/gte-Qwen2-7B-instruct"
    context_length: int = 32768
    batch_size: int = 1


class SearchRequest(BaseModel):
    """Request object for executing a search operation."""

    query: Union[str, Dict[str, str]]
    strategies: List[RetrieverType]
    fusion_enabled: bool = True
    rerank_enabled: bool = True
    distill_enabled: bool = True  # Enable The Scout?
    top_k: int = 5
    filters: Optional[Dict[str, Any]] = None  # e.g., {"year": {"$gt": 2024}}


class Hit(BaseModel):
    """Represents a single search result document."""

    doc_id: str
    content: str
    original_text: str  # Full text
    distilled_text: str  # Post-Scout text
    score: float
    source_strategy: str  # "dense" or "sparse"
    metadata: Dict[str, Any]


class SearchResponse(BaseModel):
    """Response object containing search results and metadata."""

    hits: List[Hit]
    total_found: int
    execution_time_ms: float
    provenance_hash: str  # For audit
