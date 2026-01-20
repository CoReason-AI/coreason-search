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

from pydantic import BaseModel, Field


class RetrieverType(str, Enum):
    """Enumeration of available retriever types."""

    LANCE_DENSE = "lance_dense"
    LANCE_FTS = "lance_fts"
    GRAPH_NEIGHBOR = "graph_neighbor"


class SearchRequest(BaseModel):
    """Request model for search operations.

    Attributes:
        query: The search query. Can be a string for RAG or a Dict for Boolean/Structured search.
        strategies: List of retrieval strategies to execute.
        fusion_enabled: Whether to enable Reciprocal Rank Fusion. Defaults to True.
        rerank_enabled: Whether to enable Re-ranking. Defaults to True.
        distill_enabled: Whether to enable The Scout (context distillation). Defaults to True.
        top_k: Number of results to return. Defaults to 5.
        filters: Optional metadata filters (e.g., {"year": {"$gt": 2024}}).
    """

    query: Union[str, Dict[str, str]] = Field(..., description="String for RAG, Dict for Boolean")
    strategies: List[RetrieverType] = Field(..., min_length=1, description="List of retrieval strategies to execute")
    fusion_enabled: bool = True
    rerank_enabled: bool = True
    distill_enabled: bool = Field(default=True, description="Enable The Scout context distillation")
    top_k: int = Field(default=5, gt=0, description="Number of results to return")
    filters: Optional[Dict[str, Any]] = Field(default=None, description="Metadata filters")


class Hit(BaseModel):
    """Model representing a single search result (hit).

    Attributes:
        doc_id: Unique document identifier.
        content: The main content of the hit.
        original_text: The full original text of the document.
        distilled_text: The text after processing by The Scout.
        score: The relevance score of the hit.
        source_strategy: The strategy that found this hit.
        metadata: Associated metadata.
    """

    doc_id: str
    content: str
    original_text: str = Field(..., description="Full text")
    distilled_text: str = Field(..., description="Post-Scout text")
    score: float
    source_strategy: str  # "dense" or "sparse"
    metadata: Dict[str, Any]


class SearchResponse(BaseModel):
    """Response model for search operations.

    Attributes:
        hits: The list of search hits.
        total_found: Total number of documents found (may be approximate or pre-truncation).
        execution_time_ms: Total execution time in milliseconds.
        provenance_hash: Hash for audit and reproducibility.
    """

    hits: List[Hit]
    total_found: int
    execution_time_ms: float
    provenance_hash: str  # For audit
