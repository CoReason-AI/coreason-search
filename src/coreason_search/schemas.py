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
    LANCE_DENSE = "lance_dense"
    LANCE_FTS = "lance_fts"
    GRAPH_NEIGHBOR = "graph_neighbor"


class EmbeddingConfig(BaseModel):
    model_name: str = "Alibaba-NLP/gte-Qwen2-7B-instruct"
    context_length: int = Field(default=32768, gt=0)
    batch_size: int = Field(default=1, gt=0)


class SearchRequest(BaseModel):
    query: Union[str, Dict[str, str]] = Field(..., description="String for RAG, Dict for Boolean")
    strategies: List[RetrieverType] = Field(..., min_length=1, description="List of retrieval strategies to execute")
    fusion_enabled: bool = True
    rerank_enabled: bool = True
    top_k: int = Field(default=5, gt=0, description="Number of results to return")
    filters: Optional[Dict[str, Any]] = Field(default=None, description="Metadata filters")


class Hit(BaseModel):
    doc_id: str
    content: str
    score: float
    source_strategy: str  # "dense" or "sparse"
    metadata: Dict[str, Any]


class SearchResponse(BaseModel):
    hits: List[Hit]
    total_found: int
    execution_time_ms: float
    provenance_hash: str  # For audit
