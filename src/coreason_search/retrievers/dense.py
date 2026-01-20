# Copyright (c) 2025 CoReason, Inc.
#
# This software is proprietary and dual-licensed.
# Licensed under the Prosperity Public License 3.0 (the "License").
# A copy of the license is available at https://prosperitylicense.com/versions/3.0.0
# For details, see the LICENSE file.
# Commercial use beyond a 30-day trial requires a separate license.
#
# Source Code: https://github.com/CoReason-AI/coreason_search

from typing import List

from coreason_search.db import get_db_manager
from coreason_search.embedder import get_embedder
from coreason_search.interfaces import BaseRetriever
from coreason_search.schemas import Hit, RetrieverType, SearchRequest
from coreason_search.utils.common import extract_query_text
from coreason_search.utils.filters import matches_filters
from coreason_search.utils.mapper import LanceMapper


class DenseRetriever(BaseRetriever):
    """Dense Vector Retriever strategy using LanceDB and Qwen3/Mock Embeddings."""

    def __init__(self) -> None:
        """Initialize the Dense Retriever."""
        self.db_manager = get_db_manager()
        self.embedder = get_embedder()
        self.table = self.db_manager.get_table()

    def retrieve(self, request: SearchRequest) -> List[Hit]:
        """Execute dense vector retrieval.

        Args:
            request: The search request containing the query and parameters.

        Returns:
            List[Hit]: List of hits matching the query.
        """
        query_text = extract_query_text(request.query)

        # Embed the query
        # Embedder expects Union[str, List[str]]. We have str.
        query_vector = self.embedder.embed(query_text)[0]

        # Execute Search
        # LanceDB search returns a LanceQueryBuilder
        # We need to handle limit/top_k from request

        # Apply oversampling for post-filtering
        limit = request.top_k
        if request.filters:
            # Heuristic: fetch 10x or at least 100 more to allow for filtering
            limit = max(limit * 10, 100)

        # We use `to_list()` to get `_distance` which is not available in `to_pydantic`
        # unless mapped explicitly.
        results_list = self.table.search(query_vector).limit(limit).to_list()

        hits = []
        for item in results_list:
            # _distance is returned by LanceDB for vector search
            distance = item.get("_distance", 0.0)
            score = 1.0 - distance

            # Map to Hit using helper
            hit = LanceMapper.map_hit(item, RetrieverType.LANCE_DENSE.value, score)

            # Apply Metadata Filters (Python-side)
            if request.filters and not matches_filters(hit.metadata, request.filters):
                continue

            hits.append(hit)

        # Return only top_k after filtering
        return hits[: request.top_k]
