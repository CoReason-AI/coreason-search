# Copyright (c) 2025 CoReason, Inc.
#
# This software is proprietary and dual-licensed.
# Licensed under the Prosperity Public License 3.0 (the "License").
# A copy of the license is available at https://prosperitylicense.com/versions/3.0.0
# For details, see the LICENSE file.
# Commercial use beyond a 30-day trial requires a separate license.
#
# Source Code: https://github.com/CoReason-AI/coreason_search

import json
from typing import List

from coreason_search.db import get_db_manager
from coreason_search.embedder import get_embedder
from coreason_search.interfaces import BaseRetriever
from coreason_search.schemas import Hit, RetrieverType, SearchRequest
from coreason_search.utils.filters import matches_filters


class DenseRetriever(BaseRetriever):
    """
    Dense Vector Retriever strategy using LanceDB and Qwen3/Mock Embeddings.
    """

    def __init__(self) -> None:
        self.db_manager = get_db_manager()
        self.embedder = get_embedder()
        self.table = self.db_manager.get_table()

    def retrieve(self, request: SearchRequest) -> List[Hit]:
        """
        Execute dense vector retrieval.

        Args:
            request: The search request containing the query and parameters.

        Returns:
            List[Hit]: List of hits matching the query.
        """
        query_text = request.query
        if isinstance(query_text, dict):
            # If a dict is passed to dense retriever (which expects string for RAG),
            # we might need to stringify it or extract a specific field.
            # However, SearchRequest schema says Union[str, Dict].
            # Dense retriever typically works on semantic meaning of a string.
            # We'll convert to string or handle error?
            # PRD says: "Find papers conceptually related to liver failure" -> String.
            # Strategy B (Boolean) uses Dict.
            # If dense is called with Dict, we should probably fail or stringify.
            # Let's stringify for robustness but log a warning?
            # Or assume the caller handles this.
            # Let's extract values if it's a dict, or just str() it.
            # Actually, if the request has multiple strategies, the engine might pass the same request to all.
            # The query field is shared.
            # If `query` is a Dict, it's for Boolean.
            # But the user might want hybrid.
            # If hybrid, query might need to be a string, and boolean part separate?
            # The PRD says: "SearchRequest... query: Union[str, Dict]".
            # "Strategy B... Query: Dict".
            # "Strategy A... Use Case: 'Find papers...'".
            # This implies if query is a Dict, Dense might not be applicable or we need a way to extract text.
            # Let's assume for now if it's a dict, we try to use a "text" or "query" key, or join values.
            # Or just raise ValueError if Dense is used with structured query?
            # Let's support simple conversion: str(query_text).
            if "text" in query_text:
                query_text = str(query_text["text"])
            else:
                # Fallback: join values
                query_text = " ".join(str(v) for v in query_text.values())

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
            # item is a dict
            doc_id = item["doc_id"]
            content = item["content"]
            metadata_str = item["metadata"]

            try:
                metadata = json.loads(metadata_str) if metadata_str else {}
            except json.JSONDecodeError:  # pragma: no cover
                metadata = {}

            # Apply Metadata Filters (Python-side)
            if request.filters and not matches_filters(metadata, request.filters):
                continue

            # _distance is returned by LanceDB for vector search
            distance = item.get("_distance", 0.0)
            # Convert distance to similarity score (assuming cosine distance 0..2?)
            # Usually sim = 1 - distance/2 or similar.
            score = 1.0 - distance

            hits.append(
                Hit(
                    doc_id=doc_id,
                    content=content,
                    original_text=content,  # Assuming content is full text? PRD: "original_text (raw content)"
                    distilled_text="",  # Populated by Scout later
                    score=score,
                    source_strategy=RetrieverType.LANCE_DENSE.value,
                    metadata=metadata,
                )
            )

        # Return only top_k after filtering
        return hits[: request.top_k]
