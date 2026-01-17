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
from typing import Any, Dict, Iterator, List, Union

from coreason_search.db import get_db_manager
from coreason_search.interfaces import BaseRetriever
from coreason_search.schemas import Hit, RetrieverType, SearchRequest
from coreason_search.utils.filters import matches_filters
from coreason_search.utils.query_parser import parse_pubmed_query


class SparseRetriever(BaseRetriever):
    """
    Sparse/Boolean Retriever strategy using LanceDB FTS (Tantivy).
    Supports Systematic Review mode with generators.
    """

    def __init__(self) -> None:
        self.db_manager = get_db_manager()
        self.table = self.db_manager.get_table()
        self.systematic_batch_size = 1000

    def retrieve(self, request: SearchRequest) -> List[Hit]:
        """
        Execute sparse/boolean retrieval.

        Args:
            request: The search request.

        Returns:
            List[Hit]: Top k hits.
        """
        query_str = self._prepare_query(request.query)

        # LanceDB FTS
        # Note: 'fts' query type requires an FTS index.
        # If we are in 'Systematic' mode, we might want ALL results.
        # But this method returns List[Hit] (top_k).
        # The PRD says Systematic Mode "Returns a generator".
        # This standard retrieve method is for the Engine to call for standard searches.
        # If the Engine wants systematic, it should call `retrieve_systematic`?
        # The `BaseRetriever` defines `retrieve` -> `List[Hit]`.
        # So we implement standard top_k here.

        try:
            # Apply oversampling for post-filtering
            limit = request.top_k
            if request.filters:
                limit = max(limit * 10, 100)

            results_list = self.table.search(query_str, query_type="fts").limit(limit).to_list()
        except Exception:
            # Fallback if FTS index missing or other error?
            # For now, let's propagate or return empty to adhere to fail-fast/explicit error?
            # Or log and return empty.
            # But "Atomic unit must be robust".
            # If FTS index is missing, LanceDB raises ValueError usually.
            # We should probably let it raise so the user knows configuration is wrong.
            raise

        hits = self._map_results(results_list)
        if request.filters:
            hits = [h for h in hits if matches_filters(h.metadata, request.filters)]

        return hits[: request.top_k]

    def get_table_version(self) -> int:
        """
        Get the current version of the LanceDB table.
        Used for audit snapshots.
        """
        return int(self.table.version)

    def retrieve_systematic(self, request: SearchRequest) -> Iterator[Hit]:
        """
        Systematic Search Mode.
        Returns a generator yielding ALL results matching the boolean query.
        Uses offset-based pagination loop for true streaming without loading everything into RAM.
        """
        query_str = self._prepare_query(request.query)

        offset = 0
        batch_size = self.systematic_batch_size

        while True:
            # Execute query with limit/offset
            # Note: We must re-build the query builder each time because offset is stateful
            # or builder might be consumed. Safest to rebuild.
            batch_results = self.table.search(query_str, query_type="fts").limit(batch_size).offset(offset).to_list()

            if not batch_results:
                break

            for item in batch_results:
                hit = self._map_single_result(item)
                if request.filters:
                    if matches_filters(hit.metadata, request.filters):
                        yield hit
                else:
                    yield hit

            if len(batch_results) < batch_size:
                # Less than requested means end of results
                break

            offset += batch_size

    def _prepare_query(self, query: Union[str, Dict[str, Any]]) -> str:
        """Helper to prepare query string."""
        if isinstance(query, dict):
            # If it's a dict, we might need to convert to Tantivy syntax.
            # e.g. {"Title": "Aspirin"} -> "Title:Aspirin"
            # Basic implementation: AND them together
            parts = []
            for k, v in query.items():
                # naive escaping might be needed
                parts.append(f"{k}:{v}")
            return " AND ".join(parts)
        return parse_pubmed_query(str(query))

    def _map_results(self, results_list: List[Dict[str, Any]]) -> List[Hit]:
        """Map generic list of dicts to Hits."""
        return [self._map_single_result(item) for item in results_list]

    def _map_single_result(self, item: Dict[str, Any]) -> Hit:
        """Map a single dict to a Hit."""
        doc_id = item["doc_id"]
        content = item["content"]
        metadata_str = item["metadata"]

        # _score is returned by Tantivy
        score = item.get("_score", 0.0)

        try:
            metadata = json.loads(metadata_str) if metadata_str else {}
        except json.JSONDecodeError:  # pragma: no cover
            metadata = {}

        return Hit(
            doc_id=doc_id,
            content=content,
            original_text=content,
            distilled_text="",
            score=score,
            source_strategy=RetrieverType.LANCE_FTS.value,
            metadata=metadata,
        )
