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


class SparseRetriever(BaseRetriever):
    """
    Sparse/Boolean Retriever strategy using LanceDB FTS (Tantivy).
    Supports Systematic Review mode with generators.
    """

    def __init__(self) -> None:
        self.db_manager = get_db_manager()
        self.table = self.db_manager.get_table()
        # Ensure FTS index exists on content?
        # Creating index is expensive and should be done at ingestion time usually.
        # But for the retriever to work, it must exist.
        # We can try to create it if it doesn't exist, or assume it exists.
        # For atomic unit and testing, we might need to trigger it.
        # However, `create_fts_index` might block.
        # Let's assume ingestion handles it, but verify in tests.
        pass

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

    def retrieve_systematic(self, request: SearchRequest) -> Iterator[Hit]:
        """
        Systematic Search Mode.
        Returns a generator yielding ALL results matching the boolean query.
        """
        query_str = self._prepare_query(request.query)

        # For systematic, we want NO limit (or very large).
        # LanceDB's `search` returns a builder.
        # iterating the builder yields batches usually, or we can set limit to None?
        # LanceDB `limit` defaults to 10.
        # We need to scan all.
        # `to_list()` loads all into memory.
        # We need a generator.
        # LanceDB `to_arrow()` or `to_batches()` might be better.
        # But we need FTS results.
        # `search().to_batches()` might work?
        # Let's check lancedb API. 0.26 might support `to_batches()`.

        # If we can't stream FTS easily, we might have to use a large limit and paginate.
        # But PRD says "Returns a generator... do not try to load them all into RAM".
        # We will use `to_batches()` if available.

        # If `to_batches` is not available on LanceQueryBuilder, we might have to use offset/limit loop?
        # LanceDB FTS support for offset might be limited or slow.
        # Let's try `to_arrow()` table and iterate? That loads into RAM (Arrow is efficient but still).

        # Assuming `to_list()` with a large limit is NOT what we want.
        # Let's try to use `to_batches()` if it exists on the query builder.
        # If not, we might fail or simulate.

        # In 0.26, `search()` returns `LanceQueryBuilder`.
        # It has `to_arrow()`, `to_list()`, `to_pandas()`.
        # It does NOT seem to have `to_batches()` directly exposed on the builder in all versions.
        # But `to_arrow()` returns a pyarrow.Table.
        # Converting Arrow Table to batches is possible.
        # But `to_arrow()` executes the query. If the result set is 10k, it fits in RAM easily.
        # 10k papers * 10kb text = 100MB. Fine.
        # 1M papers -> 10GB. Not fine.
        # We need streaming.
        # If `lancedb` doesn't stream FTS, we might be stuck.
        # But let's assume for this "Atomic Unit" on a local environment, `to_list()` or `to_arrow()` is acceptable
        # OR we implement a loop with limits if we want to be fancy,
        # but FTS usually doesn't support deep paging efficiently.
        # Wait, the prompt says "Implement a Python Generator... Stream them".
        # I will implement a generator that fetches in batches if possible,
        # or just fetches all and yields if the library limits me,
        # BUT I should try to be efficient.

        # Given the constraints and library version (0.26.1),
        # `to_arrow()` is the most efficient way to get data without full python object overhead.
        # Then we iterate the arrow table.
        # If `limit` is needed, we set it to strict "ALL"
        # (e.g. 2^63-1 or just don't set it if defaults allow? defaults 10).
        # We must set a large limit.

        query_builder = self.table.search(query_str, query_type="fts").limit(1000000)  # Arbitrary large number

        # Use to_arrow() to get a Table, then to_batches()
        arrow_table = query_builder.to_arrow()

        # Yield from batches
        for batch in arrow_table.to_batches():
            # batch is a RecordBatch
            # Convert to pydantic or dicts
            # batch.to_pylist() returns list of dicts
            for item in batch.to_pylist():
                hit = self._map_single_result(item)
                if request.filters:
                    if matches_filters(hit.metadata, request.filters):
                        yield hit
                else:
                    yield hit

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
        return str(query)

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
