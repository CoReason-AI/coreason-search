# Copyright (c) 2025 CoReason, Inc.
#
# This software is proprietary and dual-licensed.
# Licensed under the Prosperity Public License 3.0 (the "License").
# A copy of the license is available at https://prosperitylicense.com/versions/3.0.0
# For details, see the LICENSE file.
# Commercial use beyond a 30-day trial requires a separate license.
#
# Source Code: https://github.com/CoReason-AI/coreason_search

import hashlib
import time
from typing import Iterator, List

from coreason_search.fusion import FusionEngine
from coreason_search.reranker import get_reranker
from coreason_search.retrievers.dense import DenseRetriever
from coreason_search.retrievers.graph import GraphRetriever
from coreason_search.retrievers.sparse import SparseRetriever
from coreason_search.schemas import Hit, RetrieverType, SearchRequest, SearchResponse
from coreason_search.scout import get_scout
from coreason_search.utils.logger import logger
from coreason_search.veritas import get_veritas_client


class SearchEngine:
    """
    Unified Retrieval Execution Engine.
    Orchestrates Embedder, Retrievers, Fusion, Reranker, and Scout.
    """

    def __init__(self) -> None:
        # Initialize components
        # In a real app, these might be lazy loaded or injected
        self.dense_retriever = DenseRetriever()
        self.sparse_retriever = SparseRetriever()
        self.graph_retriever = GraphRetriever()
        self.fusion_engine = FusionEngine()
        self.reranker = get_reranker()
        self.scout = get_scout()
        self.veritas = get_veritas_client()

    def execute(self, request: SearchRequest) -> SearchResponse:
        """
        Execute a standard search request (RAG, Ad-hoc).
        Returns a SearchResponse with top-k hits.
        """
        start_time = time.time()
        logger.info(f"Executing search: {request.strategies} query={request.query}")

        all_hits: List[List[Hit]] = []

        # 1. Retrieval
        for strategy in request.strategies:
            try:
                if strategy == RetrieverType.LANCE_DENSE:
                    hits = self.dense_retriever.retrieve(request)
                    all_hits.append(hits)
                elif strategy == RetrieverType.LANCE_FTS:
                    hits = self.sparse_retriever.retrieve(request)
                    all_hits.append(hits)
                elif strategy == RetrieverType.GRAPH_NEIGHBOR:
                    hits = self.graph_retriever.retrieve(request)
                    all_hits.append(hits)
                else:
                    logger.warning(f"Unknown strategy: {strategy}")  # pragma: no cover
            except Exception as e:
                logger.error(f"Error in strategy {strategy}: {e}")
                # Continue with other strategies? Or fail?
                # "Robustness". Let's log and continue if others succeed.
                # But if all fail, we return empty.
                pass

        # 2. Fusion
        if request.fusion_enabled and len(all_hits) > 0:
            fused_hits = self.fusion_engine.fuse(all_hits)
        else:
            # Flatten if single or just take first non-empty?
            # If fusion disabled, but multiple strategies?
            # PRD implies RRF is used when "mixing strategies".
            # If fusion disabled, what do we return?
            # Concatenation? Or just the first one?
            # Let's flatten and dedup simple.
            # But usually if fusion is disabled, it implies single strategy?
            # Let's just use fusion engine with k=0? No.
            # We'll just flatten.
            flat_hits = [h for sublist in all_hits for h in sublist]
            # Simple dedup by doc_id
            seen = set()
            fused_hits = []
            for h in flat_hits:
                if h.doc_id not in seen:
                    seen.add(h.doc_id)
                    fused_hits.append(h)

        # 3. Re-Ranking
        # Limit to reasonable number for reranking if strictly required?
        # But reranker takes top_k.
        # Reranker needs to see candidates.
        # Pass all fused hits to reranker? Or top 50?
        # PRD: "Input: Top 50 results from Fusion."
        rerank_candidates = fused_hits[:50]  # Hardcoded 50 per PRD

        if request.rerank_enabled and rerank_candidates:
            reranked_hits = self.reranker.rerank(request.query, rerank_candidates, top_k=request.top_k)
        else:
            reranked_hits = fused_hits[: request.top_k]

        # 4. Scout (Distillation)
        if request.distill_enabled and reranked_hits:
            final_hits = self.scout.distill(request.query, reranked_hits)
        else:
            final_hits = reranked_hits

        # 5. Response Construction
        execution_time = (time.time() - start_time) * 1000

        # Provenance hash
        prov_str = f"{request.query}{[h.doc_id for h in final_hits]}"
        provenance_hash = hashlib.sha256(prov_str.encode()).hexdigest()

        return SearchResponse(
            hits=final_hits,
            total_found=len(final_hits),  # Or total from DB? "Total Found" usually means total matches.
            # But retrieval truncates. We don't have total count unless we ask DB for count.
            # For now, total returned.
            execution_time_ms=execution_time,
            provenance_hash=provenance_hash,
        )

    def execute_systematic(self, request: SearchRequest) -> Iterator[Hit]:
        """
        Execute a Systematic Search (Review Mode).
        Returns a generator of Hits.
        Strictly assumes SPARSE/Boolean strategy or similar.
        Disables Rerank/Distill by default/enforced.
        Logs to Veritas for audit.
        """
        # Get snapshot ID for audit
        try:
            snapshot_id = self.sparse_retriever.get_table_version()
        except Exception:
            snapshot_id = -1  # Fallback if DB not ready

        audit_data = {
            "query": request.query,
            "strategies": [s.value for s in request.strategies],
            "snapshot_id": snapshot_id,
        }
        self.veritas.log_audit("SYSTEMATIC_SEARCH_START", audit_data)
        logger.info(f"Executing SYSTEMATIC search: {request.query}")

        count = 0
        try:
            # We only support LANCE_FTS or maybe DENSE if requested (Grey Lit).
            for strategy in request.strategies:
                if strategy == RetrieverType.LANCE_FTS:
                    for hit in self.sparse_retriever.retrieve_systematic(request):
                        yield hit
                        count += 1
                elif strategy == RetrieverType.LANCE_DENSE:
                    # Dense usually isn't systematic generator, but if requested...
                    logger.warning("Dense strategy used in systematic mode - only top_k results will be yielded.")
                    hits = self.dense_retriever.retrieve(request)
                    for hit in hits:
                        yield hit
                        count += 1

        finally:
            # Log completion even if generator is interrupted (if possible, but finally works on generator close)
            complete_data = {
                "total_found": count,
                # Provenance hash would ideally be calculated from all IDs, but that requires buffering.
                # We can log the count.
            }
            self.veritas.log_audit("SYSTEMATIC_SEARCH_COMPLETE", complete_data)
