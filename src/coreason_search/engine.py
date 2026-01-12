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
from coreason_search.retrievers.sparse import SparseRetriever
from coreason_search.schemas import Hit, RetrieverType, SearchRequest, SearchResponse
from coreason_search.scout import get_scout
from coreason_search.utils.logger import logger


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
        self.fusion_engine = FusionEngine()
        self.reranker = get_reranker()
        self.scout = get_scout()

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
                    # Not implemented in this atomic unit, log warning or skip
                    logger.warning("Graph retrieval not implemented yet.")
                    # all_hits.append([])
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
        """
        logger.info(f"Executing SYSTEMATIC search: {request.query}")

        # Enforce constraints
        # "Disables Re-ranking & Scouting"
        # "Enforces Exact Boolean Logic" -> SparseRetriever

        # We only support LANCE_FTS or maybe DENSE if requested (Grey Lit).
        # But for the generator, we iterate the retrievers.

        # Systematic usually implies ONE exhaustive search.
        # If multiple strategies, we chain generators?

        generators = []
        for strategy in request.strategies:
            if strategy == RetrieverType.LANCE_FTS:
                generators.append(self.sparse_retriever.retrieve_systematic(request))
            elif strategy == RetrieverType.LANCE_DENSE:
                # Dense usually isn't systematic generator, but if requested...
                # PRD: "Disables Vector Search (unless explicitly requested...)"
                # DenseRetriever `retrieve` returns List, not generator.
                # If we need dense generator, we need to implement `retrieve_systematic` in Dense too.
                # For now, assuming Sparse is the main one.
                # Or just yield from the list.
                hits = self.dense_retriever.retrieve(request)  # This is top_k limited!
                # Systematic needs ALL.
                # Dense search for ALL is expensive/weird (nearest neighbors of everything?).
                # Usually systematic implies boolean.
                # We'll skip dense generator for now unless required.
                logger.warning("Dense strategy used in systematic mode - only top_k results will be yielded.")
                yield from hits

        for gen in generators:
            yield from gen
