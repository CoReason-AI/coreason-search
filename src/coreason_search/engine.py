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
from typing import Any, AsyncIterator, Iterator, List, Optional, Union

import anyio
import httpx
from anyio import to_thread

from coreason_search.config import Settings, load_config
from coreason_search.db import get_db_manager
from coreason_search.embedder import get_embedder
from coreason_search.fusion import FusionEngine
from coreason_search.reranker import get_reranker
from coreason_search.retrievers.dense import DenseRetriever
from coreason_search.retrievers.graph import GraphRetriever
from coreason_search.retrievers.sparse import SparseRetriever
from coreason_search.schemas import Hit, RetrieverType, SearchRequest, SearchResponse
from coreason_search.scout import get_scout
from coreason_search.utils.logger import logger
from coreason_search.veritas import get_veritas_client


class SearchEngineAsync:
    """Async Unified Retrieval Execution Engine.

    Handles all logic for search operations in an async-first manner.
    Implements the core business logic.

    Attributes:
        config: The application configuration.
        db_manager: Manager for the LanceDB database.
        embedder: The embedding model instance.
        dense_retriever: Retriever for dense vector search.
        sparse_retriever: Retriever for sparse/boolean search.
        graph_retriever: Retriever for graph-based search.
        fusion_engine: Engine for fusing results from multiple retrievers.
        reranker: The re-ranking model instance.
        scout: The context distillation (scout) instance.
        veritas: Client for audit logging.
    """

    def __init__(
        self,
        config: Optional[Union[Settings, str]] = None,
        client: Optional[httpx.AsyncClient] = None,
    ) -> None:
        """Initialize the Search Engine.

        Args:
            config: A Settings object or path to a config file. If None,
                defaults are loaded.
            client: An optional external httpx.AsyncClient to use.
        """
        if isinstance(config, str):
            self.config = load_config(config)
        elif isinstance(config, Settings):
            self.config = config
        else:
            self.config = load_config()

        self._internal_client = client is None
        self._client = client or httpx.AsyncClient()

        # Initialize global Singletons with config
        self.db_manager = get_db_manager(self.config.database_uri)

        # Initialize local components
        self.embedder = get_embedder(self.config.embedding)

        # Initialize Retrievers
        self.dense_retriever = DenseRetriever()
        self.sparse_retriever = SparseRetriever()
        self.graph_retriever = GraphRetriever()
        self.fusion_engine = FusionEngine()

        # Initialize Reranker and Scout with Config
        self.reranker = get_reranker(self.config.reranker)
        self.scout = get_scout(self.config.scout)
        self.veritas = get_veritas_client()

    async def __aenter__(self) -> "SearchEngineAsync":
        return self

    async def __aexit__(
        self, exc_type: Optional[type[BaseException]], exc_val: Optional[BaseException], exc_tb: Optional[Any]
    ) -> None:
        if self._internal_client:
            await self._client.aclose()
        # Close other resources if necessary (LanceDB is sync and auto-closes usually)

    async def execute(self, request: SearchRequest) -> SearchResponse:
        """Execute a standard search request (RAG, Ad-hoc).

        Performs retrieval, fusion, re-ranking, and context distillation.
        Wraps synchronous I/O and CPU-bound tasks in threads.

        Args:
            request: The search request containing query and parameters.

        Returns:
            SearchResponse: The search results including hits and metadata.
        """
        start_time = time.time()
        logger.info(f"Executing search: {request.strategies} query={request.query}")

        all_hits: List[List[Hit]] = []

        # 1. Retrieval
        for strategy in request.strategies:
            try:
                hits: List[Hit] = []
                if strategy == RetrieverType.LANCE_DENSE:
                    hits = await to_thread.run_sync(self.dense_retriever.retrieve, request)
                elif strategy == RetrieverType.LANCE_FTS:
                    hits = await to_thread.run_sync(self.sparse_retriever.retrieve, request)
                elif strategy == RetrieverType.GRAPH_NEIGHBOR:
                    hits = await to_thread.run_sync(self.graph_retriever.retrieve, request)
                else:
                    logger.warning(f"Unknown strategy: {strategy}")  # pragma: no cover
                    continue  # pragma: no cover

                if hits:
                    all_hits.append(hits)
            except Exception as e:
                logger.error(f"Error in strategy {strategy}: {e}")
                pass

        # 2. Fusion
        if request.fusion_enabled and len(all_hits) > 0:
            fused_hits = await to_thread.run_sync(self.fusion_engine.fuse, all_hits)
        else:
            flat_hits = [h for sublist in all_hits for h in sublist]
            # Simple dedup by doc_id
            seen = set()
            fused_hits = []
            for h in flat_hits:
                if h.doc_id not in seen:
                    seen.add(h.doc_id)
                    fused_hits.append(h)

        # 3. Re-Ranking
        rerank_candidates = fused_hits[:50]  # Hardcoded 50 per PRD

        if request.rerank_enabled and rerank_candidates:
            reranked_hits = await to_thread.run_sync(
                self.reranker.rerank, request.query, rerank_candidates, request.top_k
            )
        else:
            reranked_hits = fused_hits[: request.top_k]

        # 4. Scout (Distillation)
        if request.distill_enabled and reranked_hits:
            final_hits = await to_thread.run_sync(self.scout.distill, request.query, reranked_hits)
        else:
            final_hits = reranked_hits

        # 5. Response Construction
        execution_time = (time.time() - start_time) * 1000

        # Provenance hash
        prov_str = f"{request.query}{[h.doc_id for h in final_hits]}"
        provenance_hash = hashlib.sha256(prov_str.encode()).hexdigest()

        return SearchResponse(
            hits=final_hits,
            total_found=len(final_hits),
            execution_time_ms=execution_time,
            provenance_hash=provenance_hash,
        )

    async def execute_systematic(self, request: SearchRequest) -> AsyncIterator[Hit]:
        """Execute a Systematic Search (Review Mode).

        Returns an async generator of Hits.

        Args:
            request: The search request.

        Yields:
            Hit: Search hits one by one.
        """
        # Get snapshot ID for audit
        try:
            snapshot_id = self.sparse_retriever.get_table_version()
        except Exception:
            snapshot_id = -1

        audit_data = {
            "query": request.query,
            "strategies": [s.value for s in request.strategies],
            "snapshot_id": snapshot_id,
        }
        self.veritas.log_audit("SYSTEMATIC_SEARCH_START", audit_data)
        logger.info(f"Executing SYSTEMATIC search: {request.query}")

        count = 0
        try:
            for strategy in request.strategies:
                if strategy == RetrieverType.LANCE_FTS:
                    # Sparse systematic returns a generator.
                    # We cannot simply await a generator function.
                    # We need to iterate it in a thread, or run the whole thing in a thread?
                    # Generators are hard to run in threads because `next()` is called repeatedly.
                    # We will use an adapter to run `next()` in a thread for each item.
                    # This might be slow if overhead is high, but ensures non-blocking.

                    # Alternatively, if we assume the generator does bulk fetches, we can fetch chunks.
                    # But the interface yields hits.

                    # Let's instantiate the sync generator first.
                    sync_gen = self.sparse_retriever.retrieve_systematic(request)

                    while True:
                        try:
                            # Run next(sync_gen) in a thread
                            hit = await to_thread.run_sync(next, sync_gen)
                            yield hit
                            count += 1
                        except StopIteration:
                            break  # pragma: no cover
                        except Exception as e:
                            logger.error(f"Error in systematic search stream: {e}")
                            break

                elif strategy == RetrieverType.LANCE_DENSE:
                    # Dense returns list, not generator usually.
                    logger.warning("Dense strategy used in systematic mode - only top_k results will be yielded.")
                    hits = await to_thread.run_sync(self.dense_retriever.retrieve, request)
                    for hit in hits:
                        yield hit
                        count += 1

        finally:
            complete_data = {
                "total_found": count,
            }
            self.veritas.log_audit("SYSTEMATIC_SEARCH_COMPLETE", complete_data)


class SearchEngine:
    """Synchronous Facade for SearchEngineAsync.

    Wraps the async core to provide a blocking interface.
    """

    def __init__(self, config: Optional[Union[Settings, str]] = None) -> None:
        self._async = SearchEngineAsync(config)

    def __enter__(self) -> "SearchEngine":
        return self

    def __exit__(
        self, exc_type: Optional[type[BaseException]], exc_val: Optional[BaseException], exc_tb: Optional[Any]
    ) -> None:
        # We run the async cleanup in a one-off loop
        anyio.run(self._async.__aexit__, exc_type, exc_val, exc_tb)

    def execute(self, request: SearchRequest) -> SearchResponse:
        """Execute search synchronously."""
        return anyio.run(self._async.execute, request)

    def execute_systematic(self, request: SearchRequest) -> Iterator[Hit]:
        """Execute systematic search synchronously.

        Note: This currently collects all results into memory to satisfy the sync iterator interface
        via anyio.run. For large datasets, use SearchEngineAsync directly.
        """

        async def _collect() -> List[Hit]:
            hits = []
            async for hit in self._async.execute_systematic(request):
                hits.append(hit)
            return hits

        return iter(anyio.run(_collect))
