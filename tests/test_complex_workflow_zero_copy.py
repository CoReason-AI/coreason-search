# Copyright (c) 2025 CoReason, Inc.
#
# This software is proprietary and dual-licensed.
# Licensed under the Prosperity Public License 3.0 (the "License").
# A copy of the license is available at https://prosperitylicense.com/versions/3.0.0
# For details, see the LICENSE file.
# Commercial use beyond a 30-day trial requires a separate license.
#
# Source Code: https://github.com/CoReason-AI/coreason_search

from typing import Any, Dict, Generator, Optional
from unittest.mock import MagicMock

import pytest

from coreason_search.config import Settings
from coreason_search.db import get_db_manager, reset_db_manager
from coreason_search.embedder import reset_embedder
from coreason_search.engine import SearchEngineAsync
from coreason_search.schemas import Hit, RetrieverType, SearchRequest
from coreason_search.scout import MockScout


class TestComplexWorkflowZeroCopy:
    """Complex workflow tests for Zero-Copy / JIT fetching via SearchEngine."""

    @pytest.fixture(autouse=True)
    def setup_teardown(self, tmp_path: str) -> Generator[None, None, None]:
        self.db_path = str(tmp_path) + "/lancedb_complex"
        reset_db_manager()
        get_db_manager(self.db_path)
        reset_embedder()
        yield
        reset_db_manager()
        reset_embedder()

    def _get_engine(self, fetcher_func: Any) -> SearchEngineAsync:
        """Helper to initialize engine with custom Scout fetcher."""
        config = Settings(database_uri=self.db_path)
        engine = SearchEngineAsync(config)

        # Override the Scout's fetcher or the Scout itself
        # Since get_scout is cached, we replace the scout instance on the engine
        engine.scout = MockScout(config=engine.config.scout, content_fetcher=fetcher_func)
        return engine

    @pytest.mark.asyncio
    async def test_authenticated_search_flow(self) -> None:
        """
        Test a full search flow where hits are pointers and access is controlled by user_context.

        Scenario:
        1. Retriever returns a Pointer Hit for "secret_doc".
        2. Fetcher checks for `token="valid"`.
        3. If valid, returns text. If invalid, returns None.
        4. Distiller (Scout) processes text if fetched.
        """

        # 1. Setup DB with Pointer Hits (mocked via Retriever override)
        # We can mock the retriever to return a hit without text but with pointer.

        def secure_fetcher(ptr: Dict[str, str], ctx: Optional[Dict[str, Any]]) -> Optional[str]:
            if ctx and ctx.get("token") == "valid":
                if ptr.get("id") == "secret":
                    return "This is top secret content."
            return None

        engine = self._get_engine(secure_fetcher)

        # Mock Dense Retriever to return our pointer hit
        pointer_hit = Hit(
            doc_id="secret_doc",
            content=None,
            original_text=None,
            distilled_text="",
            score=0.9,
            source_strategy="dense",
            metadata={},
            source_pointer={"id": "secret"},
        )
        engine.dense_retriever.retrieve = MagicMock(return_value=[pointer_hit])  # type: ignore

        # 2. Execute with VALID Token
        req_valid = SearchRequest(
            query="secret",
            strategies=[RetrieverType.LANCE_DENSE],
            user_context={"token": "valid"},
            distill_enabled=True,
        )

        async with engine:
            res_valid = await engine.execute(req_valid)

            assert len(res_valid.hits) == 1
            hit_valid = res_valid.hits[0]
            # Should have distilled text
            assert "secret content" in hit_valid.distilled_text
            # Should NOT have original text (Ephemeral)
            assert hit_valid.original_text is None
            assert hit_valid.content is None

        # 3. Execute with INVALID Token
        req_invalid = SearchRequest(
            query="secret",
            strategies=[RetrieverType.LANCE_DENSE],
            user_context={"token": "hacker"},
            distill_enabled=True,
        )

        async with engine:
            res_invalid = await engine.execute(req_invalid)

            assert len(res_invalid.hits) == 1
            hit_invalid = res_invalid.hits[0]
            # Fetcher returns None -> Distilled text empty
            assert hit_invalid.distilled_text == ""
            assert hit_invalid.original_text is None

    @pytest.mark.asyncio
    async def test_zero_copy_redundant_flow(self) -> None:
        """
        Redundant test to ensure multiple strategies + fusion + JIT works together.
        """

        def simple_fetcher(ptr: Dict[str, str], ctx: Optional[Dict[str, Any]]) -> str:
            return f"Fetched content for {ptr.get('id')}"

        engine = self._get_engine(simple_fetcher)

        # Mock Dense (Pointer 1)
        h1 = Hit(
            doc_id="d1",
            content=None,
            original_text=None,
            distilled_text="",
            score=0.9,
            source_strategy="dense",
            metadata={},
            source_pointer={"id": "d1"},
        )
        engine.dense_retriever.retrieve = MagicMock(return_value=[h1])  # type: ignore

        # Mock Sparse (Pointer 2)
        h2 = Hit(
            doc_id="d2",
            content=None,
            original_text=None,
            distilled_text="",
            score=0.8,
            source_strategy="sparse",
            metadata={},
            source_pointer={"id": "d2"},
        )
        engine.sparse_retriever.retrieve = MagicMock(return_value=[h2])  # type: ignore

        req = SearchRequest(
            query="content",
            strategies=[RetrieverType.LANCE_DENSE, RetrieverType.LANCE_FTS],
            fusion_enabled=True,
            distill_enabled=True,
        )

        async with engine:
            res = await engine.execute(req)

            assert len(res.hits) == 2
            # Check both are processed
            texts = [h.distilled_text for h in res.hits]
            assert any("Fetched content for d1" in t for t in texts)
            assert any("Fetched content for d2" in t for t in texts)

            # Verify Ephemeral
            assert all(h.original_text is None for h in res.hits)
