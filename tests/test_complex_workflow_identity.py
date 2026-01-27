# Copyright (c) 2025 CoReason, Inc.
#
# This software is proprietary and dual-licensed.
# Licensed under the Prosperity Public License 3.0 (the "License").
# A copy of the license is available at https://prosperitylicense.com/versions/3.0.0
# For details, see the LICENSE file.
# Commercial use beyond a 30-day trial requires a separate license.
#
# Source Code: https://github.com/CoReason-AI/coreason_search

from typing import Callable, Dict, Generator, Optional
from unittest.mock import MagicMock

import pytest
from coreason_identity.models import UserContext

from coreason_search.config import Settings
from coreason_search.db import get_db_manager, reset_db_manager
from coreason_search.embedder import reset_embedder
from coreason_search.engine import SearchEngineAsync
from coreason_search.schemas import Hit, RetrieverType, SearchRequest
from coreason_search.scout import MockScout


class TestIdentityAwareRetrieval:
    """Tests for Identity-Aware Retrieval Workflows (Multi-Tenancy)."""

    @pytest.fixture(autouse=True)
    def setup_teardown(self, tmp_path: str) -> Generator[None, None, None]:
        self.db_path = str(tmp_path) + "/lancedb_identity"
        reset_db_manager()
        get_db_manager(self.db_path)
        reset_embedder()
        yield
        reset_db_manager()
        reset_embedder()

    def _get_engine(
        self, fetcher_func: Callable[[Dict[str, str], Optional[UserContext]], Optional[str]]
    ) -> SearchEngineAsync:
        """Helper to initialize engine with custom Scout fetcher."""
        config = Settings(database_uri=self.db_path)
        engine = SearchEngineAsync(config)
        # Override Scout
        engine.scout = MockScout(config=engine.config.scout, content_fetcher=fetcher_func)
        return engine

    @pytest.mark.asyncio
    async def test_multi_tenant_jit_fetching(self) -> None:
        """
        Test Multi-Tenancy via JIT Fetching.
        The same document pointer yields different content based on UserContext.project_context.
        """

        # 1. Define Multi-Tenant Fetcher
        def tenant_fetcher(ptr: Dict[str, str], ctx: Optional[UserContext]) -> Optional[str]:
            doc_id = ptr.get("id")
            if not ctx:
                return "Public Abstract"

            if ctx.project_context == "Project_A":
                return f"Project A Content for {doc_id}"
            elif ctx.project_context == "Project_B":
                return f"Project B Content for {doc_id}"

            return "Public Abstract"

        engine = self._get_engine(tenant_fetcher)

        # 2. Mock Retriever to return a shared pointer hit
        pointer_hit = Hit(
            doc_id="shared_doc",
            content=None,
            original_text=None,
            distilled_text="",
            score=0.9,
            source_strategy="dense",
            metadata={},
            source_pointer={"id": "doc_123"},
        )
        engine.dense_retriever.retrieve = MagicMock(return_value=[pointer_hit])  # type: ignore

        # 3. Test Project A Context
        ctx_a = UserContext(sub="user_a", email="a@co.ai", project_context="Project_A")
        # Query matches "Content" which is in the returned text
        req_a = SearchRequest(query="Content", strategies=[RetrieverType.LANCE_DENSE], user_context=ctx_a)

        async with engine:
            res_a = await engine.execute(req_a)
            assert "Project A Content" in res_a.hits[0].distilled_text
            assert "Project B Content" not in res_a.hits[0].distilled_text

        # 4. Test Project B Context
        ctx_b = UserContext(sub="user_b", email="b@co.ai", project_context="Project_B")
        req_b = SearchRequest(query="Content", strategies=[RetrieverType.LANCE_DENSE], user_context=ctx_b)

        async with engine:
            res_b = await engine.execute(req_b)
            assert "Project B Content" in res_b.hits[0].distilled_text
            assert "Project A Content" not in res_b.hits[0].distilled_text

    @pytest.mark.asyncio
    async def test_unauthorized_jit_fetching(self) -> None:
        """
        Test that unauthorized users see restricted content masked or empty.
        """

        def auth_fetcher(ptr: Dict[str, str], ctx: Optional[UserContext]) -> Optional[str]:
            if ctx and "CLASSIFIED" in ctx.permissions:
                return "Classified Data"
            return None  # Or "Redacted"

        engine = self._get_engine(auth_fetcher)

        pointer_hit = Hit(
            doc_id="classified_doc",
            content=None,
            original_text=None,
            distilled_text="",
            score=0.9,
            source_strategy="dense",
            metadata={},
            source_pointer={"id": "c1"},
        )
        engine.dense_retriever.retrieve = MagicMock(return_value=[pointer_hit])  # type: ignore

        # Case 1: Authorized
        ctx_auth = UserContext(sub="spy", email="spy@agency.gov", permissions=["CLASSIFIED"])
        req_auth = SearchRequest(query="data", strategies=[RetrieverType.LANCE_DENSE], user_context=ctx_auth)

        async with engine:
            res_auth = await engine.execute(req_auth)
            assert "Classified Data" in res_auth.hits[0].distilled_text

        # Case 2: Unauthorized
        ctx_civilian = UserContext(sub="civ", email="civ@public.com", permissions=[])
        req_civ = SearchRequest(query="data", strategies=[RetrieverType.LANCE_DENSE], user_context=ctx_civilian)

        async with engine:
            res_civ = await engine.execute(req_civ)
            # Should be empty because fetcher returned None, so Scout filtered it or returned empty string
            assert res_civ.hits[0].distilled_text == ""
