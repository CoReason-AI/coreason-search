# Copyright (c) 2025 CoReason, Inc.
#
# This software is proprietary and dual-licensed.
# Licensed under the Prosperity Public License 3.0 (the "License").
# A copy of the license is available at https://prosperitylicense.com/versions/3.0.0
# For details, see the LICENSE file.
# Commercial use beyond a 30-day trial requires a separate license.
#
# Source Code: https://github.com/CoReason-AI/coreason_search

from typing import Generator
from unittest.mock import MagicMock, patch

import pytest
from coreason_identity.models import UserContext

from coreason_search.config import Settings
from coreason_search.db import get_db_manager, reset_db_manager
from coreason_search.embedder import reset_embedder
from coreason_search.engine import SearchEngineAsync
from coreason_search.schemas import Hit, RetrieverType, SearchRequest


class TestIdentityPropagation:
    """Verify identity flows through the system correctly."""

    @pytest.fixture(autouse=True)
    def setup_teardown(self, tmp_path: str) -> Generator[None, None, None]:
        self.db_path = str(tmp_path) + "/lancedb_prop"
        reset_db_manager()
        get_db_manager(self.db_path)
        reset_embedder()
        yield
        reset_db_manager()
        reset_embedder()

    def _get_engine(self) -> SearchEngineAsync:
        config = Settings(database_uri=self.db_path)
        return SearchEngineAsync(config)

    @pytest.mark.asyncio
    async def test_identity_passed_to_scout(self) -> None:
        """Verify the exact UserContext instance is passed to Scout."""
        engine = self._get_engine()

        # Mock Retriever to return something so Scout is called
        hit = Hit(
            doc_id="1",
            content="c",
            original_text="o",
            distilled_text="",
            score=1.0,
            source_strategy="dense",
            metadata={},
        )
        engine.dense_retriever.retrieve = MagicMock(return_value=[hit])  # type: ignore

        # Context
        ctx = UserContext(user_id="prop_user", email="prop@co.ai")
        req = SearchRequest(
            query="test", strategies=[RetrieverType.LANCE_DENSE], user_context=ctx, distill_enabled=True
        )

        # Patch Scout.distill
        with patch.object(engine.scout, "distill", wraps=engine.scout.distill) as mock_distill:
            async with engine:
                await engine.execute(req)

            # Verify call args
            assert mock_distill.call_count == 1
            call_args = mock_distill.call_args
            # distill(query, hits, user_context=...)
            passed_ctx = call_args.kwargs.get("user_context")

            assert passed_ctx is not None
            assert isinstance(passed_ctx, UserContext)
            assert passed_ctx.user_id == "prop_user"
            # Ideally it's the same object (unless copied somewhere)
            # Pydantic models might be copied if validation runs again, but here it's passed through.
            assert passed_ctx == ctx

    @pytest.mark.asyncio
    async def test_identity_preservation_in_fusion(self) -> None:
        """Verify identity is preserved even when fusion happens."""
        engine = self._get_engine()

        # Mock multiple retrievers
        h1 = Hit(
            doc_id="1",
            content="c",
            original_text="o",
            distilled_text="",
            score=1.0,
            source_strategy="dense",
            metadata={},
        )
        h2 = Hit(
            doc_id="2",
            content="c",
            original_text="o",
            distilled_text="",
            score=1.0,
            source_strategy="sparse",
            metadata={},
        )

        engine.dense_retriever.retrieve = MagicMock(return_value=[h1])  # type: ignore
        engine.sparse_retriever.retrieve = MagicMock(return_value=[h2])  # type: ignore

        ctx = UserContext(user_id="fusion_user", email="f@co.ai")
        req = SearchRequest(
            query="test",
            strategies=[RetrieverType.LANCE_DENSE, RetrieverType.LANCE_FTS],
            fusion_enabled=True,
            user_context=ctx,
            distill_enabled=True,
        )

        with patch.object(engine.scout, "distill") as mock_distill:
            mock_distill.return_value = []
            async with engine:
                await engine.execute(req)

            passed_ctx = mock_distill.call_args.kwargs.get("user_context")
            assert passed_ctx == ctx
