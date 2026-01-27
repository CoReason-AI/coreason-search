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

import pytest

from coreason_search.db import get_db_manager, reset_db_manager
from coreason_search.embedder import reset_embedder


@pytest.fixture
def setup_teardown_db_and_embedder(tmp_path: str) -> Generator[None, None, None]:
    """Shared fixture to setup DB and Embedder."""
    db_path = str(tmp_path) + "/lancedb_shared"
    reset_db_manager()
    get_db_manager(db_path)
    reset_embedder()
    yield
    reset_db_manager()
    reset_embedder()
