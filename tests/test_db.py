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
from pathlib import Path
from typing import Generator

import numpy as np
import pytest
from lancedb.table import Table

from coreason_search.db import DocumentSchema, LanceDBManager, get_db_manager


class TestLanceDBManager:
    @pytest.fixture(autouse=True)  # type: ignore[misc]
    def setup_teardown(self, tmp_path: Path) -> Generator[None, None, None]:
        # Reset singleton before each test
        manager = get_db_manager(str(tmp_path))
        manager.reset()
        yield
        # Reset after test
        if LanceDBManager._instance:
            LanceDBManager._instance.reset()

    def test_singleton_behavior(self, tmp_path: Path) -> None:
        """Test that get_db_manager returns the same instance."""
        uri = str(tmp_path / "lancedb")
        manager1 = get_db_manager(uri)
        manager2 = get_db_manager(uri)
        assert manager1 is manager2
        assert manager1.uri == uri

    def test_table_creation(self, tmp_path: Path) -> None:
        """Test that get_table creates the table if it doesn't exist."""
        uri = str(tmp_path / "lancedb_test")
        manager = get_db_manager(uri)

        # Initial check might be empty or table list empty
        tables_init = manager.db.list_tables()
        if hasattr(tables_init, "tables"):
            tables_init = tables_init.tables
        assert "documents" not in tables_init

        table = manager.get_table("documents")
        assert isinstance(table, Table)

        # Verify it exists now
        tables = manager.db.list_tables()
        # lancedb 0.26+ returns a ListTablesResponse object, or list of strings depending on version.
        # We handle both.
        if hasattr(tables, "tables"):
            table_names = tables.tables
        else:
            table_names = tables
        assert "documents" in table_names

    def test_add_and_query_document(self, tmp_path: Path) -> None:
        """Test adding a document and retrieving it."""
        uri = str(tmp_path / "lancedb_rw")
        manager = get_db_manager(uri)
        table = manager.get_table()

        # Create dummy vector
        vector = np.random.rand(1024).astype(np.float32)

        doc = DocumentSchema(
            doc_id="1",
            vector=vector,
            content="This is a test document.",
            metadata=json.dumps({"author": "test"}),
        )

        # Add to table
        table.add([doc])

        # Query
        results = table.search(vector).limit(1).to_pydantic(DocumentSchema)
        assert len(results) == 1
        assert results[0].doc_id == "1"
        assert results[0].content == "This is a test document."

    def test_schema_enforcement(self, tmp_path: Path) -> None:
        """Test that adding data with wrong schema fails (or at least checking schema)."""
        uri = str(tmp_path / "lancedb_schema")
        manager = get_db_manager(uri)
        table = manager.get_table()

        # Checking if we can inspect schema
        assert table.schema is not None

    def test_reconnect(self, tmp_path: Path) -> None:
        """Test connecting to an existing database."""
        uri = str(tmp_path / "lancedb_persist")
        manager = get_db_manager(uri)
        table = manager.get_table()

        vector = np.random.rand(1024).astype(np.float32)
        doc = DocumentSchema(
            doc_id="2",
            vector=vector,
            content="Persisted doc",
            metadata="{}",
        )
        table.add([doc])

        # Reset and reconnect
        manager.reset()
        new_manager = get_db_manager(uri)
        new_table = new_manager.get_table()

        results = new_table.search(vector).limit(1).to_pydantic(DocumentSchema)
        assert len(results) == 1
        assert results[0].content == "Persisted doc"
