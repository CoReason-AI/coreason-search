# Copyright (c) 2025 CoReason, Inc.
#
# This software is proprietary and dual-licensed.
# Licensed under the Prosperity Public License 3.0 (the "License").
# A copy of the license is available at https://prosperitylicense.com/versions/3.0.0
# For details, see the LICENSE file.
# Commercial use beyond a 30-day trial requires a separate license.
#
# Source Code: https://github.com/CoReason-AI/coreason_search

from pathlib import Path
from typing import Generator
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from lancedb.pydantic import LanceModel, Vector
from pydantic import Field, ValidationError

from coreason_search.db import VECTOR_DIM, DocumentSchema, get_db_manager, reset_db_manager


# --- Helper for Legacy Schema ---
class LegacyDocumentSchema(LanceModel):  # type: ignore[misc]
    """
    Simulates the schema before title/abstract were added.
    """

    doc_id: str = Field(..., description="Unique document identifier")
    vector: Vector(VECTOR_DIM) = Field(..., description="Dense vector embedding")  # type: ignore
    content: str = Field(..., description="Main content of the document")
    metadata: str = Field(..., description="JSON stringified metadata")


class TestDBEdgeCases:
    @pytest.fixture(autouse=True)  # type: ignore[misc]
    def setup_teardown(self, tmp_path: Path) -> Generator[None, None, None]:
        # Reset singleton before each test
        reset_db_manager()
        yield
        # Reset after test
        reset_db_manager()

    def test_boundary_strings(self, tmp_path: Path) -> None:
        """Test empty strings, whitespace, and Unicode in title/abstract."""
        uri = str(tmp_path / "lancedb_boundary")
        manager = get_db_manager(uri)
        table = manager.get_table()

        vector = np.random.rand(VECTOR_DIM).astype(np.float32)

        docs = [
            DocumentSchema(doc_id="empty", vector=vector, content="c", metadata="{}", title="", abstract=""),
            DocumentSchema(
                doc_id="whitespace", vector=vector, content="c", metadata="{}", title="   ", abstract=" \t "
            ),
            DocumentSchema(
                doc_id="unicode",
                vector=vector,
                content="c",
                metadata="{}",
                title="Study on 💊",
                abstract="The 🧬 results were ✅.",
            ),
        ]

        table.add(docs)

        # Retrieve and verify
        # Note: We sort or fetch by ID to be sure
        res_empty = table.search().where("doc_id = 'empty'").limit(1).to_pydantic(DocumentSchema)[0]
        assert res_empty.title == ""
        assert res_empty.abstract == ""

        res_ws = table.search().where("doc_id = 'whitespace'").limit(1).to_pydantic(DocumentSchema)[0]
        assert res_ws.title == "   "
        assert res_ws.abstract == " \t "

        res_uni = table.search().where("doc_id = 'unicode'").limit(1).to_pydantic(DocumentSchema)[0]
        assert res_uni.title == "Study on 💊"
        assert res_uni.abstract == "The 🧬 results were ✅."

    def test_type_validation_failure(self, tmp_path: Path) -> None:
        """Test that passing wrong types raises ValidationError."""
        vector = np.random.rand(VECTOR_DIM).astype(np.float32)

        # Integer instead of string for title
        with pytest.raises(ValidationError):
            DocumentSchema(
                doc_id="fail_type",
                vector=vector,
                content="c",
                metadata="{}",
                title=12345,
            )

    def test_batch_mixed_fields(self, tmp_path: Path) -> None:
        """Test batch insertion where some docs have fields and others don't."""
        uri = str(tmp_path / "lancedb_batch_mixed")
        manager = get_db_manager(uri)
        table = manager.get_table()

        vector = np.random.rand(VECTOR_DIM).astype(np.float32)

        docs = [
            DocumentSchema(doc_id="full", vector=vector, content="c", metadata="{}", title="T", abstract="A"),
            DocumentSchema(doc_id="none", vector=vector, content="c", metadata="{}", title=None, abstract=None),
            DocumentSchema(doc_id="partial", vector=vector, content="c", metadata="{}", title="T", abstract=None),
        ]

        table.add(docs)
        assert table.count_rows() == 3

        # Verify 'none'
        r_none = table.search().where("doc_id = 'none'").limit(1).to_pydantic(DocumentSchema)[0]
        assert r_none.title is None
        assert r_none.abstract is None

        # Verify 'partial'
        r_partial = table.search().where("doc_id = 'partial'").limit(1).to_pydantic(DocumentSchema)[0]
        assert r_partial.title == "T"
        assert r_partial.abstract is None

    def test_legacy_schema_compatibility(self, tmp_path: Path) -> None:
        """
        Complex Scenario: Existing DB created with Legacy Schema (no title/abstract).
        We attempt to open it with the New Schema.
        """
        uri = str(tmp_path / "lancedb_legacy")

        # 1. Manually create a table using the Legacy Schema
        # We bypass the manager's get_table default to force legacy schema
        import lancedb

        db = lancedb.connect(uri)

        vector = np.random.rand(VECTOR_DIM).astype(np.float32)
        legacy_doc = LegacyDocumentSchema(doc_id="old_1", vector=vector, content="Old content", metadata="{}")

        # Create table with explicit legacy schema
        table = db.create_table("documents", schema=LegacyDocumentSchema)
        table.add([legacy_doc])

        # 2. Now, use the App's Manager (which uses DocumentSchema) to access it
        manager = get_db_manager(uri)
        # exist_ok=True ensures we open the existing one.
        # CAUTION: LanceDB create_table with schema might try to overwrite or might just open?
        # If schema doesn't match, what happens?
        # get_table calls: self.db.create_table(name, schema=DocumentSchema, exist_ok=True)

        # In LanceDB:
        # If exist_ok=True and table exists, it opens it.
        # It usually DOES NOT automatically migrate schema just by opening.
        app_table = manager.get_table()

        # 3. Try to Read existing data using New Schema
        # The physical table lacks 'title' and 'abstract'.
        # The Pydantic model expects them (Optional).
        # LanceDB's to_pydantic usually maps columns by name.
        # If column missing, does it default to None?

        results = app_table.search().limit(1).to_pydantic(DocumentSchema)

        # If this works, LanceDB fills missing columns with None for Optional fields?
        assert len(results) == 1
        assert results[0].doc_id == "old_1"
        # The crucial check:
        assert results[0].title is None
        assert results[0].abstract is None

        # 4. Try to Add NEW data with title/abstract to this legacy table
        # Currently, LanceDB's simple `add` does NOT automatically migrate schema if columns are missing.
        # This part of the test verifies that we fail safely (ValueError) rather than corrupting data,
        # documenting that schema migration is required for Writes.
        new_doc = DocumentSchema(
            doc_id="new_1", vector=vector, content="New content", metadata="{}", title="Evolution", abstract="Works"
        )

        # Expect ValueError because 'title' field is missing in physical table
        with pytest.raises(ValueError, match="not found in target schema"):
            app_table.add([new_doc])

    def test_list_tables_legacy_return_type(self, tmp_path: Path) -> None:
        """
        Test that get_table handles older lancedb versions where list_tables returns a list of strings,
        covering the 'else' branch in db.py.
        """
        uri = str(tmp_path / "lancedb_legacy_return")
        manager = get_db_manager(uri)

        # Mock the db instance's list_tables method
        with patch.object(manager.db, "list_tables") as mock_list_tables:
            # Simulate returning a simple list of strings
            mock_list_tables.return_value = ["documents"]

            # Since "documents" is in the list, it should try to OPEN, not CREATE
            # We also mock open_table to verify it's called
            with patch.object(manager.db, "open_table") as mock_open:
                mock_open.return_value = MagicMock()

                manager.get_table("documents")

                mock_open.assert_called_once_with("documents")
