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
from typing import Optional

import lancedb
from lancedb.pydantic import LanceModel, Vector
from pydantic import Field, field_validator

from coreason_search.utils.logger import logger

# Constants
VECTOR_DIM = 1024  # Must match the Embedder dimension
DEFAULT_TABLE_NAME = "documents"


class DocumentSchema(LanceModel):  # type: ignore[misc]
    """
    Schema for documents stored in LanceDB.
    Using LanceModel for direct Pydantic integration.
    """

    doc_id: str = Field(..., description="Unique document identifier")
    vector: Vector(VECTOR_DIM) = Field(..., description="Dense vector embedding")  # type: ignore
    content: str = Field(..., description="Main content of the document")
    title: Optional[str] = Field(default=None, description="Document title for FTS")
    abstract: Optional[str] = Field(default=None, description="Document abstract for FTS")
    metadata: str = Field(..., description="JSON stringified metadata")

    @field_validator("metadata")
    @classmethod
    def validate_metadata_json(cls, v: str) -> str:
        """Ensure metadata is a valid JSON string."""
        if not v:
            return v
        if v.strip() == "":
            return v
        try:
            json.loads(v)
        except json.JSONDecodeError as e:
            raise ValueError(f"Metadata must be a valid JSON string: {e}") from e
        return v


class LanceDBManager:
    """
    Manager for LanceDB connection and table access.
    """

    def __init__(self, uri: str = "/tmp/lancedb"):
        """
        Initialize the connection.
        """
        self.uri = uri
        self.connect(uri)

    def connect(self, uri: str) -> None:
        """Connect to the LanceDB instance."""
        logger.info(f"Connecting to LanceDB at {uri}")
        self.db = lancedb.connect(uri)
        self.uri = uri

    def get_table(self, name: str = DEFAULT_TABLE_NAME) -> lancedb.table.Table:
        """
        Get the table, creating it if it doesn't exist.
        """
        # Check if table exists
        tables = self.db.list_tables()
        if hasattr(tables, "tables"):
            table_names = tables.tables
        else:
            table_names = tables

        if name in table_names:
            return self.db.open_table(name)

        return self.db.create_table(name, schema=DocumentSchema)


# Global singleton instance
_DB_MANAGER: Optional[LanceDBManager] = None


def get_db_manager(uri: Optional[str] = None) -> LanceDBManager:
    """
    Factory function to get the DB manager.
    Implements explicit singleton pattern.
    If 'uri' is provided, initializes/overwrites the singleton with that URI.
    If 'uri' is None, returns the existing singleton (or initializes default).
    """
    global _DB_MANAGER

    if uri is not None:
        if _DB_MANAGER is None or _DB_MANAGER.uri != uri:
            _DB_MANAGER = LanceDBManager(uri)
        return _DB_MANAGER

    if _DB_MANAGER is None:
        _DB_MANAGER = LanceDBManager()

    return _DB_MANAGER


def reset_db_manager() -> None:
    """Reset the singleton (useful for tests)."""
    global _DB_MANAGER
    _DB_MANAGER = None
