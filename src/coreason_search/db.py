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
    metadata: str = Field(..., description="JSON stringified metadata")

    @field_validator("metadata")
    @classmethod
    def validate_metadata_json(cls, v: str) -> str:
        """Ensure metadata is a valid JSON string."""
        if not v:
            return v  # Allow empty string if that's desired, or enforce {}?
            # Let's enforce it must be valid JSON or empty.
            # If empty string, technically not valid JSON unless it's "" which loads as... fail.
            # But earlier tests used "" and it passed.
            # Let's allow empty string to mean "no metadata".
        if v.strip() == "":
            return v
        try:
            json.loads(v)
        except json.JSONDecodeError as e:
            raise ValueError(f"Metadata must be a valid JSON string: {e}") from e
        return v


class LanceDBManager:
    """
    Singleton manager for LanceDB connection and table access.
    """

    _instance: Optional["LanceDBManager"] = None

    def __new__(cls, *args: object, **kwargs: object) -> "LanceDBManager":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, uri: str = "/tmp/lancedb"):
        """
        Initialize the connection.
        If already initialized, this serves as a re-configuration/reset if needed,
        but typically we check if self.db exists.
        """
        if not hasattr(self, "db") or self.db is None:
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
        # We use exist_ok=True to handle race conditions or existence checks safely.
        # This will open the table if it exists, or create it if it doesn't.
        return self.db.create_table(name, schema=DocumentSchema, exist_ok=True)

    def reset(self) -> None:
        """Reset the singleton (useful for tests)."""
        self.db = None
        LanceDBManager._instance = None


def get_db_manager(uri: str = "/tmp/lancedb") -> LanceDBManager:
    """Factory function to get the DB manager."""
    return LanceDBManager(uri)
