# Copyright (c) 2025 CoReason, Inc.
#
# This software is proprietary and dual-licensed.
# Licensed under the Prosperity Public License 3.0 (the "License").
# A copy of the license is available at https://prosperitylicense.com/versions/3.0.0
# For details, see the LICENSE file.
# Commercial use beyond a 30-day trial requires a separate license.
#
# Source Code: https://github.com/CoReason-AI/coreason_search

import pytest
from coreason_identity.models import UserContext
from pydantic import ValidationError

from coreason_search.schemas import RetrieverType, SearchRequest


def test_user_context_minimal() -> None:
    """Test UserContext with only required fields."""
    ctx = UserContext(sub="user1", email="user1@example.com")
    assert ctx.sub == "user1"
    assert ctx.email == "user1@example.com"
    assert ctx.project_context is None
    assert ctx.permissions == []

    req = SearchRequest(query="test", strategies=[RetrieverType.LANCE_DENSE], user_context=ctx)
    assert req.user_context == ctx


def test_user_context_full() -> None:
    """Test UserContext with all fields."""
    ctx = UserContext(
        sub="user2",
        email="user2@example.com",
        project_context="proj_alpha",
        permissions=["read", "write", "admin"],
    )
    assert ctx.project_context == "proj_alpha"
    assert "admin" in ctx.permissions

    req = SearchRequest(query="test", strategies=[RetrieverType.LANCE_DENSE], user_context=ctx)
    assert req.user_context == ctx


def test_user_context_invalid_email() -> None:
    """Test validation failure on invalid email."""
    # coreason-identity models likely validate email format if they use Pydantic's EmailStr
    with pytest.raises(ValidationError):
        UserContext(sub="user3", email="not-an-email")


def test_user_context_missing_required() -> None:
    """Test validation failure on missing required fields."""
    with pytest.raises(ValidationError):
        UserContext(sub="user4")  # Missing email


def test_search_request_integration_none() -> None:
    """Test SearchRequest accepts None for user_context."""
    req = SearchRequest(query="test", strategies=[RetrieverType.LANCE_DENSE], user_context=None)
    assert req.user_context is None


def test_search_request_type_mismatch() -> None:
    """Test SearchRequest rejects dictionary if strict type is enforced."""
    # Pydantic *might* try to coerce a dict into the model if it matches the schema.
    # Let's see if it coerces or fails. Usually it coerces.
    # If it coerces, that's fine, but the type on the object should be UserContext.

    ctx_dict = {"sub": "user5", "email": "user5@example.com"}
    req = SearchRequest(query="test", strategies=[RetrieverType.LANCE_DENSE], user_context=ctx_dict)

    assert isinstance(req.user_context, UserContext)
    assert req.user_context.sub == "user5"


def test_search_request_invalid_dict_coercion() -> None:
    """Test SearchRequest fails if dict is passed but missing fields."""
    ctx_dict = {"sub": "user6"}  # Missing email
    with pytest.raises(ValidationError):
        SearchRequest(query="test", strategies=[RetrieverType.LANCE_DENSE], user_context=ctx_dict)
