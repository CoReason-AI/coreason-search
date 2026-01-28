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

from coreason_identity.models import UserContext

from coreason_search.schemas import RetrieverType, SearchRequest


def test_search_request_serialization() -> None:
    """Test that SearchRequest with UserContext serializes to JSON correctly."""
    ctx = UserContext(
        user_id="user_serial",
        email="serial@co.ai",
        scopes=["read"],
        claims={"project_context": "proj_1"},
    )
    req = SearchRequest(
        query="test",
        strategies=[RetrieverType.LANCE_DENSE],
        user_context=ctx,
    )

    # Serialize
    json_str = req.model_dump_json()
    data = json.loads(json_str)

    assert data["query"] == "test"
    assert data["user_context"]["user_id"] == "user_serial"
    assert data["user_context"]["email"] == "serial@co.ai"
    assert data["user_context"]["scopes"] == ["read"]
    assert data["user_context"]["claims"]["project_context"] == "proj_1"


def test_search_request_deserialization() -> None:
    """Test that SearchRequest can be deserialized from JSON with UserContext."""
    json_data = {
        "query": "test",
        "strategies": ["lance_dense"],
        "user_context": {
            "user_id": "user_deserial",
            "email": "deserial@co.ai",
            "scopes": ["write"],
            "claims": {},
        },
    }
    json_str = json.dumps(json_data)

    # Deserialize
    req = SearchRequest.model_validate_json(json_str)

    assert isinstance(req.user_context, UserContext)
    assert req.user_context.user_id == "user_deserial"
    assert req.user_context.email == "deserial@co.ai"
    assert "write" in req.user_context.scopes


def test_search_request_round_trip() -> None:
    """Test full JSON round trip."""
    ctx = UserContext(user_id="u1", email="u1@e.com")
    req = SearchRequest(query="q", strategies=[RetrieverType.LANCE_DENSE], user_context=ctx)

    json_str = req.model_dump_json()
    req2 = SearchRequest.model_validate_json(json_str)

    assert req2.user_context == req.user_context
    assert req2.user_context is not None
    assert req2.user_context.user_id == "u1"
