# Copyright (c) 2025 CoReason, Inc.
#
# This software is proprietary and dual-licensed.
# Licensed under the Prosperity Public License 3.0 (the "License").
# A copy of the license is available at https://prosperitylicense.com/versions/3.0.0
# For details, see the LICENSE file.
# Commercial use beyond a 30-day trial requires a separate license.
#
# Source Code: https://github.com/CoReason-AI/coreason_search

import os
import pytest
from unittest.mock import patch
from fastapi.testclient import TestClient
from coreason_search.server import app
from coreason_search.schemas import RetrieverType
from coreason_search.db import reset_db_manager
from coreason_search.embedder import reset_embedder
from coreason_search.scout import reset_scout

@pytest.fixture
def client(monkeypatch, tmp_path):
    # Reset singletons before the test
    reset_db_manager()
    reset_embedder()
    reset_scout()

    # Configure the environment to use mock implementations and temporary storage
    monkeypatch.setenv("APP__EMBEDDING__PROVIDER", "mock")
    db_path = str(tmp_path / "lancedb")
    monkeypatch.setenv("APP__DATABASE_URI", db_path)

    # Use TestClient with the lifespan logic
    with TestClient(app) as c:
        yield c

    # Cleanup resets
    reset_db_manager()
    reset_embedder()
    reset_scout()

def test_health(client):
    """Verify health check endpoint returns correct status and configuration."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ready"
    assert data["database"] == "connected"
    assert data["embedder"] == "mock"

def test_search_ad_hoc(client):
    """Verify standard search endpoint accepts request and returns response."""
    request_data = {
        "query": "test query",
        "strategies": [RetrieverType.LANCE_DENSE],
        "top_k": 1,
        "fusion_enabled": False,
        "rerank_enabled": False,
        "distill_enabled": False
    }
    response = client.post("/search", json=request_data)
    assert response.status_code == 200
    data = response.json()
    assert "hits" in data
    assert "provenance_hash" in data
    assert data["total_found"] >= 0

def test_search_systematic(client):
    """Verify systematic search endpoint streams results."""
    request_data = {
        "query": "systematic test",
        "strategies": [RetrieverType.LANCE_DENSE],
        "top_k": 2
    }
    response = client.post("/search/systematic", json=request_data)
    assert response.status_code == 200
    assert response.headers["content-type"] == "application/x-ndjson"

    # Check that we can read the stream (even if empty)
    content = response.text
    # Mock engine with empty DB returns empty list
    # DenseRetriever returns [] if DB empty?
    # Yes, DB is empty in this test unless we populated it.
    # But the endpoint should work without crashing.

def test_search_systematic_audit(client):
    """Verify systematic search triggers audit event."""
    request_data = {
        "query": "audit test",
        "strategies": [RetrieverType.LANCE_DENSE],
        "top_k": 1
    }

    # Access the engine from the app state
    engine = client.app.state.engine

    with patch.object(engine.veritas, 'log_audit') as mock_log:
        response = client.post("/search/systematic", json=request_data)
        assert response.status_code == 200

        # Consume the stream to trigger execution
        _ = response.text

        # Verify call
        # We expect at least SYSTEMATIC_SEARCH_START
        calls = [args[0] for args, _ in mock_log.call_args_list]
        assert "SYSTEMATIC_SEARCH_START" in calls

        # Verify payload for start event
        start_call = [c for c in mock_log.call_args_list if c[0][0] == "SYSTEMATIC_SEARCH_START"][0]
        event_data = start_call[0][1]
        assert event_data["query"] == "audit test"
