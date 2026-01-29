# Copyright (c) 2025 CoReason, Inc.
#
# This software is proprietary and dual-licensed.
# Licensed under the Prosperity Public License 3.0 (the "License").
# A copy of the license is available at https://prosperitylicense.com/versions/3.0.0
# For details, see the LICENSE file.
# Commercial use beyond a 30-day trial requires a separate license.
#
# Source Code: https://github.com/CoReason-AI/coreason_search

from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI
from fastapi.responses import JSONResponse, StreamingResponse

from coreason_search.engine import SearchEngineAsync
from coreason_search.schemas import SearchRequest, SearchResponse


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Lifespan context manager for initializing resources."""
    # Initialize the engine (loads DB, Embedder, etc.)
    engine = SearchEngineAsync()
    await engine.__aenter__()
    app.state.engine = engine
    yield
    # Cleanup resources
    await engine.__aexit__(None, None, None)


app = FastAPI(lifespan=lifespan, title="CoReason Search Service (Service L)")


@app.post("/search", response_model=SearchResponse)  # type: ignore[untyped-decorator]
async def search(request: SearchRequest) -> SearchResponse:
    """Ad-Hoc Search Endpoint (RAG).

    Executes a standard search query and returns aggregated results.
    """
    engine: SearchEngineAsync = app.state.engine
    return await engine.execute(request)


@app.post("/search/systematic")  # type: ignore[untyped-decorator]
async def search_systematic(request: SearchRequest) -> StreamingResponse:
    """Systematic Search Endpoint (Review Mode).

    Executes a systematic search and streams results as NDJSON.
    Triggers 'SYSTEMATIC_SEARCH_START' audit event before yielding hits.
    """
    engine: SearchEngineAsync = app.state.engine

    async def stream_generator() -> AsyncGenerator[str, None]:
        async for hit in engine.execute_systematic(request):
            yield hit.model_dump_json() + "\n"

    return StreamingResponse(stream_generator(), media_type="application/x-ndjson")


@app.get("/health")  # type: ignore[untyped-decorator]
async def health() -> JSONResponse:
    """Health Check Endpoint.

    Verifies database connectivity and returns embedder configuration.
    """
    engine: SearchEngineAsync = app.state.engine

    db_status = "connected"
    try:
        # Check if the DB connection object exists
        if engine.db_manager.db is None:
            db_status = "disconnected"
    except Exception:  # pragma: no cover
        db_status = "error"

    return JSONResponse(
        {
            "status": "ready",
            "database": db_status,
            "embedder": engine.config.embedding.provider,
        }
    )
