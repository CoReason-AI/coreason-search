# Usage Guide

coreason-search can be used as a standalone Python library or as a microservice (Service L).

## Server Mode (Service L)

### Running the Server

You can run the server using the Docker image or directly via Uvicorn.

**Docker:**

```bash
docker run -p 8000:8000 -v /path/to/data:/tmp/lancedb coreason-search:latest
```

**Local (Uvicorn):**

```bash
uvicorn coreason_search.server:app --host 0.0.0.0 --port 8000
```

### API Endpoints

#### 1. Ad-Hoc Search (`POST /search`)

Standard Retrieval-Augmented Generation (RAG) search.

**Request:**

```json
{
  "query": "mechanism of action of aspirin",
  "strategies": ["lance_dense", "lance_fts"],
  "top_k": 5,
  "fusion_enabled": true,
  "rerank_enabled": true,
  "distill_enabled": true
}
```

**Response:** Returns a JSON object with a list of `Hit` objects and provenance metadata.

#### 2. Systematic Search (`POST /search/systematic`)

Streamed results for systematic reviews. Returns Newline Delimited JSON (NDJSON).

**Request:**

```json
{
  "query": {"title": "aspirin", "abstract": "liver"},
  "strategies": ["lance_fts"],
  "top_k": 1000
}
```

**Response:** Stream of JSON objects (one `Hit` per line).

#### 3. Health Check (`GET /health`)

Returns the service status.

```json
{
  "status": "ready",
  "database": "connected",
  "embedder": "hf"
}
```

## Library Usage

You can still use the synchronous facade in your Python code:

```python
from coreason_search.engine import SearchEngine
from coreason_search.schemas import SearchRequest, RetrieverType

with SearchEngine() as engine:
    request = SearchRequest(
        query="test query",
        strategies=[RetrieverType.LANCE_DENSE]
    )
    response = engine.execute(request)
    print(response.hits)
```
