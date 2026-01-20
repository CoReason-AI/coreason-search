# The Architecture and Utility of coreason-search

## 1. The Philosophy (The Why)

**coreason-search** functions as the "Execution Engine" of the CoReason ecosystem. It rejects the prevailing "retrieval as a commodity" mindset, instead positioning information retrieval as a multi-stage, strategic refinement process.

The specific architecture addresses three critical pain points in modern RAG (Retrieval-Augmented Generation) systems:

1.  **The Chunking Fallacy:** Most systems chop documents into 512-token chunks, destroying global context. `coreason-search` adheres to a "Late Chunking / No Chunking" philosophy, designed to leverage 32k-context embeddings (like Qwen2-7B or NV-Embed) to vectorize entire scientific papers as sovereign units.
2.  **The Precision-Recall Tradeoff:** It implements a disciplined pipeline—**Embed, Retrieve, Fuse, Rerank, Distill**. By decoupling high-recall retrieval from high-precision reranking and final "context distillation" (The Scout), it ensures that downstream Agents receive high-signal evidence, not just keyword matches.
3.  **The Dual-Mode Requirement:** It uniquely serves two opposing masters: the **Ad-Hoc Agent** needing rapid, synthesized answers, and the **Systematic Researcher** needing rigid, reproducible, exhaustive boolean searches (PRISMA compliance).

## 2. Under the Hood (The Dependencies & logic)

The package relies on a carefully curated stack to deliver this "Research-Grade" retrieval:

*   **LanceDB:** The backbone. Chosen for its unique ability to handle both high-performance Vector Search and robust Full-Text Search (via `tantivy`) in a single, serverless file format. This enables true Hybrid Search without managing separate Elasticsearch and Pinecone instances.
*   **Pydantic & Pydantic-Settings:** Ensures strict schema validation for every search request and configuration object, critical for the reproducibility required in systematic reviews.
*   **Rank-BM25 & Tantivy:** Powers the Sparse/Boolean retrieval strategies, allowing for complex "PubMed-style" queries (e.g., `("Drug A"[Title] OR "Drug B"[Abstract])`).
*   **The Scout (Distiller):** A specialized logic layer that takes the top retrieved documents and surgically removes irrelevant sentences before they reach the LLM context window.

Internally, `coreason-search` uses a **Strategy Pattern**. The `SearchEngine` acts as an orchestrator, dispatching requests to specific Retrievers (`DenseRetriever`, `SparseRetriever`, `GraphRetriever`). It then aggregates results using **Reciprocal Rank Fusion (RRF)**, ensuring that a document found by both Vector and Graph strategies floats to the top.

## 3. In Practice (The How)

### Example 1: The "Hybrid RAG" Query
This is the "Happy Path" for an Agent. It combines Dense Vector search (conceptual similarity) with Graph traversal (symbolic connections) to find deep context.

```python
from coreason_search.engine import SearchEngine
from coreason_search.schemas import SearchRequest, RetrieverType

# Initialize the engine (loads Singletons for DB, Embedder, etc.)
engine = SearchEngine()

# Design a Hybrid Strategy: Vector + Graph
request = SearchRequest(
    query="What are the adverse events associated with protein inhibition?",
    strategies=[RetrieverType.LANCE_DENSE, RetrieverType.GRAPH_NEIGHBOR],
    top_k=5,
    fusion_enabled=True,   # Merge results via RRF
    rerank_enabled=True,   # Re-sort for precision
    distill_enabled=True   # Activate 'The Scout' to strip fluff
)

# Execute
response = engine.execute(request)

for hit in response.hits:
    # 'distilled_text' is the surgically cleaned content ready for the LLM
    print(f"[{hit.score:.4f}] {hit.doc_id}: {hit.distilled_text[:200]}...")
```

### Example 2: The Systematic Review Stream
For researchers who need to screen 5,000 papers without crashing memory, `coreason-search` switches to a generator-based streaming mode.

```python
from coreason_search.engine import SearchEngine
from coreason_search.schemas import SearchRequest, RetrieverType

engine = SearchEngine()

# Strict Boolean Logic for a Systematic Review
# Note: "Systematic" mode is a generator, suited for batch processing
request = SearchRequest(
    query='("Liver Failure"[Title]) AND ("Paracetamol"[Abstract])',
    strategies=[RetrieverType.LANCE_FTS],
    # Filters can be applied (e.g., Year > 2020)
    filters={"year": {"$gt": 2020}}
)

# Returns an iterator, not a list.
# Handles large datasets efficiently via pagination.
review_stream = engine.execute_systematic(request)

print("Streaming systematic results...")
for hit in review_stream:
    # Process each paper (e.g., send to a screening agent)
    print(f"Found candidate: {hit.doc_id}")
```
