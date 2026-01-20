# coreason-search

The Unified Retrieval Execution Engine for the CoReason ecosystem.

[![License](https://img.shields.io/badge/License-Prosperity%203.0-blue)](https://github.com/CoReason-AI/coreason_search/blob/main/LICENSE)
[![CI](https://github.com/CoReason-AI/coreason_search/actions/workflows/ci.yml/badge.svg)](https://github.com/CoReason-AI/coreason_search/actions/workflows/ci.yml)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)

**coreason-search** serves as the "Librarian" of the CoReason architecture, designed to solve three distinct problems: Ad-Hoc Reasoning (RAG), Systematic Evidence Synthesis, and Context Distillation. It adopts a "Late Chunking / No Chunking" philosophy, utilizing SOTA 32k context embeddings to process full documents.

## Features

-   **Long-Context Sovereignty:** Utilizes SOTA embeddings (e.g., Qwen3-Embedding) with 32k context windows to ingest full research papers without "chunk-and-pray" methods.
-   **Modular Strategy Pattern:** Supports multiple retrieval strategies:
    -   *Dense Retriever:* Vector-based semantic search using LanceDB.
    -   *Sparse Retriever:* Boolean/Keyword search for systematic reviews (PubMed style).
    -   *Graph Retriever:* Symbolic 1-hop neighbor expansion using `coreason-graph-nexus`.
-   **Hybrid Fusion:** Implements Reciprocal Rank Fusion (RRF) to merge results from vector and keyword searches robustly.
-   **Precision Re-Ranking:** Uses Cross-Encoders to re-rank top results for maximum precision.
-   **Context Distillation (The Scout):** Compresses documents by stripping irrelevant sentences, maximizing the signal-to-noise ratio for downstream LLMs.
-   **Systematic Search Mode:** Supports "Research-Grade" reproducible reviews with strict boolean logic and generator-based pagination for large result sets.

## Installation

```bash
pip install coreason-search
```

## Usage

```python
from coreason_search.engine import SearchEngine
from coreason_search.schemas import SearchRequest, RetrieverType

# Initialize the engine (loads configuration automatically)
engine = SearchEngine()

# Define a search request
request = SearchRequest(
    query="mechanism of action of aspirin in liver failure",
    strategies=[RetrieverType.LANCE_DENSE, RetrieverType.LANCE_FTS],
    fusion_enabled=True,
    rerank_enabled=True,
    distill_enabled=True,
    top_k=5
)

# Execute the search
response = engine.search(request)

# Process results
for hit in response.hits:
    print(f"[{hit.score:.4f}] {hit.doc_id}")
    print(f"Distilled Context: {hit.distilled_text}\n")
```
