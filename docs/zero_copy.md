# Zero-Copy and Ephemeral Processing Architecture

This document describes the Zero-Copy and Ephemeral Processing architecture implemented in `coreason-search` version 0.3.0.

## Overview

Traditional search systems often persist full document content within the search index or return the full content in every search result (Hit). This leads to:
1.  **Data Duplication:** Content exists in the source of truth (e.g., S3, Database) and the search index.
2.  **Security Risks:** Access Control Lists (ACLs) must be perfectly synchronized.
3.  **Performance Overhead:** Carrying large text payloads through the retrieval pipeline is expensive.

The **Zero-Copy** architecture solves these issues by decoupling the "Index" from the "Content".

## Core Concepts

### 1. Pointer Hits
A `Hit` object can now exist as a "Pointer".
- `content` and `original_text` are `Optional` and can be `None`.
- `source_pointer` contains the necessary information (e.g., URI, ID) to fetch the content.
- `acls` field carries permission metadata.

```python
class Hit(BaseModel):
    # ...
    content: Optional[str] = None
    original_text: Optional[str] = None
    source_pointer: Optional[Dict[str, str]] = ...
    acls: List[str] = ...
```

### 2. Just-In-Time (JIT) Fetching
The `Scout` (Context Distiller) is responsible for processing content (segmentation, scoring, extraction). In the Zero-Copy architecture, the Scout fetches content **Just-In-Time** only for the top candidates that need processing.

- The Scout is initialized with a `content_fetcher` (Workspace).
- When processing a `Hit`, if `original_text` is missing but a `source_pointer` exists, the Scout calls the fetcher.

### 3. Ephemeral Processing
Crucially, the fetched content is **Ephemeral**.
- It is used immediately to generate `distilled_text` (snippets).
- It is **NOT** assigned back to `hit.original_text` or `hit.content`.
- The `Hit` returned to the user contains the `distilled_text` but remains a "Pointer" (no full text).

## Implementation Details

### Search Request Context
To support JIT fetching which may require authentication, the `SearchRequest` now includes a `user_context` field.

```python
request = SearchRequest(
    query="secret project",
    strategies=[...],
    user_context={"token": "..."}
)
```

This context is passed down to the `Scout` and the `content_fetcher`.

### Usage
To enable this architecture:
1.  Ensure your Index returns Hits with `source_pointer` and without `original_text`.
2.  Initialize the Scout (or Engine) with a valid `content_fetcher` callable.
3.  Pass `user_context` in search requests if required by your fetcher.

```python
def my_fetcher(pointer, context):
    # Fetch from S3/DB
    return "fetched content"

scout = MockScout(content_fetcher=my_fetcher)
# or configure via dependency injection
```
