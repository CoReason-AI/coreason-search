# **Product Requirements Document: coreason-search**

**Domain:** Neural Information Retrieval, Systematic Review Execution, & GraphRAG

**Architectural Role:** The "Librarian" / The Execution Engine

**Core Philosophy:** "Retrieval is a multi-stage refinement process. Fuse vectors, keywords, and graphs. Distill signal from noise. Never chunk if you don't have to."

**Dependencies:** coreason-archive (Storage), coreason-graph-nexus (Symbolic Context), coreason-protocol (Systematic Strategy), lancedb (Vector Store), rank-bm25 (Sparse), llmlingua (Distillation)

## ---

**1\. Executive Summary**

coreason-search is the **Unified Retrieval Execution Engine** for the CoReason ecosystem. It abstracts the complexity of modern information retrieval behind a clean, modular API.

It is designed to solve three distinct problems with one architecture:

1. **Ad-Hoc Reasoning (RAG):** When an Agent asks "How does Drug X work?", it uses **Hybrid Fusion** (Vector \+ Graph \+ Keyword) and **Long-Context Embeddings** to retrieve deep, semantically rich context without hallucination.
2. **Systematic Evidence Synthesis:** When a Researcher requires a "Research-Grade" review, it executes rigid, pre-specified **Boolean Protocols** (defined by coreason-protocol) with 100% reproducibility and auditability (PRISMA compliance).
3. **Context Distillation:** To prevent "Lost in the Middle" issues with large documents, it implements a **Scout** layer that reads retrieved documents and surgically removes irrelevant paragraphs before the data hits the reasoning agent.

Crucially, it adopts a **"Late Chunking / No Chunking"** philosophy for scientific literature, utilizing SOTA **32k Context Embeddings** to vectorize entire research papers as single units, preserving global context.

## ---

**2\. Functional Philosophy**

The agent must implement the **Embed-Retrieve-Fuse-Rerank-Distill Loop**:

1. **Long-Context Sovereignty (SOTA):** We reject the "chunk-and-pray" method. We use **Qwen3-Embedding** (or equivalent SOTA open weights) with a 32k context window. This allows coreason-refinery to ingest full PDF protocols as single vectors.
2. **Modular Strategy Pattern:** The search engine is not hardcoded. It executes a SearchStrategy.
   * *Strategy A (Semantic):* Dense Vector Retrieval.
   * *Strategy B (Systematic):* Boolean/Sparse Retrieval (PubMed style).
   * *Strategy C (GraphRAG):* Graph Traversal \+ Vector Hybrid.
3. **Reciprocal Rank Fusion (RRF):** When mixing strategies (e.g., Vector \+ Keyword), we use RRF to mathematically normalize and merge the ranked lists, ensuring robustness without manual weight tuning.
4. **Cross-Encoder Precision:** Retrieval optimizes for *Recall*. We add a **Re-Ranking** step using a Cross-Encoder to optimize for *Precision*.
5. **Context Distillation (The Scout):** Even relevant documents contain fluff. We apply a final **Compression** step to strip irrelevant sentences from the top documents, maximizing the signal-to-noise ratio for the LLM.

## ---

**3\. Core Functional Requirements (Component Level)**

### **3.1 The Embedder (The Encoder Module)**

**Concept:** A pluggable, unified interface for vectorization.

* **Default Model:** **Qwen3-Embedding-8B** (32k Context) or **NV-Embed-v2**.
  * *Rationale:* Beats OpenAI on MTEB benchmarks, supports sovereign local install, and handles full documents.
* **Configuration:** Defined in search\_config.yaml.
  * model\_name: "Alibaba-NLP/gte-Qwen2-7B-instruct"
  * device: "cuda" / "mps"
  * normalize\_embeddings: true
* **API:** embed(text: Union\[str, List\[str\]\]) \-\> np.ndarray
* **Role:** Used by refinery (Write) and search (Read) to guarantee vector space alignment.

### **3.2 The Retriever (The Strategy Engine)**

**Concept:** An abstract factory capable of executing different search types.

#### **Strategy A: Dense Retriever (Vector)**

* **Backend:** Queries **LanceDB**.
* **Logic:** Cosine Similarity on 32k document vectors.
* **Use Case:** "Find papers conceptually related to liver failure."

#### **Strategy B: Sparse Retriever (Systematic/Boolean)**

* **Backend:** Queries **Tantivy** or **LanceDB FTS** (Full Text Search).
* **Logic:** Strict Boolean execution.
  * *Query:* ("Aspirin"\[Title\] OR "Acetylsalicylic Acid"\[Abstract\]) AND ("Hepatotoxicity"\[MeSH\])
* **Use Case:** Systematic Reviews, Regulatory Audits, coreason-protocol execution.

#### **Strategy C: Graph Retriever (Symbolic)**

* **Backend:** Queries **coreason-graph-nexus**.
* **Logic:** 1-Hop Neighbor Expansion.
  * *Input:* "Protein X"
  * *Traversal:* Retrieve Node(Protein X) $\\to$ All connected Papers $\\to$ All connected Adverse Events.
* **Use Case:** Mechanism of Action (MoA) analysis.

### **3.3 The Fusion Engine (The Mixer)**

**Concept:** Merges results from multiple retrievers.

* **Algorithm:** **Reciprocal Rank Fusion (RRF)**.
  * $Score \= \\frac{1}{k \+ rank\_{vector}} \+ \\frac{1}{k \+ rank\_{keyword}}$
* **Function:** Takes List\[ResultList\] and outputs a single deduped List\[Document\].

### **3.4 The Re-Ranker (The Judge)**

**Concept:** Precision filtering.

* **Model:** Pluggable Cross-Encoder (e.g., BAAI/bge-reranker-v2-m3).
* **Action:**
  * Input: Top 50 results from Fusion.
  * Process: Scores every (Query, Doc) pair.
  * Output: Top 5 results strictly ordered by relevance score.

### **3.5 The Scout (Context Distiller)**

**Concept:** Intra-document filtering and compression.

* **Engine:** Wraps **LLMLingua-2** (Microsoft) or a specialized DeBERTa classifier.  Please make both available.
* **Input:** The User Query \+ The Top 5 Re-Ranked Full-Text Documents.
* **Mechanism:**
  1. **Segmentation:** Splits documents into logical units (sentences/paragraphs).
  2. **Scoring:** Calculates $P(Relevant | Unit, Query)$.
  3. **Filtering:** Drops units below threshold (e.g., \< 0.4).
* **Output:** A DistilledContext block containing only the sentences that actually answer the question.
* **Value:** Removes "distractor facts" and drastically reduces token cost.

## ---

**4\. Specialized Capability: Systematic Search Support**

To support **Research-Grade** requirements, coreason-search must implement a specific mode for reproducibility.

* **Mode:** execution\_mode="SYSTEMATIC"
* **Behavior:**
  * **Disables** Vector Search (unless explicitly requested for "Grey Literature" scoping).
  * **Enforces** Exact Boolean Logic.
  * **Disables** Re-ranking & Scouting (Systematic reviews require *all* matching papers unaltered).
  * **Logging:** Logs the exact query string, database version snapshot ID, and total hit count to coreason-veritas.
  * **Pagination:** Returns a generator for *all* results (e.g., 5,000 papers) to be fed into the coreason-protocol screening agent.

## ---

**5\. Integration Requirements**

* **coreason-refinery:**
  * Must import coreason-search.Embedder during ingestion.
  * If the embedding model changes in search, refinery must trigger a re-indexing job.
* **coreason-protocol:**
  * Generates the complex Boolean string.
  * Calls search.execute(strategy="SPARSE", query=boolean\_string).
* **coreason-cortex:**
  * Calls search.execute(strategy="HYBRID", query=natural\_language).
* **coreason-catalog:**
  * Uses search to vectorize metadata descriptions for its own discovery layer.

## ---

**6\. Data Schema**

### **SearchConfig (Modular Definition)**

Python

class RetrieverType(str, Enum):
    LANCE\_DENSE \= "lance\_dense"
    LANCE\_FTS \= "lance\_fts"
    GRAPH\_NEIGHBOR \= "graph\_neighbor"

class EmbeddingConfig(BaseModel):
    model\_name: str \= "Alibaba-NLP/gte-Qwen2-7B-instruct"
    context\_length: int \= 32768
    batch\_size: int \= 1

class SearchRequest(BaseModel):
    query: Union\[str, Dict\[str, str\]\] \# String for RAG, Dict for Boolean
    strategies: List\[RetrieverType\]   \# \["lance\_dense", "lance\_fts"\]
    fusion\_enabled: bool \= True
    rerank\_enabled: bool \= True
    distill\_enabled: bool \= True      \# Enable The Scout?
    top\_k: int \= 5
    filters: Optional\[Dict\[str, Any\]\] \# {"year": {"$gt": 2024}}

### **SearchResult**

Python

class Hit(BaseModel):
    doc\_id: str
    content: str
    original\_text: str  \# Full text
    distilled\_text: str \# Post-Scout text
    score: float
    source\_strategy: str \# "dense" or "sparse"
    metadata: Dict\[str, Any\]

class SearchResponse(BaseModel):
    hits: List\[Hit\]
    total\_found: int
    execution\_time\_ms: float
    provenance\_hash: str \# For audit

## ---

**7\. Implementation Directives for the Coding Agent**

1. **Dependency Injection:** The Embedder, Reranker, and Scout classes must be implemented as Singletons initialized from a config file. This allows swapping Qwen for NV-Embed without rewriting code.
2. **LanceDB is Mandatory:** Use **LanceDB** as the primary storage. It is the only embedded DB that supports high-performance Vector search *and* decent Full-Text Search (FTS) in a single file, which simplifies the Hybrid architecture.
3. **No-Chunking Pipeline:**
   * In refinery (ingestion): Read PDF $\\to$ Extract Text $\\to$ embedder.encode(full\_text).
   * Do *not* implement a sliding window chunker for this pipeline. Rely on the 32k context of the model.
4. **Systematic Pagination:** Implement a Python Generator for the Systematic Search mode. A review might return 10,000 papers; do not try to load them all into RAM. Stream them to the consumer.

\-\> (Dense/Sparse/Graph Workers) \-\> RRF Fusion \-\> Re-Ranker \-\> THE SCOUT (Distiller) \-\> Cortex\]
