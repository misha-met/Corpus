## Project: Offline Advanced RAG (Apple Silicon Optimized)
**Role:** Senior AI Engineer / Systems Architect
**Objective:** Build a high-performance, offline-first RAG (Retrieval-Augmented Generation) system optimized for Apple Silicon (M-series chips).
**Constraints:** strict offline functionality, highly optimized VRAM usage, Python-based.

---

## v2 Architecture: Three-Mode RAG Configuration

### Operating Modes

The system supports three operating modes, each optimized for different use cases and hardware configurations:

| Mode | LLM Model | Quantization | Context Window | Retrieval Budget | RAM Requirement |
|------|-----------|--------------|----------------|------------------|-----------------|
| `regular` | Qwen3-30B-A3B-Instruct-4bit | 4-bit | 128K | 100K | 32GB |
| `power-fast` | Qwen3-30B-A3B-Instruct-2507-8bit | 8-bit | 128K | 100K | 48GB |
| `power-deep-research` | Qwen3-Next-80B-A3B-Instruct-4bit | 4-bit | 32K | 20K | 96GB |

#### Mode Selection Precedence

1. **CLI flag**: `--mode regular|power-fast|power-deep-research`
2. **Environment variable**: `RAG_MODE=power-fast`
3. **Auto-detection**: Based on system RAM (uses psutil if available)

#### Retrieval Parameters by Mode

| Parameter | Regular | Power-Fast / Deep-Research |
|-----------|---------|---------------------------|
| `top_k_dense` | 100 | 300 |
| `top_k_sparse` | 100 | 300 |
| `top_k_fused` | 50 | 150 |
| `top_k_rerank` | 20 | 80 |
| `top_k_final` | 5 | 15 |

#### Reranker Threshold Filtering

The system implements **quality-based filtering** after reranking to prevent low-quality documents from diluting context:

| Mode | Threshold Score | Safety Net (Min Docs) | Rationale |
|------|----------------|----------------------|-----------|
| `regular` | -6.0 (Aggressive) | 3 | Hardware Protection: On 32GB RAM, every token counts. Only high-confidence text. |
| `power-fast` | -8.0 (Balanced) | 5 | Speed Optimization: Cuts garbage but keeps context rich. |
| `power-deep-research` | -10.0 (Permissive) | 10 | Max Recall: Weak documents might contain subtle connections the 80B model needs. |

**Algorithm**: After reranking, filter documents with `rerank_score >= threshold`. If filtered count < min_docs, use top min_docs instead (safety net).

---

### Token Budget Packing

The system implements **greedy token budget packing** to maximize context utilization:

1. **Greedy Selection**: Documents are packed in priority order (by rerank score)
2. **Consecutive Fail Guard**: Stops after 3 consecutive documents that don't fit
3. **Truncation Fallback**: Large documents are truncated at sentence boundaries
4. **Accurate Counting**: Uses MLX native tokenizer for precise token counts

```
Algorithm: enforce_token_budget(docs, max_tokens, tokenizer)
  packed = []
  consecutive_fails = 0
  for doc in docs (priority order):
    if doc fits:
      pack it, reset fails
    elif truncation possible:
      truncate and pack
    else:
      skip, increment fails
      if fails >= 3: STOP
  return packed
```

---

### Metrics Collection

The retrieval pipeline collects comprehensive metrics:

#### Budget Metrics
- `budget_tokens`: Maximum allowed tokens
- `used_tokens`: Actual tokens used
- `utilization_pct`: Budget utilization percentage
- `docs_packed/skipped/truncated`: Document counts

#### Timing Metrics (per stage)
- `dense_search_ms`: Embedding search time
- `sparse_search_ms`: BM25 search time
- `rrf_fusion_ms`: RRF fusion time
- `rerank_ms`: Reranking time
- `dedup_ms`: Deduplication time
- `total_ms`: Total pipeline time

#### Reranker Metrics
- Score distribution: min, max, mean, std

#### Threshold Filtering Metrics
- `threshold_value`: Score threshold applied
- `items_before/after_threshold`: Document counts
- `safety_net_triggered`: Whether minimum docs override was used

#### Deduplication Metrics
- Before/after counts, reduction percentage

---

### Early Parent-Level Deduplication

To reduce redundant reranking computations, deduplication runs **before** reranking:

1. After RRF fusion, group children by `parent_id`
2. Keep only the highest-scored child per parent
3. Trim to `top_k_fused` results
4. Track reduction metrics for analysis

---

### 1. Technology Stack
*   **LLM Inference:** `mlx-lm` (Apple Machine Learning/Metal optimized).
*   **Embeddings:** `sentence-transformers` (Run on CPU to save VRAM).
*   **Re-ranking:** `FlagEmbedding` (BGE-Reranker).
*   **Sparse Search:** `rank_bm25`.
*   **Vector Store:** `chromadb` (Local) or `faiss-cpu`.
*   **Key-Value Store:** `sqlite3` (for Parent Chunks).

### 2. Core Architecture Modules

#### Module A: Data Layer (The Parent-Child Splitter)
Do not use simple character splitting. Implement a hierarchical class structure:
*   **`ParentChunk`**: 1000–1500 tokens. Contains the full "meat" of the text. Stored in SQLite.
*   **`ChildChunk`**: 200–300 tokens. Overlapping slides derived from the Parent. Stored in Vector Index.
*   **`Metadata`**: Every chunk must track: `source_id`, `page_number`, `header_path` (e.g., "Intro > Methodology"), and `parent_id`.
*   **Markdown Awareness**: Use a markdown header splitter to ensure `ChildChunks` inherit the semantic header path of their parent section.

#### Module B: The Storage Engine
*   **Vector Index**: Stores embeddings of `ChildChunks`. Use `int8` quantization if possible for the index.
*   **Lexical Index**: Maintain a BM25 index of the raw text tokens from `ChildChunks` for keyword matching.
*   **Context Store (SQLite)**: A persistent DB mapping `parent_id` -> `full_text_content`.

#### Module C: The Hybrid Retrieval Pipeline (Two-Stage)
Implement a class `RetrievalEngine` with this exact flow:
1.  **Parallel Search**:
    *   *Dense*: Cosine similarity on Vector Index (configurable top_k).
    *   *Sparse*: BM25 score on Lexical Index (configurable top_k).
2.  **Fusion (RRF)**: Combine results using Reciprocal Rank Fusion:
    $$Score = \sum (1 / (60 + rank))$$
3.  **Early Deduplication**: Keep highest-RRF-scored child per unique parent_id.
4.  **Re-Ranking (The Filter)**:
    *   Take top N fused results (configurable).
    *   Pass query + text pairs through `BGE-Reranker-v2-m3`.
    *   Apply threshold filtering to remove low-quality documents.
    *   Collect score distribution metrics.
    *   Select Top K absolute best matches (configurable).
5.  **Context Expansion**:
    *   Map the final results to their `ParentChunks`.
    *   Apply token budget packing with greedy algorithm.
    *   Return the *Parent* text to the LLM to ensure full context.

#### Module D: Hardware Configuration & Model Loader
Implement a configuration manager `config.py` that:
1. Detects system RAM (using psutil if available, with platform fallbacks)
2. Supports CLI override (`--mode`) and environment variable (`RAG_MODE`)
3. Validates RAM requirements and warns if insufficient
4. Maintains backward compatibility with legacy tier names

**Mode Configurations (v2):**

| Mode | Use Case | Model | Notes |
|------|----------|-------|-------|
| `regular` | Daily queries | Qwen3-30B-4bit | Balanced speed/quality |
| `power-fast` | Deep retrieval | Qwen3-30B-8bit | Higher precision |
| `power-deep-research` | Complex research | Qwen3-Next-80B-4bit | Best reasoning |

*Note: All embedding and BM25 operations must be forced to CPU (`device='cpu'`) to reserve Metal/GPU VRAM strictly for the LLM.*

### 3. Implementation Steps for the Agent

1.  **Setup**: Create a clean Python project structure (`/src`, `/data`, `/models`).
2.  **Data Structures**: Define the Pydantic models for `ParentChunk` and `ChildChunk`.
3.  **Ingestion**: Write the Markdown/PDF ingestion script that populates the Vector Store and SQLite.
4.  **Retrieval Logic**: Implement the Hybrid Search + RRF + Early Dedup + Reranker pipeline with metrics.
5.  **Generation**: Implement the `mlx-lm` generator class with tokenizer exposure and budget packing.
6.  **CLI/Interface**: A simple script to query the system with mode selection and metrics reporting.

---

### CLI Usage

```bash
# Auto-detect mode based on RAM
python -m src.cli query "What is the main argument?"

# Specify mode explicitly
python -m src.cli query "Analyze the methodology" --mode power-fast

# Set mode via environment variable
export RAG_MODE=power-deep-research
python -m src.cli query "Compare the two approaches"

# Ingest with specific mode
python -m src.cli ingest document.md --source-id doc1 --mode regular
```

---

### Dependencies

**Required:**
- chromadb, rank-bm25, pydantic, sentence-transformers
- FlagEmbedding, mlx-lm, pypdf, pdfminer.six, PyMuPDF
- pytesseract, pdf2image

**Optional:**
- psutil (for cross-platform RAM detection)

***

>