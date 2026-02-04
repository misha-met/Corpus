## Project: Offline Advanced RAG (Apple Silicon Optimized)

---

## v2 Architecture: Three-Mode RAG Configuration

### Operating Modes

The system supports three operating modes, each optimized for different use cases and hardware configurations:

| Mode | LLM Model | Quantization | Context Window | Retrieval Budget | RAM Requirement |
|------|-----------|--------------|----------------|------------------|-----------------||
| `regular` | Qwen3-30B-A3B-Instruct-4bit | 4-bit | 64K | 32K | 32GB |
| `power-fast` | Qwen3-30B-A3B-Instruct-2507-8bit | 8-bit | 96K | 50K | 64GB |
| `power-deep-research` | Qwen3-Next-80B-A3B-Instruct-4bit | 4-bit | 32K | 20K | 64GB |

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

Documents are filtered after reranking based on score thresholds:

| Mode | Threshold Score | Minimum Documents |
|------|----------------|-------------------|
| `regular` | -6.0 | 3 |
| `power-fast` | -8.0 | 5 |
| `power-deep-research` | -10.0 | 10 |

After reranking, documents with `rerank_score >= threshold` are kept. If filtered count < min_docs, top min_docs results are used instead.

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

#### Module A: Data Layer

Hierarchical chunk structure:

*   **ParentChunk**: 1000–1500 tokens. Full text content stored in SQLite.
*   **ChildChunk**: 200–300 tokens. Overlapping chunks derived from parent. Stored in vector index.
*   **Metadata**: Tracks `source_id`, `page_number`, `header_path`, `parent_id`.

#### Module B: Storage Engine

*   **Vector Index**: Embeddings of child chunks with int8 quantization.
*   **Lexical Index**: BM25 index of child chunk text for keyword search.
*   **Context Store**: SQLite database mapping parent_id to full text content.

#### Module C: Retrieval Pipeline

1.  **Parallel Search**:
    *   Dense: Cosine similarity on vector index (configurable top_k).
    *   Sparse: BM25 search on lexical index (configurable top_k).
2.  **RRF Fusion**: Combine results using Reciprocal Rank Fusion ($$Score = \sum (1 / (60 + rank))$$).
3.  **Early Deduplication**: Keep highest-scored child per unique parent_id.
4.  **Reranking**: Score document pairs with BGE-Reranker-v2-m3.
5.  **Threshold Filtering**: Remove documents below score threshold.
6.  **Context Expansion**: Map to parent chunks and apply token budget packing.

#### Module D: Configuration

System configuration via `config.py`:

1. Detects system RAM (psutil if available, fallback to platform detection)
2. Supports CLI override (`--mode`) and environment variable (`RAG_MODE`)
3. Auto-selects mode based on available RAM
4. Embedding and BM25 operations forced to CPU to preserve LLM VRAM

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