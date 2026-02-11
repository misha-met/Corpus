## Project: Offline Advanced RAG (Apple Silicon Optimized)

---

## 1. Core Architecture Overview

The system implements a **hybrid retrieval-augmented generation (RAG) pipeline** optimized for Apple Silicon, combining:

- **Dense retrieval** (embedding-based semantic search)
- **Sparse retrieval** (BM25 keyword search)
- **Reciprocal Rank Fusion** (RRF) for combining results
- **Reranking** (Jina Reranker v3 MLX – listwise cosine scoring with hardware-aware context windowing)
- **Token budget packing** (greedy document packing with truncation fallback)
- **Intent-based generation** (task-specific prompt engineering)
- **Toggleable citations** (optional academic-style source attribution)

---

## 2. Technology Stack

*   **LLM Inference:** `mlx-lm` (Apple Machine Learning/Metal optimized).
*   **Embeddings:** `sentence-transformers` (Run on CPU to save VRAM).
*   **Re-ranking:** `jinaai/jina-reranker-v3-mlx` (Jina Reranker v3, MLX-native).
*   **Sparse Search:** `rank_bm25`.
*   **Vector Store:** `chromadb` (Local, HNSW with cosine distance).
*   **Key-Value Store:** `sqlite3` (for Parent Chunks).

---

## 3. Core Architecture Modules

### Module A: Data Layer

Hierarchical chunk structure for efficient retrieval and context expansion:

*   **ParentChunk**: 1000–1500 tokens. Full text content stored in SQLite.
*   **ChildChunk**: 200–300 tokens. Overlapping chunks derived from parent. Stored in vector index for efficient retrieval.
*   **Metadata**: Tracks `source_id`, `page_number`, `page_label`, `header_path`, `parent_id`.

### Module B: Storage Engine

*   **Vector Index**: Embeddings of child chunks stored in Chroma (HNSW, cosine distance).
*   **Lexical Index**: BM25 index of child chunk text for keyword-based search.
*   **Context Store**: SQLite database mapping `parent_id` to full parent text for context expansion.

### Module C: Retrieval Pipeline

1.  **Parallel Search**:
    *   **Dense**: Cosine similarity on vector index (configurable `top_k`).
    *   **Sparse**: BM25 search on lexical index (configurable `top_k`).
2.  **RRF Fusion**: Combine dense and sparse results using Reciprocal Rank Fusion ($$Score = \sum (1 / (60 + rank))$$).
3.  **Early Deduplication**: Keep highest-scored child per unique `parent_id` to reduce redundancy before expensive reranking.
4.  **Reranking**: Jina Reranker v3 (MLX-native) performs listwise scoring via cosine similarity between projected query and document embeddings. A context-window safety guard dynamically truncates the document list to stay within the backbone’s 131K context limit.
5.  **Threshold Filtering**: Remove documents below score threshold based on operating mode.
6.  **Context Expansion**: Map child chunks to parent chunks and apply token budget packing.

### Module D: Configuration

System configuration via `config.py`:

1. Detects system RAM (psutil if available, fallback to platform detection)
2. Supports CLI override (`--mode`) and environment variable (`RAG_MODE`)
3. Auto-selects mode based on available RAM
4. Embedding and BM25 operations forced to CPU to preserve LLM VRAM

---

## 4. Three-Mode RAG Configuration

### Operating Modes

The system supports two operating modes, with `regular` mode auto-scaling based on detected RAM:

| Mode | LLM Model | Quantization | Context Window | Retrieval Budget | RAM Requirement |
|------|-----------|--------------|----------------|------------------|-----------------|
| `regular` (32GB) | Qwen3-30B-A3B-Instruct-4bit | 4-bit | 16K | 8K | 32GB |
| `regular` (48-63GB) | Qwen3-30B-A3B-Instruct-4bit | 4-bit | 64K | 32K | 48GB |
| `regular` (64GB+) | Qwen3-30B-A3B-Instruct-4bit | 4-bit | 64K | 48K | 64GB |
| `power-deep-research` | Qwen3-Next-80B-A3B-Instruct-4bit | 4-bit | 48K | 40K | 64GB |

#### Mode Selection Precedence

1. **CLI flag**: `--mode regular|power-deep-research`
2. **Environment variable**: `RAG_MODE=power-deep-research`
3. **Auto-detection**: Defaults to `regular` (uses RAM-aware parameter scaling)

#### Retrieval Parameters by Mode

| Parameter | Regular (32GB) | Regular (48-63GB) | Regular Plus (64GB+) | Deep-Research |
|-----------|----------------|-------------------|----------------------|---------------|
| `top_k_dense` | 100 | 100 | 200 | 400 |
| `top_k_sparse` | 100 | 100 | 200 | 400 |
| `top_k_fused` | 50 | 50 | 100 | 200 |
| `top_k_rerank` | 20 | 20 | 40 | 60 |
| `top_k_final` | 5 | 5 | 8 | 10 |

#### Reranker Threshold Filtering

Documents are filtered after reranking based on score thresholds.  Jina v3 scores are cosine similarities in the **0.0–1.0** range:

| Mode | Threshold Score | Minimum Documents |
|------|----------------|-------------------|
| `regular` (32-63GB) | 0.05 | 3 |
| `regular` (64GB+) | 0.04 | 4 |
| `power-deep-research` | 0.015 | 6 |

After reranking, documents with `rerank_score >= threshold` are kept. If filtered count < min_docs, top min_docs results are used instead.

---

## 5. Retrieval Pipeline Details

### Early Parent-Level Deduplication

To reduce redundant reranking computations, deduplication runs **before** reranking:

1. After RRF fusion, group children by `parent_id`
2. Keep only the highest-scored child per parent
3. Trim to `top_k_fused` results
4. Track reduction metrics for analysis

---

## 6. Token Budget Packing

The system implements **greedy token budget packing** to maximize context utilization while staying within token limits:

1. **Greedy Selection**: Documents are packed in priority order (by rerank score)
2. **Consecutive Fail Guard**: Stops after 3 consecutive documents that don't fit
3. **Truncation Fallback**: Large documents are truncated at sentence boundaries
4. **Accurate Counting**: Uses MLX native tokenizer for precise token counts
5. **Metadata Preservation**: Tracks original document indices to maintain metadata alignment even when documents are truncated

```
Algorithm: enforce_token_budget(docs, max_tokens, tokenizer)
  packed = []
  packed_indices = []
  consecutive_fails = 0
  for i, doc in enumerate(docs) (priority order):
    if doc fits:
      pack it, record index i, reset fails
    elif truncation possible:
      truncate and pack, record index i
    else:
      skip, increment fails
      if fails >= 3: STOP
  return packed, packed_indices
```

---

## 7. Toggleable Inline Citations (Academic Mode)

### Feature Overview

The system supports **toggleable inline citations** that enable academic-style source attribution in generated responses. This is useful for research contexts where source transparency is critical.

| Mode | Flag | Description |
|------|------|-------------|
| **Academic Mode** | `--cite` | Enables inline citations with `[SourceID, p. X]` format |
| **Casual Mode** | `--no-cite` | Disables citations for cleaner, conversational output |

### Citation Format

When Academic Mode is enabled, chunks are formatted with citation markers:

```
[CHUNK START | SOURCE: document_id | PAGE: 42]
... chunk content ...
[CHUNK END]
```

The LLM is instructed to cite sources using the format `[SourceID, p. X]` for every factual claim.

### Implementation Details

1. **Context Formatting**: `format_context_with_citations()` wraps chunks with source/page markers
2. **Source Legend**: Non-trivial source mappings (where source_id ≠ document name) are appended for disambiguation
3. **Prompt Engineering**: Citation rules are conditionally injected into the system message only when enabled
4. **Metadata Preservation**: Token budget packing tracks original indices to preserve metadata alignment even for truncated documents

### CLI Usage

```bash
# Enable Academic Mode (inline citations)
python -m src.cli query "What does Chomsky argue?" --cite

# Disable citations (Casual Mode)
python -m src.cli query "Explain the main idea" --no-cite

# Default behavior controlled by CITATIONS_ENABLED_DEFAULT in config.py
python -m src.cli query "Summarize this document"
```

### Configuration

The default citation behavior is controlled by `CITATIONS_ENABLED_DEFAULT` in `src/config.py`. CLI flags take precedence:

1. `--cite` → Citations enabled
2. `--no-cite` → Citations disabled
3. Neither → Uses `CITATIONS_ENABLED_DEFAULT`

---

## 8. Metrics Collection

The retrieval and packing pipeline collects comprehensive metrics for analysis and optimization:

### Budget Metrics
- `budget_tokens`: Maximum allowed tokens
- `used_tokens`: Actual tokens used
- `utilization_pct`: Budget utilization percentage
- `docs_packed/skipped/truncated`: Document counts

### Timing Metrics (per stage)
- `dense_search_ms`: Embedding search time
- `sparse_search_ms`: BM25 search time
- `rrf_fusion_ms`: RRF fusion time
- `rerank_ms`: Reranking time
- `dedup_ms`: Deduplication time
- `budget_packing_ms`: Token budget packing time
- `total_ms`: Total pipeline time

### Reranker Metrics
- Score distribution: min, max, mean, std

### Threshold Filtering Metrics
- `threshold_value`: Score threshold applied
- `items_before/after_threshold`: Document counts
- `safety_net_triggered`: Whether minimum docs override was used

### Deduplication Metrics
- Before/after counts, reduction percentage

---

## 9. CLI Usage Examples

```bash
# Auto-detect mode based on RAM (uses regular with RAM-based parameter scaling)
python -m src.cli query "What is the main argument?"

# Specify deep-research mode explicitly
python -m src.cli query "Analyze the methodology" --mode power-deep-research

# Set mode via environment variable
export RAG_MODE=power-deep-research
python -m src.cli query "Compare the two approaches"

# Ingest with specific mode
python -m src.cli ingest document.md --source-id doc1 --mode regular

# Enable citations with custom intent
python -m src.cli query "Explain the concept" --cite --intent explain

# Only show retrieved context without generation
python -m src.cli query "Search for key terms" --no-generate
```

---

## 10. Dependencies

### Required
- `chromadb` - Vector database (HNSW, cosine distance)
- `rank-bm25` - Sparse search indexing
- `pydantic` - Configuration validation
- `sentence-transformers` - Embedding model inference
- `jinaai/jina-reranker-v3-mlx` - Reranker model (MLX-native)
- `mlx-lm` - LLM inference on Apple Silicon
- `pypdf` - PDF text extraction
- `pdfminer.six` - Fallback PDF extraction
- `PyMuPDF` - Additional PDF processing with page labels
- `pytesseract` - OCR fallback for scanned PDFs
- `pdf2image` - Image extraction for OCR

### Optional
- `psutil` - Cross-platform RAM detection for auto mode selection

---

## 11. Known Limitations & Future Improvements

- **Reranker Context Safety**: A dynamic guard truncates the document list via binary search so the listwise prompt never exceeds the backbone’s 131K context window
- **Tokenizer Robustness**: Special token IDs (`<|embed_token|>`, `<|rerank_token|>`) are resolved from the tokenizer vocabulary at runtime; hard-coded fallbacks are used only when lookup fails
- **Numerical Stability**: An $\epsilon$ term is added to the cosine similarity denominator to prevent NaN from zero-vector embeddings
- **Offline Mode**: `_enable_offline_if_cached` verifies that model snapshots contain actual weight files before forcing `HF_HUB_OFFLINE`
- **Metadata Alignment**: Now correctly preserved across truncation (tracked via indices)
- **PDF Page Labels**: Uses per-page API for consistent label extraction across PyMuPDF and pypdf
- **Source Legend**: Filters out redundant mappings to reduce noise
- **Citation Rules**: Conditionally applied only in Academic Mode to avoid prompt bloat

