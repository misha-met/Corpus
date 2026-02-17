## Project: DH Notebook Offline RAG Application (Apple Silicon Optimized)

## 1. Core Architecture Overview

The application is now a **full-stack offline RAG system** composed of:

- **FastAPI backend** (`src/api.py`) exposing ingest, source management, and query endpoints
- **Next.js frontend** (`frontend/`) with source panel, chat panel, and citation viewer
- **Decoupled RAG engine** (`src/rag_engine.py`) reusable by CLI and API
- **Hybrid retrieval pipeline** (LanceDB ANN + FTS with native RRF)
- **Mode-aware retrieval pathing** (including a low-latency `turbo` path)
- **Optional inline citations** for source-grounded responses

---

## 2. Technology Stack

- **LLM Inference:** `mlx-lm` (Metal-optimized Apple Silicon inference)
- **Embeddings:** `sentence-transformers` (CPU device to preserve VRAM)
- **Re-ranking:** `jinaai/jina-reranker-v3-mlx` (MLX-native listwise reranker)
- **Storage:** `lancedb` + `pyarrow` (vector + FTS hybrid retrieval, parent/source tables)
- **Backend API:** `fastapi`, `uvicorn`, `sse-starlette`, `python-multipart`
- **Frontend:** `next.js` (React/TypeScript app in `frontend/`)
- **Tokenizer compatibility dependency:** `mistral-common`

---

## 3. Core Architecture Modules

### Module A: Data Model

Hierarchical chunk structure for retrieval efficiency and context expansion:

- **ParentChunk:** ~1000–1500 tokens, canonical context text in `parent_chunks`
- **ChildChunk:** ~200–300 tokens, overlapping retrieval units in `child_chunks`
- **Metadata:** `source_id`, `parent_id`, `page_number`, `display_page`, `header_path`, etc.

### Module B: Storage Engine (LanceDB)

Unified LanceDB-backed storage with three core tables:

- **`child_chunks`**: vectors + FTS, queried with native hybrid search + RRF
- **`parent_chunks`**: parent text for post-retrieval context expansion
- **`source_summaries`**: per-source summaries and source/snapshot paths

### Module C: Retrieval Pipeline

1. **Hybrid Search**: single LanceDB hybrid query (vector ANN + BM25 FTS + RRF)
2. **Early Parent Dedup**: keep highest-scored child per `parent_id`
3. **Conditional Rerank**:
   - `regular` / `power-deep-research`: rerank top candidates with Jina v3
   - `turbo`: reranker bypassed
4. **Threshold Filtering + Safety Net**: keep scores above threshold, then enforce minimum docs
5. **Final Dedup / Boilerplate Filter**
6. **Conditional Context Expansion**:
   - enabled modes: expand children to parents
   - `turbo`: keep child text (no parent expansion)
7. **Token Budget Packing** for generation context

### Module D: RAG Engine Layer

`RagEngine` centralizes ingest/query orchestration and exposes:

- **Synchronous query path** returning structured `QueryResult`
- **Streaming query event path** (`query_events`) for frontend token/status streams
- **Latency reporting integration** (returned in `QueryResult.latency_report`)

### Module E: Application Layer

- **CLI** (`src/cli.py`): local query/ingest workflows (`--mode`, `--cite`, `--latency`)
- **API** (`src/api.py`):
  - `POST /api/query` (JSON or SSE stream)
  - `POST /api/chat` (text stream for chat transport)
  - source CRUD + upload/ingest endpoints
- **Frontend** (`frontend/src`): two-panel UI (sources + chat), streaming tokens, citation modal

---

## 4. RAG Configuration and Modes

### Operating Modes

The system supports **three user-selectable modes** (`turbo`, `regular`, `power-deep-research`).
`regular` auto-scales internally by detected RAM tier.

| Mode | LLM Model | Quantization | Context Window | Retrieval Budget | RAM Requirement |
|------|-----------|--------------|----------------|------------------|-----------------|
| `turbo` | Ministral-3-3B-Instruct-2512-4bit (MLX) | 4-bit | 8K | 4K | 16GB+ |
| `regular` (32GB) | Qwen3-30B-A3B-Instruct-2507-4bit | 4-bit | 16K | 8K | 32GB |
| `regular` (48-63GB) | Qwen3-30B-A3B-Instruct-2507-4bit | 4-bit | 64K | 32K | 48GB |
| `regular` (64GB+) | Qwen3-30B-A3B-Instruct-2507-4bit | 4-bit | 64K | 48K | 64GB |
| `power-deep-research` | Qwen3-Next-80B-A3B-Instruct-4bit | 4-bit | 48K | 40K | 64GB |

#### Mode Selection Precedence

1. **CLI flag**: `--mode turbo|regular|power-deep-research`
2. **Environment variable**: `RAG_MODE=<mode>`
3. **Auto-detection**: defaults to `regular` and RAM-aware parameter scaling

#### Retrieval Parameters by Mode

| Parameter | Turbo | Regular (32GB) | Regular (48-63GB) | Regular Plus (64GB+) | Deep-Research |
|-----------|-------|----------------|-------------------|----------------------|---------------|
| `top_k_fused` | 20 | 50 | 50 | 100 | 200 |
| `top_k_rerank` | Disabled in pipeline | 20 | 20 | 40 | 60 |
| `top_k_final` | 3 | 5 | 5 | 8 | 8 |

#### Turbo Retrieval Path

`turbo` mode is optimized for lowest latency:

- reranker disabled
- context expansion disabled
- lower retrieval budget and context window
- thresholds applied on fused retrieval score

#### Threshold Filtering

| Mode | Threshold Score | Minimum Documents |
|------|----------------|-------------------|
| `turbo` | 0.01 (fused score) | 3 |
| `regular` (32-63GB) | 0.05 (rerank score) | 3 |
| `regular` (64GB+) | 0.04 (rerank score) | 4 |
| `power-deep-research` | 0.04 (rerank score) | 4 |

---

## 5. Token Budget Packing

The system uses greedy packing to maximize context utility within token limits:

1. Priority-order packing (post-retrieval ranking order)
2. Stop after 3 consecutive non-fitting documents
3. Sentence-boundary truncation fallback for oversized docs
4. MLX tokenizer-based token counting
5. Index preservation for metadata/citation alignment

---

## 6. Citations and Context Formatting

### Toggleable Citation Modes

- **Academic mode**: `--cite` (or API `citations_enabled=true`)
- **Casual mode**: `--no-cite` (or API `citations_enabled=false`)

### Chunk Context Format

When citations are enabled, chunks are wrapped as:

```
[CHUNK N | SOURCE: source_id | PAGE: display_page]
... chunk text ...
[CHUNK END]
```

The system also builds a source legend when source IDs differ from display names.

### Streaming Citation Events

During streaming query execution, a structured citation list is emitted as a dedicated event so the frontend can bind numbered references to source/chunk metadata.

---

## 7. Streaming and Protocols

The backend supports two stream styles:

1. **SSE query stream** (`/api/query` with `stream=true`) emitting typed events:
   - `status`, `intent`, `sources`, `citations`, `token`, `error`, `complete`
2. **Text stream chat endpoint** (`/api/chat`) for chat transport compatibility

Internal stream events are typed dataclasses in `src/query_events.py` and encoded by `src/stream_protocol.py` utilities.

---

## 8. Metrics and Latency Instrumentation

### Retrieval/Budget Metrics

- token budget utilization and packed/skipped/truncated counts
- per-stage timing (`hybrid_search_ms`, `dedup_ms`, `rerank_ms`, `budget_packing_ms`, `total_ms`)
- reranker score distributions
- threshold filtering and safety-net metrics

### Latency Reporting

`RagEngine` records wall and stage spans via `LatencyProfiler`; CLI can print the formatted report with `--latency`.

---

## 9. Intent Classification

The system uses a two-stage intent classifier:

1. **Heuristic regex classifier** for common intents
2. **LLM fallback** when confidence is below threshold (default `0.70`)

Default fallback model: `mlx-community/LFM2-8B-A1B-4bit`.

---

## 10. Usage Examples

```bash
# CLI query (auto mode)
python -m src.cli query "What is the main argument?"

# Turbo mode + citations
python -m src.cli query "What docs are in here" --mode turbo --cite

# Print latency report
python -m src.cli query "Analyze this" --latency

# API health
curl http://127.0.0.1:8000/api/health

# API streamed query (SSE)
curl -N -H "Content-Type: application/json" \
  -d '{"query":"Compare these views","stream":true}' \
  http://127.0.0.1:8000/api/query
```

---

## 11. Dependencies

### Required

- `lancedb`, `pyarrow`, `pydantic`
- `sentence-transformers`, `mlx-lm`, `mistral-common`, `safetensors`
- `pypdf`, `pdfminer.six`, `PyMuPDF`, `pytesseract`, `pdf2image`
- `fastapi`, `uvicorn`, `httpx`, `python-multipart`, `sse-starlette`

### Optional

- `psutil` for robust RAM detection and mode auto-scaling
