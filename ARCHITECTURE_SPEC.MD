## Project: DH Notebook Offline RAG Application (Apple Silicon Optimized)

## 1. Core Architecture Overview

The application is a **full-stack offline RAG system** designed for local-first document analysis on Apple Silicon. All inference runs on-device via MLX -- no cloud APIs, no internet required after initial model download.

**Components:**

- **FastAPI backend** (`src/api.py`) -- REST + SSE streaming for ingest, source management, and query
- **Next.js frontend** (`frontend/`) -- two-panel chat UI with source management, streaming responses, and citation viewer
- **Decoupled RAG engine** (`src/rag_engine.py`) -- shared by CLI and API, orchestrates the full pipeline
- **Hybrid retrieval pipeline** (`src/retrieval.py`) -- LanceDB ANN + BM25 FTS with RRF fusion, Jina v3 reranking
- **MLX inference stack** -- generator (`src/generator.py`), embeddings (`src/embeddings.py`), reranker (`src/reranker.py`)
- **Mode-aware configuration** (`src/config.py`) -- RAM-based auto-scaling across active operating modes
- **Inline academic citations** -- source-grounded responses with chunk-level references

---

## 2. Technology Stack

| Layer | Technology | Notes |
|-------|-----------|-------|
| LLM Inference | `mlx-lm` (Metal-native) | KVCache support, streaming, thinking-block filtering |
| Embeddings | `mlx-lm` via `MlxEmbeddingModel` wrapper | Mean-pooled hidden states with L2 normalization |
| Re-ranking | `jinaai/jina-reranker-v3-mlx` | Listwise MLX-native reranker (Qwen3-0.6B backbone) |
| Storage | `lancedb` + `pyarrow` | Vector ANN + FTS hybrid search with native RRF |
| Backend API | `fastapi`, `uvicorn`, `python-multipart` | SSE streaming + multipart upload |
| Frontend | `next.js 16` (App Router), React 19, TypeScript 5 | Tailwind CSS 4, react-markdown, @assistant-ui/react |
---

## 3. Module Map

### 3.1 Data Models (`src/models.py`)

Hierarchical chunk structure for retrieval efficiency and context expansion:

- **`Metadata`** -- `source_id`, `parent_id`, `page_number`, `page_label`, `display_page`, `header_path`
- **`ParentChunk`** -- ~1000-1500 tokens, full paragraph context; UUID-based `id`, `text`, `metadata`
- **`ChildChunk`** -- ~200-300 tokens, overlapping retrieval units; references parent via `metadata.parent_id`

All dataclasses are frozen (immutable) with UUID generation at creation.

### 3.2 Storage Engine (`src/storage.py`)

Unified LanceDB-backed storage managing three table types:

| Table | Purpose | Key Columns |
|-------|---------|-------------|
| `child_chunks` | Vectors + FTS, hybrid-searched | `id`, `text`, `vector`, `source_id`, `parent_id`, `page_number`, `header_path` |
| `parent_chunks` | Parent text for context expansion | `parent_id`, `source_id`, `text`, page metadata |
| `source_summaries` | Per-source metadata (schema v2) | `source_id`, `summary`, `source_path`, `snapshot_path` |

**Key capabilities:**
- Native hybrid search (vector ANN + BM25 + RRF) via single LanceDB call
- FTS index management with configurable rebuild policies (`immediate`, `deferred`, `batch`)
- Automatic schema migration for v2 summary columns
- SQL-safe literal escaping and chunked IN clauses (>256 values)
- `_row_to_metadata()` helper for consistent metadata extraction across all query paths

### 3.3 Document Ingestion (`src/ingest.py`)

**Chunking parameters:**
- Parent chunks: 1000-1500 tokens (1200 target, 150 overlap)
- Child chunks: 200-300 tokens (250 target, 50 overlap)

**Child trailing-fragment handling:**
- Child segments below `CHILD_MIN_TOKENS` are not dropped.
- If a prior child exists, short trailing text is merged into that previous child.
- If no prior child exists, a sole short child is created so sparse facts remain searchable.

**Markdown ingestion:** Parses headers (H1-H6) into sections with hierarchical `header_path` (e.g., "Document > Chapter 1 > Section A"), then chunks each section into parent/child pairs.

**PDF ingestion:** Four-strategy fallback chain, each producing `_PageData` tuples that feed into a shared `_chunk_pages()` helper:

1. **pypdf** -- per-page text extraction with page labels
2. **pdfminer.six** -- whole-document fallback (no page boundaries)
3. **PyMuPDF (fitz)** -- per-page alternative extraction
4. **OCR (pytesseract + pdf2image)** -- rasterize-and-OCR, one page at a time to limit memory

Text cleaning: `clean_ocr_artifacts()` merges hyphenated line breaks.

**Summarization:** Optional LLM-generated summary (max 12,000 chars context), stored in `source_summaries` table.

### 3.4 Embeddings (`src/embeddings.py`)

`MlxEmbeddingModel` wraps MLX models as SentenceTransformer-compatible encoders:

- Forward through backbone `model.model.hidden_states`
- Mean pooling with attention mask: `sum(hidden * mask) / mask_sum`
- L2 normalization on CPU for numerical stability
- Batch processing (default batch_size=16, max_length=512)
- LRU-cached query vectors (max 100 queries) in the retrieval engine

### 3.5 Retrieval Pipeline (`src/retrieval.py`)

Six-stage pipeline orchestrated by `RetrievalEngine`:

```
query_text
  |
  v
[1. Query Embedding] -- MlxEmbeddingModel.encode()
  |
  v
[2. LanceDB Hybrid Search] -- vector ANN + BM25 FTS + RRF fusion
  |                           returns top_k_fused candidates
  v
[3. Rerank] -- Jina v3 listwise reranker
  |             replaces fusion score with cosine similarity score
  v
[4. Parent Dedup] -- keep highest-scored child per parent_id (uses reranker scores)
  |
  v
[5. Threshold Filter] -- adaptive threshold + safety net: min_docs guaranteed
  |
  v
[6. Context Expansion] -- fetch parent text
  |
  v
List[RetrievalResult] with text, metadata, scores, optional parent_text
```

Additional retrieval controls in the current pipeline:
- **Adaptive reranker threshold**: effective threshold is `max(config_threshold, top_score * 0.15)`.
- **Budget-aware expansion**: when selected chunks underfill budget, `k_final` expands up to a budget ceiling.
- **Sub-threshold expansion**: for starved contexts, additional below-threshold chunks are admitted up to a capped budget share.

Boilerplate filtering removes known LLM-speak ("as an AI", "as a language model", etc.) from retrieved chunks.

### 3.6 Reranker (`src/reranker.py`)

`JinaRerankerMLX` -- drop-in FlagReranker replacement, MLX-native:

- **Backbone:** Qwen3-0.6B (28 layers, 1024 hidden) loaded via `mlx_lm`
- **Projector:** Linear(1024->512) -> ReLU -> Linear(512->512), loaded from `projector.safetensors`
- **Scoring:** Listwise -- single forward pass for all query-doc pairs, cosine similarity between projected embeddings
- **Budget enforcement:** Truncates docs to 384 tokens, binary-searches prompt budget (120K tokens max)
- **Prompt format:** `<passage id="N">...doc<|embed_token|></passage>...<query>...query<|rerank_token|></query>`

### 3.7 Generator (`src/generator.py`)

`MlxGenerator` -- MLX-based text generation with streaming:

- **Model loading:** `mlx_lm.load()` with tokenizer special-token handling
- **KVCache:** Uses `make_prompt_cache` if available in mlx_lm, graceful degradation
- **Streaming:** `generate_chat_stream()` yields tokens with:
  - Incremental `<think>...</think>` block filtering
  - On-the-fly stop-token detection (regex pattern matching with boundary guards)
  - 500-char buffer limit to prevent unbounded accumulation
- **Thinking mode:** `stream_chat_with_thinking()` exposes the model's internal reasoning chain as `{"type": "thinking"}` / `{"type": "answer"}` dicts. Controlled via `enable_thinking` template parameter. Temperature/top_p floored to thinking-mode minimums (0.6 / 0.9).
- **Dual token budget:** Thinking mode uses `max_internal_tokens` (total thinking+answer cap) and `max_tokens` (visible answer cap). Zero-visible-token detection yields `THINKING_BUDGET_EXHAUSTED` error.
- **Sampling parameters:** All params (`temperature`, `top_p`, `top_k`, `min_p`, `presence_penalty`, `repetition_penalty`) flow through `GenerationConfig`. No separate kwargs on public methods.
- **Presence penalty:** Pure-MLX logits processor subtracts a fixed penalty from logits of unique previously-seen tokens (reduces repetitive phrasing without affecting new vocabulary).
- **Repetition penalty:** Native mlx-lm `make_logits_processors()` or fallback pure-MLX ops
- **Adaptive defaults:** Temperature/top_p/repetition_penalty tuned by model size (<30B, 30-70B, 70B+)
- **Token budget packing:** Greedy packing with early termination (3 consecutive failures), optional sentence-boundary truncation

### 3.8 Intent Classification (`src/intent.py`)

Two-stage classifier with heuristic-first design:

**Stage 1 -- Heuristic (regex + structural signals):**
- 8 intents: `OVERVIEW`, `SUMMARIZE`, `EXPLAIN`, `ANALYZE`, `COMPARE`, `CRITIQUE`, `FACTUAL`, `COLLECTION`
- Regex families include corpus/document inventory, factual extraction, analysis/comparison/critique, and summary/explain framing.
- Structural boosts include command verbs, comparative/extraction structures, corpus-scope detection, and technical how/why bias.
- **Why-question specificity override**: `why` queries with a detected named entity + action verb are promoted to `FACTUAL`.
- **Enumeration + entity-mention routing**: patterns like `list every/all`, `name all`, `what mentions of ...`, `find all references to ...`, and `where is ... discussed` are routed to `FACTUAL` for chunk-level retrieval.
- Typo tolerance via `difflib.get_close_matches(cutoff=0.82)`
- Confidence levels: 0.85 (strong, 2+ hits), 0.70 (single hit), 0.50 (weak), 0.40 (fallback)

**Stage 2 -- LLM fallback:**
- Triggers when heuristic confidence < 0.70
- Uses dedicated intent model (default: `mlx-community/LFM2-8B-A1B-4bit`)
- JSON output: `{"intent": "...", "confidence": 0.0-1.0}`
- Final gate: demotes to OVERVIEW if confidence < 0.60

`rag_engine.py` also applies a **source-selection guard**: with multiple sources loaded, queries such as “which document/source …” are forced to `COLLECTION` summary routing.

### 3.9 Prompt Construction (`src/generation.py`)

Intent-aware message builder producing chat-format messages:

- **System prompt:** Core rules (context-only, specific answers, paragraph formatting) + intent-specific instructions
- **12 instruction profiles:** Each defines `task`, `format`, `tone`. Profiles: SUMMARIZE, EXPLAIN, ANALYZE, COMPARE, CRITIQUE, FACTUAL, COLLECTION, EXTRACT, TIMELINE, HOW_TO, QUOTE_EVIDENCE, and OVERVIEW (fallback alias).
  - `EXTRACT` → numbered/bulleted list or table; exhaustive entity extraction; ends with "Total: N items found."
  - `TIMELINE` → `[DATE] — event (1-2 sentences)` format; `→` for causal links; gap note appended.
  - `HOW_TO` → numbered steps with imperative phrasing; sub-steps as indented bullets; closes with outcome sentence.
  - `QUOTE_EVIDENCE` → block-quote format (`> "exact quote"`) each followed by one relevance sentence; max 5 quotes.
- **Deep Research mode:** `INTENT_INSTRUCTIONS_DEEP_RESEARCH` diverges from `INTENT_INSTRUCTIONS_REGULAR` for thinking-enabled intents (ANALYZE, COMPARE, CRITIQUE, EXPLAIN): reasoning scaffolding is removed since the model's internal thinking phase handles it; visible output instructions describe the finished product. Non-thinking intents (FACTUAL, OVERVIEW, SUMMARIZE, COLLECTION, EXTRACT, TIMELINE, HOW_TO, QUOTE_EVIDENCE) get exhaustiveness additions to leverage deeper retrieval. Resolved via `_get_intent_instructions(mode)`.
- **Citation rules:** When enabled, adds mandatory `[N]` inline citation format; passes `PASSAGE N` labeling into context. (Always on via UI)
- **Source legend:** Maps source IDs to display names when they differ.
- **Context-sparsity warning:** Injects an explicit anti-hallucination warning when retrieved context fills <10% of retrieval budget.
- **Output:** `[{role: "system", content: ...}, {role: "user", content: context + question}]`

### 3.10 RAG Engine (`src/rag_engine.py`)

Central orchestrator consumed by both CLI and API:

- **`RagEngine`**: Lazy-loads models on first query (embedding, reranker, generator, intent classifier)
- **`query()`**: Synchronous path returning `QueryResult` (answer, intent, citations, metrics, latency)
- **`query_events()`**: Streaming path yielding `QueryEvent` objects (status, intent, sources, citations, tokens, errors, finish)
- **`enable_thinking` tri-state**: `None` (auto, intent-driven), `True` (user forced on), `False` (user forced off). RAM gating overrides to `False` when system RAM < 48GB.
- **Intent-gated thinking**: In deep-research mode, ANALYZE/COMPARE/CRITIQUE/EXPLAIN auto-enable thinking with Qwen3.5 official sampling (temp=1.0, top_p=0.95); other intents use non-thinking profiles (temp=0.3–0.7).
- **`ingest()`**: Ingests document, embeds chunks, optionally summarizes
- **Output sanitization:** `_sanitize_output()` strips instruction leakage ("Important:", "Task:", "Format:", "Tone:"), filler phrases, and deduplicates repeated halves (>85% similarity). Retrieval jargon normalization (chunk/vector-search references) was removed; the prompt now enforces this directly via system instructions.
- **Unified query pipeline:** `query()` and `query_events()` share a single `_query_pipeline()` implementation; the synchronous path collects events from the async generator rather than duplicating logic.
- **Query expansion removed:** Static BM25 query expansion terms (`_expand_query()`) have been eliminated. Retrieval operates directly on the original user query.
- **Memory-aware generation:** A constrained-RAM path adjusts context budget and generation parameters when system memory is critically low.
- **Offline mode:** `_enable_offline_if_cached()` sets `HF_HUB_OFFLINE=1` when all models are cached locally
- **Metal cache management:** Explicit `mx.clear_cache()` to release GPU memory between phases
- **Hallucination guardrail signal:** `_check_novel_proper_nouns()` logs warnings when generated output contains many proper nouns absent from retrieved context

### 3.11 Event System (`src/query_events.py`, `src/stream_protocol.py`)

**Query events** (frozen dataclasses):
- `StatusEvent`, `IntentEvent`, `SourcesEvent`, `TextTokenEvent`, `CitationListEvent`, `ErrorEvent`, `FinishEvent`

**Stream protocol** (AI SDK Data Stream Protocol v1):
- Type codes: `0` (text), `2` (data), `3` (error), `8` (annotation), `d` (finish_message), `e` (finish_step)
- Line format: `code:json_value\n`
- Headers: `Content-Type: text/plain; charset=utf-8`, `X-Vercel-AI-Data-Stream: v1`

### 3.12 API Layer (`src/api.py`)

FastAPI application with streaming support:

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/api/health` | GET | Engine status + system RAM (GB) |
| `/api/chat` | POST | Streaming chat (AI SDK text/plain format) |
| `/api/query` | POST | Query with optional SSE stream (`stream=true`) |
| `/api/sources` | GET | List ingested sources with metadata |
| `/api/sources/ingest` | POST | Ingest document by file path |
| `/api/sources/upload` | POST | Multipart file upload + ingest (50MB max, .pdf/.md) |
| `/api/sources/{id}` | DELETE | Delete source + all chunks + cache |
| `/api/sources/{id}/content` | GET | Full document text (original -> snapshot fallback) |
| `/api/sources/{id}/chunk/{cid}` | GET | Chunk detail for citation viewer |
| `/api/sources/{id}/chunks` | GET | Batch chunk fetch by comma-separated IDs (`?ids=a,b,c`) |
| `/api/transcribe` | POST | Audio transcription via MLX Whisper (returns `{"text": "..."}`) |
| `/api/freeform/chat` | POST | Freeform streaming chat (SSE, not AI SDK protocol) |

**Concurrency model:** Single-user lock (`chat_lock`) for exclusive access. Background thread executor runs `query_events()`, pushes events to asyncio.Queue via `loop.call_soon_threadsafe()`. Client disconnect detection via `should_stop` callback.

**File handling:** Uploads saved to `data/uploads/`, text snapshots cached in `data/source_cache/` for content fallback.

### 3.13 Source Cache (`src/source_cache.py`)

Caches full-text snapshots so `/content` endpoint works even if original files are moved/deleted:

- `save_snapshot()` -- writes text to `data/source_cache/{sanitized_id}.txt`
- `resolve_content()` -- resolution order: original file -> snapshot -> None
- `read_original_file()` -- handles both text and PDF (via PyMuPDF extraction)

### 3.14 CLI (`src/cli.py`)

Thin wrapper around `RagEngine` for terminal workflows:

- `ingest FILE --source-id ID [--summarize] [--mode MODE]`
- `query TEXT [--source-id ID] [--cite|--no-cite] [--mode MODE] [--latency] [--intent INTENT]`
- Prints: hardware config, retrieval metrics, answer with intent/confidence, optional latency breakdown

### 3.15 Transcription (`src/transcription.py`)

`WhisperTranscriber` -- MLX Whisper speech-to-text for voice input:

- **Model:** `mlx-community/whisper-large-v3-turbo` (or similar), loaded lazily on first use
- **Audio decode:** Handles WebM/WAV/MP3 via `ffmpeg` subprocess, resamples to 16kHz mono
- **VAD (Voice Activity Detection):** Energy-based silence trimming to avoid transcribing empty audio
- **API integration:** Exposed via `POST /api/transcribe` endpoint; returns `{"text": "..."}` JSON

### 3.16 Citation Verification (`src/citation_verification.py`)

`compute_highlight_texts` -- maps chunk citations to highlight-able text spans:

- **Sliding window:** Extracts representative text spans from cited chunks for document highlighting
- **Span finding:** Locates cited text within full document for precise highlight positioning
- **Frontend integration:** Results sent as part of `CitationListEvent` for the citation viewer modal

### 3.17 Metrics & Latency (`src/metrics.py`, `src/latency.py`)

**Retrieval metrics:** Budget utilization, per-stage timing, reranker score distributions, threshold/safety-net stats, deduplication reduction.

**Latency profiler:** Context-manager based wall-clock profiler (`with profiler.span("stage"):`), outputs ASCII bar chart breakdown. Zero-cost when disabled.

---

## 4. Frontend Architecture

### 4.1 Technology & Routing

- **Framework:** Next.js 16 App Router (single route: `/`)
- **Styling:** Tailwind CSS 4 with dark theme (`bg-gray-950`, blue accents)
- **State management:** React Context + useReducer (dual-context pattern: state + dispatch)
- **Streaming:** AsyncGenerator consuming SSE events from `/api/query`
- **Freeform chat mode:** Lightweight, session-persistent chat mode that preserves conversation history, unsent drafts, and per-session editable context; optimized for exploratory, back-and-forth browsing and quick note-style interactions.
- **Theme toggle (UI):** Small user-accessible toggle to switch visual themes.

### 4.2 Component Tree

```
Page (app/page.tsx)
  AppProvider (context/app-context.tsx)
    ThemeProvider (context/theme-context.tsx)
      Mode Tabs: RAG | Freeform
        (RAG mode)
          AssistantRuntimeProvider + Thread (assistant-ui)
            ChatMarkdown (citation-aware react-markdown)
            MessageReferences (cited/uncited indicators)
            ThinkingPanel (reasoning chain display)
            IntentSelector (intent override dropdown)
            ModelSelector (regular/deep-research)
          SourcePanel (30% width)
            Source list with select/expand/delete
            IngestModal (drag-drop upload, citation reference)
            CitationViewerModal (overlay)
              DocumentRenderer → MarkdownRenderer | PlainTextRenderer
        (Freeform mode)
          FreeformChatPanel (session-persistent chat)
            Thinking display (ReasoningRoot)
            Model selector + thinking toggle
      HistoryPanel (side drawer, session persistence)
      Theme backgrounds (WebGL: beams, meteors, rain, mesh, paths, starfield, particles)
```

### 4.3 State Shape (`AppState`)

```typescript
{
  statusMessage: string;     // Pipeline status from stream
  errorMessage: string;      // Error display
  isLockBusy: boolean;       // Server 429 status
  lastIntent: { intent, confidence, method } | null;
  lastSources: string[];     // Source IDs from latest query
  citationsByMessage: Map<string, CitationPayload[]>; // Per-message citation data
  activeCitation: { messageId, index } | null; // Currently viewing citation
  thinkingSteps: Map<string, ThinkingStep[]>; // Reasoning chain per message
  _stepCounter: number;      // Internal counter for thinking step IDs
  currentAssistantMessageId: string | null; // Active streaming message
  intentOverride: string | null; // User-forced intent
  chatMode: "rag" | "freeform"; // Active UI mode
}
```

### 4.4 API Client (`lib/api-client.ts`)

- `sourceApi` singleton -- typed methods for all source CRUD operations, including `getChunkBatch(sourceId, ids[])` for fetching multiple chunks in one call
- Direct backend URL for uploads (bypasses Next.js dev proxy ~30s timeout)

### 4.5 Citation Utilities (`lib/`)

- **`citation-meta-store.ts`:** localStorage persistence for user-provided citation reference strings (keyed by `source_id`). Used by the copy-as-citation feature. Populated at ingest time via an optional text field in `IngestModal`; cleared on source deletion.
- **`format-citations.ts`:** Formats a citation into footnote style or Harvard style from stored metadata + page numbers. Falls back to filename stem if no reference string was saved.
- **`group-citations.ts`:** Groups raw `[N]` citation indices from a response into deduped, ordered reference lists for the `MessageReferences` component.

### 4.6 Key Frontend Patterns

- **Sequential upload queue:** Processes uploads one at a time to avoid timeouts, with per-file status
- **Citation linking:** `ChatMarkdown` converts `[CHUNK N]` / `[N]` patterns to clickable citation links
- **Fuzzy text highlighting:** `text-highlighter.ts` uses progressive degradation (full -> prefix -> suffix matching) to find and `<mark>` cited text in documents
- **50ms streaming throttle:** Token accumulation in ref, batched state updates via setTimeout
- **AbortController:** Stop button cancels in-flight query stream
- **Speech-to-text:** `useSpeechToText` hook records audio via MediaRecorder, sends to `/api/transcribe` for MLX Whisper transcription, inserts result into input
- **Conversation history:** `session-store.ts` persists chat sessions to IndexedDB with title, messages, mode, and timestamps. `HistoryPanel` drawer allows browsing, searching (debounced), and restoring previous sessions.
- **Theme system:** `theme-context.tsx` manages background theme selection (none, beams, meteors, rain, mesh gradient, paths, starfield, particles). WebGL backgrounds rendered via `@react-three/fiber` (`components/gl/`). CSS-only fallbacks for simpler effects.
- **Dual-mode UI:** Tab switching between RAG mode (assistant-ui Thread with retrieval pipeline) and Freeform mode (direct LLM chat with session persistence). Mode persisted per session.

---

## 5. Configuration & Modes

### 5.1 Operating Modes

Current active modes (`VALID_MODES`) with RAM-aware scaling in `regular`:

| Mode | LLM Model | Context Window | Retrieval Budget | RAM |
|------|-----------|----------------|------------------|-----|
| `regular` (32-47GB) | Qwen3.5-35B-A3B-4bit | 16K | 8K | 32GB+ |
| `regular` (48-63GB) | Qwen3.5-35B-A3B-4bit | 64K | 32K | 48GB+ |
| `regular` (64GB+) | Qwen3.5-35B-A3B-4bit | 64K | 48K | 64GB+ |
| `deep-research` (48-63GB) | Qwen3.5-35B-A3B-4bit | 64K | 40K | 48GB+ |
| `deep-research` (64GB+) | Qwen3.5-35B-A3B-4bit | 64K | 40K | 64GB+ |

All modes use the same model (`NexVeridian/Qwen3.5-35B-A3B-Instruct-2507-4bit`). Deep-research mode differentiates through deeper retrieval, intent-gated thinking, and divergent prompt instructions.

All modes share: Qwen3-Embedding-0.6B-4bit-DWQ (embeddings) + jinaai/jina-reranker-v3-mlx (reranker).

### 5.2 Mode Selection Precedence

1. CLI flag: `--mode regular|deep-research`
2. Environment variable: `RAG_MODE=<mode>`
3. Auto-detection: defaults to `regular` with RAM-based parameter scaling

Legacy mode names (`high`, `efficiency`, `power-fast`, `power-deep-research`, etc.) are mapped to current modes with deprecation warning.

### 5.3 Retrieval Parameters by Mode

| Parameter | Regular (32-47GB) | Regular (48-63GB) | Regular (64GB+) | Deep-Research |
|-----------|-------------------|-------------------|-----------------|---------------|
| `top_k_dense` | 100 | 100 | 200 | 400 |
| `top_k_fused` | 50 | 50 | 100 | 200 |
| `top_k_rerank` | 30 | 30 | 55 | 80 |
| `top_k_final` | 5 | 5 | 8 | 8 |
| `reranker_threshold` | 0.05 | 0.05 | 0.04 | 0.04 |
| `reranker_min_docs` | 3 | 3 | 4 | 4 |

### 5.4 Sampling Profiles

| Profile | Temperature | top_p | top_k | min_p | presence_penalty |
|---------|-------------|-------|-------|-------|------------------|
| Regular (non-thinking) | 0.3–0.7 | 0.6–0.8 | 20 | 0.0 | 0.0 |
| Deep-research (non-thinking) | 0.3–0.7 | 0.6–0.8 | 20 | 0.0 | 1.5 |
| Deep-research (thinking) | 1.0 | 0.95 | 20 | 0.0 | 1.5 |

Thinking-mode profiles follow Qwen3.5 official recommendations. The model template controls thinking via `enable_thinking=True/False`.

### 5.5 Dual Token Budgets (Thinking Mode)

| RAM Tier | Internal Cap (thinking+answer) | Visible Cap (answer only) |
|----------|-------------------------------|---------------------------|
| 48–63GB | 16,384 | 2,048 |
| 64GB+ | 20,480 | 4,096 |

If the model consumes the entire internal budget on thinking without producing visible tokens, a `THINKING_BUDGET_EXHAUSTED` error is returned.

**Note:** `FACTUAL` intent overrides `reranker_min_docs` to **1**. This prevents below-threshold guard rails from pulling in loosely related chunks when the query is a narrow factual lookup. Below-threshold expansion is also scoped to sources that already have at least one above-threshold hit, preventing expansion from sources with zero relevant material.
| Reranker enabled | Yes | Yes | Yes | Yes |
| Context expansion | Yes | Yes | Yes | Yes |

### 5.6 System Detection

RAM detection cascade: `psutil` -> macOS `sysctl -n hw.memsize` -> Linux `os.sysconf`. Result cached globally.

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

The system builds a source legend when source IDs differ from display names.

### Streaming Citation Events

During streaming query execution, a structured citation list is emitted as a `CitationListEvent` so the frontend can bind numbered references to source/chunk metadata. The frontend `ChatMarkdown` component converts `[CHUNK N]` and `[N]` patterns into clickable citation links that open the `CitationViewerModal` with fuzzy text highlighting.

### Citation Reference Metadata

The `IngestModal` exposes an optional "Citation reference" text field. Whatever the user enters is persisted in `localStorage` via `citation-meta-store.ts` (key: `dh-citation-meta-v1`). This is used by `format-citations.ts` to produce properly formatted footnote or Harvard-style citations when the user clicks "Copy as citation" in `MessageReferences`. If no reference string was stored, the formatter falls back to the source filename stem + page numbers.

### MessageReferences Component

Replaces the earlier inline reference display with a dedicated `MessageReferences` component rendered below each assistant message:
- Shows all sources cited in the response with a **cited** (green) vs **uncited** (dimmed) visual indicator
- Clicking a reference opens the `CitationViewerModal` at the relevant passage
- "Copy as citation" button triggers the formatter for footnote or Harvard output

---

## 7. Data Flows

### Query Flow (Streaming)

```
Frontend POST /api/query {stream:true}
  |
  v
api.py: background thread calls rag_engine.query_events()
  |
  +--> IntentClassifier.classify(query) -> IntentResult
  |      yields: StatusEvent, IntentEvent
  |
  +--> RetrievalEngine.search(query) -> List[RetrievalResult]
  |      yields: StatusEvent, SourcesEvent
  |
  +--> generation.build_messages(context, question, intent)
  |
  +--> MlxGenerator.generate_chat_stream(messages)
  |      yields: TextTokenEvent (per token)
  |
  +--> Citation assembly
  |      yields: CitationListEvent
  |
  +--> Output sanitization + finish
         yields: FinishEvent
  |
  v
api.py: encode events -> SSE lines -> StreamingResponse
  |
  v
Frontend: queryStreaming() AsyncGenerator -> ChatPanel state updates
```

### Ingestion Flow

```
Frontend POST /api/sources/upload (multipart)
  |
  v
api.py: save to data/uploads/, call rag_engine.ingest()
  |
  v
ingest.py: parse file (PDF fallback chain or markdown sections)
  -> _chunk_pages() -> (parents, children)
  |
  v
embeddings.encode(child_texts) -> vectors
  |
  v
storage.add_parents(parents) + storage.add_children(children, embeddings)
  |
  v
Optional: generator.generate_chat(summary prompt)
  -> storage.upsert_source_summary()
  |
  v
source_cache.save_snapshot() -> data/source_cache/
```

---

## 8. File Structure

### Backend (`src/`)

| File | Lines | Purpose |
|------|-------|---------|
| `api.py` | ~1310 | FastAPI endpoints, SSE streaming, file upload, freeform chat |
| `api_schemas.py` | ~320 | Pydantic request/response models |
| `rag_engine.py` | ~1690 | Core orchestrator, query/ingest, output sanitization |
| `retrieval.py` | ~680 | 7-stage hybrid search pipeline |
| `storage.py` | ~655 | LanceDB tables, hybrid search, FTS management |
| `generator.py` | ~930 | MLX text generation, streaming, token budget packing |
| `reranker.py` | ~425 | Jina v3 MLX listwise reranker |
| `intent.py` | ~965 | Heuristic + LLM intent classification (12 intents) |
| `generation.py` | ~560 | Intent-aware prompt/message building (12 profiles) |
| `embeddings.py` | ~155 | MLX embedding model wrapper |
| `config.py` | ~345 | Mode configs, RAM detection, auto-scaling |
| `ingest.py` | ~530 | PDF/Markdown parsing, chunking |
| `models.py` | ~35 | ParentChunk, ChildChunk, Metadata |
| `stream_protocol.py` | ~250 | AI SDK Data Stream Protocol encoder |
| `query_events.py` | ~96 | Streaming event dataclasses |
| `metrics.py` | ~170 | Retrieval/budget metrics collection |
| `latency.py` | ~107 | Wall-clock latency profiler |
| `source_cache.py` | ~175 | Text snapshot caching |
| `citation_verification.py` | ~352 | Citation highlight text computation |
| `transcription.py` | ~387 | MLX Whisper speech-to-text |
| `cli.py` | ~383 | CLI entry point |

### Frontend (`frontend/src/`)

| File | Purpose |
|------|---------|
| `app/page.tsx` | Entry point, dual-mode layout (RAG/Freeform), theme picker |
| `app/layout.tsx` | Root layout, metadata, fonts (Geist) |
| `app/providers.tsx` | AssistantRuntimeProvider + AI SDK integration |
| `app/api/chat/route.ts` | Next.js API route proxying to backend `/api/chat` |
| `app/globals.css` | Theme colors, scrollbar styling, markdown prose |
| `context/app-context.tsx` | Global state (useReducer + dual context) |
| `context/theme-context.tsx` | Theme state (WebGL background selection) |
| `lib/api-client.ts` | Typed API client for source CRUD |
| `lib/backend-url.ts` | Backend URL resolution |
| `lib/event-parser.ts` | SSE event parsing utilities |
| `lib/session-store.ts` | IndexedDB session persistence (history) |
| `lib/text-highlighter.ts` | Fuzzy text matching, DOM highlight injection |
| `lib/citation-meta-store.ts` | localStorage persistence for citation references |
| `lib/format-citations.ts` | Footnote + Harvard citation formatters |
| `lib/group-citations.ts` | Citation index grouping and deduplication |
| `lib/utils.ts` | Shared utilities (cn, etc.) |
| `components/source-panel.tsx` | Source management, upload queue, view content |
| `components/freeform-chat-panel.tsx` | Freeform streaming chat with session persistence |
| `components/history-panel.tsx` | Chat history drawer with session restore |
| `components/citation-viewer-modal.tsx` | Full document viewer with citation highlight |
| `components/ingest-modal.tsx` | Drag-drop upload with optional citation reference |
| `components/chat-markdown.tsx` | Citation-aware markdown rendering |
| `components/markdown-renderer.tsx` | Markdown display with section scroll |
| `components/document-renderer.tsx` | Format router (markdown vs plain text) |
| `components/assistant-ui/thread.tsx` | assistant-ui Thread customization |
| `components/assistant-ui/thinking-panel.tsx` | Reasoning chain display panel |
| `components/assistant-ui/intent-selector.tsx` | Explicit intent override dropdown |
| `components/assistant-ui/message-references.tsx` | Per-message citation reference list |
| `components/assistant-ui/model-selector.tsx` | Regular/Deep Research model picker |
| `components/assistant-ui/reasoning.tsx` | Reasoning block components |
| `components/assistant-ui/markdown-text.tsx` | Markdown text rendering for assistant-ui |
| `components/assistant-ui/chat-markdown-renderer.tsx` | Chat-specific markdown rendering |
| `components/assistant-ui/file.tsx` | File display components |
| `components/assistant-ui/attachment.tsx` | Attachment handling |
| `components/assistant-ui/tooltip-icon-button.tsx` | Icon button with tooltip |
| `components/assistant-ui/tool-fallback.tsx` | Tool call fallback display |
| `components/ui/*.tsx` | Shared UI primitives (button, dialog, picker, checkbox, etc.) |
| `components/gl/index.tsx` | WebGL background scene root |
| `components/gl/particles.tsx` | GPU particle simulation |
| `components/gl/shaders/*.ts` | GLSL shaders for particle/vignette effects |
| `components/ui/beams.tsx` | Animated beam background |
| `components/ui/meteors.tsx` | Meteor shower background |
| `components/ui/rain.tsx` | Rain effect background |
| `components/ui/mesh-gradient.tsx` | Animated mesh gradient background |
| `components/ui/paths.tsx` | Animated path background |
| `components/ui/starfield.tsx` | Starfield background |
| `components/ui/particles.tsx` | CSS particle background |
| `hooks/useSpeechToText.ts` | Speech-to-text hook (Whisper integration) |
| `hooks/useSystemRam.ts` | System RAM detection hook |

---

## 9. Usage Examples

```bash
# CLI query (auto mode)
python -m src.cli query "What is the main argument?"

# Regular mode + citations
python -m src.cli query "What docs are in here" --mode regular --cite

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

## 10. Dependencies

### Backend (Required)

- **Storage:** `lancedb`, `pyarrow`, `pydantic`
- **ML:** `mlx`, `mlx-lm`, `safetensors`, `numpy`
- **Tokenizer:** `transformers`, `huggingface-hub`
- **PDF:** `pypdf`, `pdfminer.six`, `PyMuPDF` (fitz), `pytesseract`, `pdf2image`
- **API:** `fastapi`, `uvicorn`, `python-multipart`

### Backend (Test-only)

- `httpx` -- async HTTP client for FastAPI TestClient

### Backend (Optional)

- `psutil` -- robust cross-platform RAM detection

### Frontend

- `next` 16, `react` 19, `react-dom` 19
- `@assistant-ui/react`, `@assistant-ui/react-ai-sdk` -- chat UI framework + AI SDK integration
- `react-markdown`, `remark-gfm`
- `tailwindcss` 4, `@tailwindcss/postcss`
- `@react-three/fiber`, `@react-three/drei`, `three` -- WebGL backgrounds
- `typescript` 5, `eslint` 9
