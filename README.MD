<div align="center">

# Corpus

### A private, offline research assistant for humanities scholarship

*Ask questions of your documents. Receive grounded, cited answers. No data leaves your machine.*

![Corpus demo screenshot](docs/images/demo-screenshot.png)

[![Python 3.11+](https://img.shields.io/badge/Python-3.11+-3776AB?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![Next.js 16](https://img.shields.io/badge/Next.js-16-000000?style=flat-square&logo=nextdotjs&logoColor=white)](https://nextjs.org)
[![Apple Silicon](https://img.shields.io/badge/Apple_Silicon-Optimised-555555?style=flat-square&logo=apple&logoColor=white)](https://apple.com/silicon)
[![MLX](https://img.shields.io/badge/Powered_by-MLX-FF6B35?style=flat-square)](https://github.com/ml-explore/mlx)
[![Offline First](https://img.shields.io/badge/Offline-First-22c55e?style=flat-square)](https://github.com)
[![No Cloud](https://img.shields.io/badge/No_Cloud-No_Subscription-6366f1?style=flat-square)](https://github.com)

</div>

---

## Overview

Humanities researchers increasingly work with AI tools that require uploading documents to external servers and accepting ongoing cloud dependency. This creates a genuine problem for anyone working with sensitive archival materials, unpublished manuscripts, or embargoed sources.

Corpus runs entirely on your machine. Documents are ingested into a local vector database, queries are answered by a large language model running on your Apple Silicon GPU, and every response is grounded in retrieved passages with inline citations linking to the specific source location and page number. No data leaves your machine, and no internet connection is required after the initial model download.

*Developed as part of a final-year University College Cork Digital Humanities project.*

---

## Capabilities

| Capability | Description |
|---|---|
| **Hybrid retrieval** | Vector ANN semantic search combined with BM25 full-text search, fused via Reciprocal Rank Fusion |
| **Neural reranking** | Jina v3 listwise reranker rescores retrieved passages using a dedicated on-device MLX model |
| **Intent-aware querying** | Twelve research intents with per-intent retrieval depth, prompt instructions, and output format |
| **Inline citations** | Every claim is annotated with passage and page references, linked to a document viewer |
| **Deep Research mode** | Activates chain-of-thought reasoning for analytical intents on systems with 48 GB+ RAM |
| **Offline transcription** | Voice input transcribed locally via Whisper large-v3, with voice activity detection |
| **Freeform chat** | Direct LLM access without retrieval, with persistent session history |
| **Citation export** | Responses exportable as formatted footnotes or Harvard bibliography entries |
| **Geospatial mapping** | Optional ingest-time place NER + offline geocoding, with clustered map markers filtered by source selection |
| **People dictionary** | Optional ingest-time person NER + canonical name matching, with searchable mentions, merge tools, and document jump-to-context |
| **Ingest NER diagnostics** | Ingest responses report whether GLiNER ran or a fallback path was used (`gliner`, `regex_fallback`, `empty`) so degraded indexing is visible |
| **Phoenix tracing** | Optional OpenTelemetry tracing with trace/span ID propagation and in-UI feedback annotations |

---

## Research Intent System

Corpus classifies each query into one of twelve research intents before executing the retrieval pipeline. The classifier uses a heuristic-first approach (regex patterns, structural signals, typo tolerance via difflib) with an LLM fallback when heuristic confidence falls below 0.70. The resolved intent determines retrieval parameters, prompt instructions, and output format. It can be overridden manually from the composer toolbar.

| Intent | Typical triggers | Output format |
|---|---|---|
| `OVERVIEW` | "What is this document about?" | Prose overview |
| `SUMMARIZE` | "Summarise...", "Give me a summary of..." | Structured summary |
| `EXPLAIN` | "Explain...", "What does X mean here?" | Explanation with examples |
| `ANALYZE` | "Analyse...", "What are the implications of..." | Analytical prose |
| `COMPARE` | "Compare...", "How does X differ from Y..." | Structured comparison |
| `CRITIQUE` | "Critique...", "What are the weaknesses of..." | Critical evaluation |
| `FACTUAL` | Specific who / what / when / where questions | Direct factual answer |
| `COLLECTION` | "Which documents cover...", "What sources mention..." | Grouped by document |
| `EXTRACT` | "List every...", "Find all mentions of..." | Numbered list or table |
| `TIMELINE` | "When did...", "Trace the history of..." | `[DATE] - event` format |
| `HOW_TO` | "How do I...", "What are the steps to..." | Numbered procedure |
| `QUOTE_EVIDENCE` | "What does X say about...", "Find quotes on..." | Block-quoted passages |

---

## Architecture

```mermaid
graph TB
    subgraph Frontend["Frontend — Next.js 16 / React 19"]
        UI["Chat Interface<br/>RAG + Freeform modes"]
        SP["Source Panel<br/>Document library"]
        CV["Citation Viewer<br/>Passage drill-down"]
        PD["People Dictionary<br/>Canonical entity browser"]
        HI["Session History<br/>IndexedDB"]
    end

    subgraph Backend["Backend — FastAPI / Python"]
        API["REST + SSE API<br/>/api/chat · /api/query<br/>/api/sources · /api/geo · /api/people"]
        RAG["RAG Engine<br/>Central orchestrator"]
        IC["Intent Classifier<br/>12 research intents"]
        RT["Retrieval Engine<br/>7-stage pipeline"]
        NER["NER Layer<br/>Shared city + person extraction"]
        GCD["Offline Geocoder<br/>GeoNames + RapidFuzz"]
        PR["Person Resolver<br/>Canonical name matching"]
        GEN["Generator<br/>Streaming LLM"]
    end

    subgraph Models["On-Device Models — MLX / Metal"]
        LLM["Qwen3.5-35B MoE<br/>Language model"]
        EMB["Qwen3-Embedding-0.6B<br/>Semantic encoding"]
        RNK["Jina Reranker v3<br/>Neural reranking"]
        WSP["Whisper large-v3<br/>Speech to text"]
        INT["LFM2-8B MoE<br/>Intent fallback"]
    end

    subgraph Storage["Local Storage — LanceDB"]
        VEC["Child Chunks<br/>Vector + BM25 index"]
        PAR["Parent Chunks<br/>Full context blocks"]
        SUM["Source Summaries<br/>Document metadata"]
        GEO["Geo Mentions<br/>Map entities"]
        PPL["Person Mentions<br/>People dictionary entities"]
    end

    UI <-->|"AI SDK v6 / SSE"| API
    SP & CV & PD & HI --> API
    API --> RAG
    RAG --> IC & RT & GEN & NER
    NER --> GCD & PR
    RT <--> EMB & RNK
    RT <--> VEC & PAR & SUM
    GCD --> GEO
    PR --> PPL
    GEN <--> LLM
    IC -.->|"low-confidence fallback"| INT
```

### Query Pipeline

```mermaid
flowchart TD
    Q(["Query"])

    IC["Stage 1: Intent Classification<br/>Heuristic classifier: regex and structural signals<br/>LFM2-8B LLM fallback when confidence below 0.70<br/>Resolves to one of 12 research intents"]

    HS["Stage 2: Hybrid Search<br/>Vector ANN: semantic similarity via Qwen3 embeddings<br/>BM25 full-text: keyword and phrase matching<br/>RRF score fusion combines both ranked lists"]

    RR["Stage 3: Neural Reranking<br/>Jina v3 listwise reranker: single forward pass<br/>Cosine similarity scoring across all candidates<br/>Replaces fusion score with grounded relevance"]

    FI["Stage 4: Filtering and Deduplication<br/>Adaptive threshold: max score scaled at 15%<br/>Parent deduplication: max 2 children per parent<br/>Safety net: guaranteed minimum document count"]

    CE["Stage 5: Context Expansion<br/>Full parent chunk text fetched for each result<br/>Budget-aware fill of available token window<br/>Intent-specific sub-threshold backfill if starved"]

    GA["Stage 6: Answer Generation<br/>Intent-aware prompt assembly: 12 instruction profiles<br/>Qwen3.5-35B streaming generation<br/>Optional chain-of-thought in Deep Research mode"]

    CS["Stage 7: Citation Assembly and Sanitisation<br/>Inline PASSAGE references bound to source metadata<br/>Instruction leakage and chatter phrase removal<br/>Hallucination check: novel proper nouns flagged"]

    A(["Cited answer, streamed to browser"])

    Q --> IC --> HS --> RR --> FI --> CE --> GA --> CS --> A

    style Q fill:#4f46e5,color:#fff,stroke:none
    style A fill:#059669,color:#fff,stroke:none
```

### Document Ingestion

```mermaid
flowchart LR
    DOC(["Document<br/>.pdf or .md"])

    subgraph Parse["Parse"]
        direction TB
        P1["PyMuPDF<br/>Primary extraction"]
        P2["pypdf<br/>Fallback"]
        P3["pdfminer.six<br/>Per-page fallback"]
        P4["OCR<br/>Rasterise + pytesseract<br/>for scanned pages"]
        P1 --> P2
        P2 --> P3
        P3 --> P4
    end

    subgraph Chunk["Chunk"]
        PC["Parent chunks<br/>~1,200 tokens<br/>Full paragraph context"]
        CC["Child chunks<br/>~250 tokens<br/>Precise retrieval units"]
        PC -->|"subdivided into"| CC
    end

    subgraph Index["Index"]
        EMB2["Embed child chunks<br/>Qwen3-Embedding-0.6B"]
        STORE["Write to LanceDB<br/>Vector index + BM25 FTS"]
        EMB2 --> STORE
    end

    subgraph Opt["Optional Enrichment"]
        NERX["Shared GLiNER pass<br/>city + person candidates"]
        GEOX["Place resolution<br/>GeoNames geocoder<br/>persist geo_mentions"]
        PPLX["Person canonicalization<br/>resolver + fuzzy matching<br/>persist person_mentions"]
        SUMM["LLM summary<br/>3-4 paragraph overview<br/>stored per source"]
        NERX --> GEOX
        NERX --> PPLX
    end

    DOC --> Parse --> Chunk --> Index
    Chunk -.->|"if geotag / peopletag enabled"| NERX
    Chunk -.->|"if summarise enabled"| Opt

    style DOC fill:#7c3aed,color:#fff,stroke:none
```

Child chunks (~250 tokens) serve as retrieval units because their specificity improves vector search precision. When a child chunk is retrieved, Corpus fetches the full parent chunk (~1,200 tokens) to pass to the language model, providing the paragraph-level context needed for coherent, accurate generation.


### Operating Modes

```mermaid
graph LR
    RAM{{"System RAM"}}

    RAM -->|"32-47 GB"| REG
    RAM -->|"48 GB +"| DR

    subgraph REG["Regular Mode"]
        R1["16K token context window"]
        R2["8K token retrieval budget"]
        R3["Standard retrieval depth"]
        R4["Thinking mode unavailable"]
    end

    subgraph DR["Deep Research Mode"]
        D1["64K token context window"]
        D2["40K token retrieval budget"]
        D3["4× more retrieval candidates"]
        D4["Chain-of-thought reasoning<br/>ANALYZE · COMPARE · CRITIQUE · EXPLAIN"]
    end
```

Mode is detected automatically from system RAM. In Deep Research mode the four analytical intents (ANALYZE, COMPARE, CRITIQUE, EXPLAIN) activate Qwen3.5's chain-of-thought reasoning; the thinking is hidden from the visible output but substantially improves quality on complex questions.

---

## Interface

The application presents two independently persistent chat modes plus two optional research overlays (Map and People Dictionary). Both chat panels remain mounted simultaneously; visibility is toggled via CSS so neither loses state on mode switch.

### RAG Mode

The primary research interface. Queries are processed through the full retrieval pipeline and responses stream token-by-token with inline citation markers ([1], [2], etc.) linking to the specific passage in the source document.

- **Source panel** — lists all ingested documents, supports drag-and-drop upload, and displays optional AI-generated summaries
- **Citation viewer** — opens in place of the source panel when a citation is clicked, fetching the full document and highlighting the cited passage
- **Thinking panel** — displays pipeline events in real time (intent classified, sources retrieved, generation progress) and, in Deep Research mode, the model's reasoning chain
- **Composer** — supports intent override, Deep Research toggle, thinking mode toggle (48 GB+ systems), and offline voice input

### People Dictionary Overlay

An optional panel for person-centric exploration across selected sources.

- **Canonical entries** — variants like titles, surnames, and full-name mentions are grouped under a single canonical person entry
- **Source-aware filtering** — results follow the same source selection as RAG and Map workflows
- **Mention operations** — inspect mentions by source/location, jump to cited document context, delete noisy mentions, and merge canonical names
- **Fast browsing** — frequency/alphabetical sort modes, debounced search, and cached mention expansion

### Freeform Mode

A direct conversational interface to the LLM with no retrieval pipeline. Conversation history is maintained within the session and persisted to IndexedDB. Supports thinking mode and generates AI-derived session titles after the first exchange.

![Freeform mode demo](docs/images/freemode-demo.png)

### Background Themes

Seven visual themes: `stars`, `meteors`, `rain`, `mesh`, `starfield`, `particles`, and `darkveil`.

---

## Citation System

When citations are enabled, each retrieved passage is wrapped at prompt-construction time:

```
[PASSAGE N | SOURCE: source_id | PAGE: display_page]
... chunk text ...
[PASSAGE END]
```

The model is instructed to cite passages inline as `[N]`. After generation, a `CitationListEvent` is emitted over SSE carrying the full citation metadata; the frontend binds each marker to a Citation object with source ID, chunk ID, and page number.

Duplicate source-page citations are collapsed and renumbered after generation so citation markers remain sequential and export-safe.

Clicking a citation badge opens the citation viewer: the full document and the cited chunk are fetched in parallel, and the passage is highlighted in context. An optional citation reference string can be entered per source at ingest time, persisted in backend source metadata, and used when exporting responses as formatted footnotes or Harvard bibliography entries.

![Citation viewer demo](docs/images/citation-demo.png)

---

## Geospatial Mapping

Corpus includes an optional geospatial workflow for place-centric research:

- **Ingest-time geotagging** via the Ingest modal (`Index locations`) or CLI `--geotag`
- **Offline place resolution** using local GeoNames data (no external geocoding API)
- **Interactive map overlay** with clustering, confidence threshold filtering, and source-selection filtering
- **Citation bridge** from map mention to source chunk viewer, plus inline mention deletion for cleanup


![Mapping demo screenshot](docs/images/map-demo.png)

---

## People Dictionary

Corpus includes an optional person-centric workflow for prosopographical and network-oriented research:

- **Ingest-time people indexing** via the Ingest modal (`Index people names`) or CLI `--peopletag`
- **Canonical name resolution** using title stripping + conservative fuzzy matching to group mention variants
- **Interactive People overlay** with search, sort, mention grouping, and source-aware filtering
- **Merge and cleanup tools** for canonical entries and mention-level corrections
- **Citation bridge** from person mention to highlighted source chunk in the document viewer

![People Dictionary demo screenshot](docs/images/people-demo.png)

---

## Privacy and Local Data

All inference and storage is local. No data is transmitted to external services after the initial model download. The application sets `HF_HUB_OFFLINE=1` automatically once all required models are detected in the local cache.

```
data/
    lance/
        child_chunks/           Vector + BM25 index (retrieval units)
        parent_chunks/          Full context blocks for answer generation
        source_summaries/       Per-document metadata and optional summaries
        geo_mentions/           Geocoded place mentions for map visualisation
        person_mentions/        Canonicalized person mentions for People Dictionary
    cities500.txt              GeoNames place index used by offline geocoder
    geo_version.json           GeoNames checksum/build metadata
    source_cache/               Full-text snapshots for the citation viewer
    uploads/                    Uploaded files retained after ingestion
```

---

## Technology Stack

<details>
<summary><strong>Backend</strong></summary>

| Component | Technology | Role |
|---|---|---|
| Language model | mlx-lm / Qwen3.5-35B-A3B (4-bit MoE) | Answer generation, chain-of-thought reasoning |
| Embeddings | mlx-lm / Qwen3-Embedding-0.6B (4-bit DWQ) | Semantic document and query encoding |
| Reranker | jina-reranker-v3-mlx / Qwen3-0.6B + MLP projector | Listwise neural passage reranking |
| Intent classifier | mlx-lm / LFM2-8B-A1B (4-bit MoE) | Intent classification fallback |
| Speech-to-text | mlx-whisper / Whisper large-v3 (4-bit) | Offline audio transcription with VAD |
| Vector database | lancedb + pyarrow | Vector ANN + BM25 FTS with RRF fusion |
| API framework | fastapi + uvicorn | REST endpoints and SSE streaming |
| PDF parsing | PyMuPDF, pypdf, pdfminer.six, pytesseract | Quality-first per-page fallback chain including OCR |
| Geospatial + people stack | GeoNames + rapidfuzz + GLiNER + PersonResolver | Offline place geocoding and person canonicalization for map + people dictionary |
| Observability | arize-phoenix-otel | OpenTelemetry tracing across ingest, retrieval, and generation |

</details>

<details>
<summary><strong>Frontend</strong></summary>

| Component | Technology | Role |
|---|---|---|
| Framework | Next.js 16 (App Router), React 19, TypeScript 5 | Application shell and routing |
| Chat UI | @assistant-ui/react, AI SDK v6 | Streaming chat with UI Message Stream protocol |
| Markdown | @assistant-ui/react-streamdown, react-markdown | Smooth-streaming markdown with citation rendering |
| Styling | Tailwind CSS 4 | Dark monochrome theme |
| 3D / visual | @react-three/fiber, Three.js | WebGL background themes and particle systems |
| Animations | framer-motion | UI transitions |
| Session storage | IndexedDB | Persistent chat history with reasoning content |
| Citation storage | LanceDB source metadata | User-provided citation reference strings per source |
| Geospatial UI | maplibre-gl, @vis.gl/react-maplibre, @tanstack/react-query | Clustered map overlay with confidence and source filters |

</details>

<details>
<summary><strong>On-device models (MLX / Apple Metal)</strong></summary>

All models are 4-bit quantised. After the initial download, no internet connection is required.

| Model | Parameters | Approx. size | Role |
|---|---|---|---|
| NexVeridian/Qwen3.5-35B-A3B-4bit | 35B total / 3B active | ~20 GB | Primary language model |
| mlx-community/Qwen3-Embedding-0.6B-4bit-DWQ | 0.6B | ~350 MB | Document and query embeddings |
| jinaai/jina-reranker-v3-mlx | 0.6B backbone | ~350 MB | Neural passage reranking |
| mlx-community/LFM2-8B-A1B-4bit | 8B total / 1B active | ~4.5 GB | Intent classification fallback |
| mlx-community/whisper-large-v3-4bit | Large-v3 | ~800 MB | Speech-to-text transcription |

</details>

## Installation

> **System requirements:** Apple Silicon Mac (M1 or later), macOS 13+, minimum 32 GB unified memory (48 GB recommended for Deep Research mode and thinking features). Python 3.11+ and Node.js 18+ are required.

### 1. Clone the repository

```bash
git clone https://github.com/mc32452/dh-notebook-offline.git
cd dh-notebook-offline
```

### 2. Set up the Python environment

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 3. Install frontend dependencies

```bash
cd frontend
npm install
cd ..
```

### 4. Download AI models

Models are downloaded automatically from Hugging Face on first use. You can also pre-download them all at once with the commands below, which is recommended so startup does not stall on a slow connection:

```bash
source .venv/bin/activate

# Primary language model (~20 GB)
python -c "from mlx_lm import load; load('NexVeridian/Qwen3.5-35B-A3B-4bit')"

# Embedding model (~350 MB)
python -c "from mlx_lm import load; load('mlx-community/Qwen3-Embedding-0.6B-4bit-DWQ')"

# Intent classifier (~4.5 GB)
python -c "from mlx_lm import load; load('mlx-community/LFM2-8B-A1B-4bit')"

# Reranker (~350 MB)
python -c "from mlx_lm import load; load('jinaai/jina-reranker-v3-mlx')"

# Whisper speech recognition (~800 MB)
python -c "import mlx_whisper; print('Whisper ready')"
```

> **Total download: approximately 26 GB**, of which the primary language model (~20 GB) accounts for the bulk. Models are cached in `~/.cache/huggingface/` — no further internet access is needed after this one-time download.


---

## Usage

### Starting the application

```bash
# Start the backend API server
source .venv/bin/activate
uvicorn src.api:app --host 127.0.0.1 --port 8000

# In a separate terminal, start the frontend
cd frontend && npm run dev
```

Open http://localhost:3000 in your browser.

### CLI

```bash
# Ingest a document
python -m src.cli ingest docs/paper.pdf --source-id my_paper --summarize

# Ingest + geotag places for map exploration
python -m src.cli ingest docs/paper.pdf --source-id my_paper --summarize --geotag

# Ingest + index people names for People Dictionary
python -m src.cli ingest docs/paper.pdf --source-id my_paper --summarize --peopletag

# Ingest + geotag + people indexing in one pass
python -m src.cli ingest docs/paper.pdf --source-id my_paper --summarize --geotag --peopletag

# Query with automatic intent detection
python -m src.cli query "What is the central argument?" --cite

# Query with a specific intent in Deep Research mode
python -m src.cli query "Compare the two theoretical frameworks" \
  --intent compare --mode deep-research

# Inspect the exact prompt sent to the model
python -m src.cli query "Explain this concept" --dump-prompt

# Print retrieval and generation latency report
python -m src.cli query "Analyse the methodology" --latency
```

### API

```bash
# Health check
curl http://127.0.0.1:8000/api/health

# Streamed query via SSE
curl -N -H "Content-Type: application/json" \
  -d '{"query":"Compare these views","stream":true,"mode":"regular","citations_enabled":true}' \
  http://127.0.0.1:8000/api/query

# Upload a document
curl -F "file=@paper.pdf" -F "source_id=my_paper" -F "summarize=true" \
  http://127.0.0.1:8000/api/sources/upload

# Upload + geotag locations during ingest
curl -F "file=@paper.pdf" -F "source_id=my_paper" -F "summarize=true" -F "geotag=true" \
    http://127.0.0.1:8000/api/sources/upload

# Upload + people indexing during ingest
curl -F "file=@paper.pdf" -F "source_id=my_paper" -F "summarize=true" -F "peopletag=true" \
    http://127.0.0.1:8000/api/sources/upload

# List grouped geo mentions for selected source(s)
curl "http://127.0.0.1:8000/api/geo/mentions?min_confidence=0.75&source_ids=my_paper"

# Filter geo mentions by search text (server-side, with safe fallback)
curl "http://127.0.0.1:8000/api/geo/mentions?min_confidence=0.75&source_ids=my_paper&q=pari"

# List canonical people for selected source(s)
curl "http://127.0.0.1:8000/api/people?source_ids=my_paper"

# Filter people list by canonical/raw name text
curl "http://127.0.0.1:8000/api/people?source_ids=my_paper&q=foucault"

# List mention rows for a canonical person
curl "http://127.0.0.1:8000/api/people/mentions?canonical_name=Noam%20Chomsky&source_ids=my_paper"
```

### Retrieval and ingest diagnostics notes

- Retrieval now exposes two core knobs in runtime config (`src/config.py`):
    - `bm25_weight` (default `0.5`): balances lexical BM25 vs vector scoring in LanceDB hybrid search.
    - `use_hybrid` (default `true`): toggles hybrid retrieval on/off (`false` runs dense vector-only retrieval).
- Ingest endpoints now return optional NER diagnostics fields:
    - `geotag_ner`: diagnostics for place extraction.
    - `peopletag_ner`: diagnostics for person extraction.
- Diagnostics include `ner_available`, `method`, and optional `warning`, allowing the UI to flag degraded indexing without failing ingest.

### Phoenix Observability (Arize)

Corpus now includes native Phoenix tracing for:

- Intent classification
- Retrieval stage pipeline (hybrid search, rerank, dedup, threshold filter, budget expand, context expansion)
- Token budget packing
- Prompt construction
- LLM generation (synchronous and streaming)
- Freeform chat spans with emitted `trace_id` / `span_id`
- User feedback annotations via `POST /api/feedback`
- Benchmark retrieval-mode traces (`hybrid_rrf`, `vector_only`, `bm25_only`)

Enable tracing via environment variables:

```bash
export RAG_PHOENIX_ENABLED=1
export PHOENIX_PROJECT_NAME=corpus-local
export PHOENIX_COLLECTOR_ENDPOINT=http://127.0.0.1:6006/v1/traces
```

When `RAG_PHOENIX_ENABLED` is truthy, backend lifespan startup attempts to auto-start a local Phoenix server (`python -m phoenix.server.main serve`) and shuts it down on API process exit.

Optional cloud auth:

```bash
export PHOENIX_API_KEY=your-api-key
```

CLI overrides are also available on `query`, `ingest`, `benchmark`, and retrieval-only benchmark commands:

```bash
python -m src.cli query "Compare these arguments" --phoenix --phoenix-project corpus-cli

python -m src.benchmark.retrieval_only benchmarks/ground_truth/bm25vsvector.json \
    --phoenix --phoenix-project corpus-retrieval-bench
```

Check runtime tracing status from health:

```bash
curl http://127.0.0.1:8000/api/health
```

Health responses now include `phoenix_configured`, `phoenix_active`, `phoenix_project_name`, `phoenix_endpoint`, and `phoenix_error`.

RAG streams emit `data-trace-id` annotations; freeform streams emit `trace_id` SSE events so the frontend can attach thumbs-up / thumbs-down feedback to exact spans.

![Phoenix demo screenshot](docs/images/phoenix-demo.png)

---

## Project Structure

<details>
<summary><strong>Backend (src/)</strong></summary>

| File | Lines | Purpose |
|---|---|---|
| api.py | 2,098 | FastAPI endpoints, SSE streaming, source/geo/people APIs, engine hot-swap, freeform chat |
| rag_engine.py | 2,596 | Central orchestrator: query pipeline, ingest pipeline, citation dedup/sanitisation |
| retrieval.py | 1,390 | 7-stage hybrid retrieval pipeline with intent-scaled parameters |
| generator.py | 945 | MLX text generation, streaming, token budget packing, presence penalty |
| intent.py | 973 | Heuristic and LLM intent classifier: 12 intents, typo tolerance |
| generation.py | 635 | Intent-aware prompt construction: 12 instruction profiles across 2 mode variants |
| storage.py | 1,606 | LanceDB vector/FTS storage with geo + person mention tables and cascades |
| ingest.py | 1,003 | PDF/Markdown parsing, hierarchical chunking, shared geo/person NER enrichment |
| geocoder.py | 1,381 | Offline GeoNames geocoder (forward, reverse, near) with hardening and coherence scoring |
| ner.py | 678 | GLiNER-based place/person NER with windowed inference and typed candidates |
| person_resolver.py | 438 | Canonical person-name resolver with exact/fuzzy matching and promotion rules |
| geo_types.py | 18 | Shared geocoder enums and readiness states |
| reranker.py | 423 | Jina v3 MLX listwise reranker with MLP projector |
| config.py | 441 | Mode configs plus geotag/peopletag thresholds and RAM-aware overrides |
| phoenix_tracing.py | 752 | Phoenix/OpenTelemetry setup, span helpers, and feedback annotation logging |
| transcription.py | 590 | MLX Whisper with pure-NumPy voice activity detection |
| embeddings.py | 475 | MLX embedding wrapper: last-token pooling, left-padding, instruction prefixes |
| stream_protocol.py | 251 | AI SDK v6 UI Message Stream encoder (SSE frames) |
| query_events.py | 105 | Pipeline event dataclasses: 9 event types (including TraceEvent) |
| source_cache.py | 182 | Full-text snapshot caching for the citation viewer |
| metrics.py | 170 | Retrieval and budget metrics: 6 dataclasses |
| latency.py | 107 | Zero-cost latency profiler with ASCII chart output |
| cli.py | 583 | CLI entry point with ingest/query and geotag/peopletag controls |

</details>

<details>
<summary><strong>Frontend (frontend/src/)</strong></summary>

| File | Purpose |
|---|---|
| app/page.tsx | Entry point, modular layout composition, dual-mode mount, map/people overlays |
| app/api/chat/route.ts | Next.js streaming proxy to backend (unbuffered SSE) |
| context/app-context.tsx | Global state: useReducer, 15 action types, source selection + streaming context |
| context/theme-context.tsx | Background theme state: 7 themes, localStorage persistence |
| components/source-panel.tsx | Document library: sequential upload queue, source filtering, ingest status |
| components/people/PeopleDictionary.tsx | People dictionary overlay: search, merge, mention browse/delete, document jump |
| components/layout/top-bar.tsx | Header chrome: mode tabs, map/people toggles, theme, history |
| components/layout/source-panel-container.tsx | Collapsible source sidebar wrapper with glass chrome styles |
| components/layout/overlay-panel.tsx | Shared animated overlay shell for map and people panels |
| components/layout/rag-area.tsx | RAG runtime wrapper with AI SDK stream callbacks |
| components/map/CorpusMap.tsx | Interactive MapLibre map: clustering, source filters, confidence threshold, mention actions |
| components/freeform-chat-panel.tsx | Freeform streaming chat: model and thinking selection, AI-generated titles |
| components/citation-viewer-modal.tsx | Inline citation reader: parallel content and chunk fetch |
| components/history-panel.tsx | Session history drawer: debounced search, glassmorphism styling |
| components/ingest-modal.tsx | Drag-drop upload with geotag/peopletag toggles and optional citation reference |
| components/assistant-ui/thread.tsx | Chat thread: composer with model, intent, think, and speech controls |
| components/assistant-ui/thinking-panel.tsx | Pipeline status steps and streaming reasoning display |
| components/assistant-ui/message-references.tsx | Citation drawer: cited/uncited badges, lazy chunk loading, citation export |
| lib/session-store.ts | IndexedDB session persistence with reasoning content |
| lib/format-citations.ts | Footnote and Harvard citation formatters with marker renumbering |
| lib/text-highlighter.ts | DOM TreeWalker exact substring match and `<mark>` injection |
| hooks/use-stream-handler.ts | Stream annotation dispatcher bridge into AppContext |
| hooks/use-persisted-state.ts | localStorage-backed numeric state with cross-tab sync |
| hooks/useSpeechToText.ts | Offline voice input: MediaRecorder, VAD, silence detection |
| hooks/useSystemRam.ts | RAM detection via /api/health: gates Deep Research and Think toggle |

</details>

---
