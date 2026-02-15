"""FastAPI application wrapping the RAG engine.

Provides streaming chat via the AI SDK Data Stream Protocol and
CRUD endpoints for source management.

Architecture
~~~~~~~~~~~~
- Single ``RagEngine`` instance shared across requests (loaded at startup)
- ``chat_lock`` enforces single-user concurrency for the chat endpoint
- Chat streaming uses ``RagEngine.query_events()`` in a background thread,
  pushing events through an ``asyncio.Queue`` to the response generator.
- Client disconnect detection via a ``should_stop`` callback that checks
  both an ``asyncio.Event`` and ``request.is_disconnected()``.
"""

from __future__ import annotations

import asyncio
import json
import logging
import threading
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncGenerator, Optional

from fastapi import FastAPI, File, Form, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from sse_starlette.sse import EventSourceResponse

from .api_schemas import (
    ChatRequest,
    ChunkDetailResponse,
    ErrorResponse,
    HealthResponse,
    IngestRequest,
    IngestResponse,
    QueryRequest,
    QueryResponse,
    SourceContentResponse,
    SourceDeleteResponse,
    SourceInfo,
    SourceListResponse,
)
from .query_events import (
    CitationListEvent,
    ErrorEvent,
    FinishEvent,
    IntentEvent,
    QueryEvent,
    SourcesEvent,
    StatusEvent,
    TextTokenEvent,
)
from .stream_protocol import (
    annotation_error,
    annotation_intent,
    annotation_sources,
    annotation_status,
    encode_error,
    encode_finish_message,
    encode_finish_step,
    encode_text,
    http_error_body,
)

# Ensure chat/producer logs are visible under uvicorn
def _ensure_app_logging() -> None:
    root = logging.getLogger()
    if root.level > logging.INFO:
        root.setLevel(logging.INFO)
    # If no handler accepts INFO, add one (uvicorn may only log its own records)
    if not any(h.level <= logging.INFO for h in root.handlers):
        h = logging.StreamHandler()
        h.setLevel(logging.INFO)
        h.setFormatter(logging.Formatter("%(levelname)s: %(name)s: %(message)s"))
        root.addHandler(h)


_ensure_app_logging()
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module-level state (set during lifespan)
# ---------------------------------------------------------------------------

_engine: Optional[object] = None  # Will be RagEngine once loaded
_chat_lock: asyncio.Lock = asyncio.Lock()
_ingest_lock: asyncio.Lock = asyncio.Lock()
_engine_loaded: bool = False


def _get_engine():
    """Return the RagEngine instance, importing lazily to avoid heavy imports at module level."""
    global _engine, _engine_loaded
    if _engine is not None:
        return _engine

    from .rag_engine import RagEngine, RagEngineConfig

    logger.info("Initializing RagEngine...")
    config = RagEngineConfig()
    _engine = RagEngine(config)
    _engine_loaded = True
    logger.info("RagEngine initialized successfully")
    return _engine


# ---------------------------------------------------------------------------
# Lifespan
# ---------------------------------------------------------------------------


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan: initialize engine on startup, cleanup on shutdown."""
    global _engine, _engine_loaded
    try:
        import os

        if os.environ.get("RAG_EAGER_LOAD", "").strip() == "1":
            logger.info("RAG_EAGER_LOAD=1: loading engine at startup...")
            await asyncio.to_thread(_get_engine)
    except Exception:
        logger.exception("Failed to initialize RagEngine at startup")
    yield
    if _engine is not None:
        try:
            _engine.close()  # type: ignore[union-attr]
        except Exception:
            pass
    _engine = None
    _engine_loaded = False


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(
    title="DH Notebook RAG API",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Error handlers
# ---------------------------------------------------------------------------


@app.exception_handler(422)
async def validation_exception_handler(request: Request, exc: Exception):
    """Convert FastAPI validation errors to our error contract."""
    logger.warning("Validation error: %s", exc)
    return JSONResponse(
        status_code=422,
        content=http_error_body("VALIDATION_ERROR", str(exc)),
    )


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------


@app.get("/api/health", response_model=HealthResponse)
async def health():
    return HealthResponse(status="ok", engine_loaded=_engine_loaded)


# ---------------------------------------------------------------------------
# Chat stream: plain-text format for AI SDK Text Stream Protocol
# ---------------------------------------------------------------------------

# Headers for useChat with TextStreamChatTransport (plain text, no data protocol)
TEXT_STREAM_HEADERS: dict[str, str] = {
    "Content-Type": "text/plain; charset=utf-8",
    "Cache-Control": "no-cache",
    "X-Accel-Buffering": "no",
}


def _event_to_text_chunk(event: QueryEvent) -> Optional[str]:
    """Convert a QueryEvent to a plain-text chunk for the text stream.

    We send status events as lines so the client receives data during long
    model-load phases and does not timeout. Text tokens and errors are sent as-is.
    CitationListEvent is serialised as a ``CITATIONS:{json}`` line.
    """
    if isinstance(event, StatusEvent):
        return event.status + "\n"
    if isinstance(event, TextTokenEvent):
        return event.token
    if isinstance(event, CitationListEvent):
        import json as _json
        return f"CITATIONS:{_json.dumps(event.citations)}\n"
    if isinstance(event, ErrorEvent):
        return f"Error: {event.message}\n"
    return None


def _encode_event(event: QueryEvent) -> Optional[str]:
    """Convert a QueryEvent to an AI SDK Data Stream Protocol line.

    Returns None for events that don't need a protocol line (shouldn't happen,
    but handles unexpected event types gracefully).
    """
    if isinstance(event, StatusEvent):
        return annotation_status(event.status)
    elif isinstance(event, IntentEvent):
        return annotation_intent(event.intent, event.confidence, event.method)
    elif isinstance(event, SourcesEvent):
        return annotation_sources(event.source_ids)
    elif isinstance(event, TextTokenEvent):
        return encode_text(event.token)
    elif isinstance(event, ErrorEvent):
        return annotation_error(event.code, event.message) + encode_error(event.message)
    elif isinstance(event, FinishEvent):
        return (
            encode_finish_step(event.finish_reason)
            + encode_finish_message(
                event.finish_reason,
                completion_tokens=event.completion_tokens,
            )
        )
    logger.warning("Unknown event type: %s", type(event).__name__)
    return None


# ---------------------------------------------------------------------------
# Chat (streaming with query_events)
# ---------------------------------------------------------------------------

# Sentinel value to signal end of event stream
_SENTINEL = object()


async def _chat_stream_generator(
    request: Request,
    chat_request: ChatRequest,
    stop_event: threading.Event,
) -> AsyncGenerator[str, None]:
    """Generate AI SDK Data Stream Protocol lines from query_events.

    Runs ``RagEngine.query_events()`` in a background thread. Events are
    pushed into an ``asyncio.Queue`` via ``loop.call_soon_threadsafe`` and
    consumed here for encoding and yielding.
    """
    queue: asyncio.Queue = asyncio.Queue()
    loop = asyncio.get_running_loop()

    # --- Extract query parameters ---
    last_message = chat_request.messages[-1]
    query_text = last_message.get_text().strip()

    if not query_text:
        yield "Error: Empty query\n"
        return

    # Send an immediate chunk so the client gets a first byte before model loading.
    # _get_engine() in the producer can take 30–60+ seconds on first request.
    logger.info("Chat stream: sending initial status (engine may load next)")
    yield "Loading RAG engine…\n"

    source_id = None
    citations_enabled = None
    if chat_request.data:
        # Support both source_ids (plural, from frontend) and source_id (singular, legacy).
        # The engine only supports single source_id filtering; if exactly one source
        # is selected, pass it through.  Otherwise pass None (use all sources).
        source_ids = chat_request.data.get("source_ids")
        if isinstance(source_ids, list) and len(source_ids) == 1:
            source_id = source_ids[0]
        elif not source_ids:
            source_id = chat_request.data.get("source_id")
        if "citations_enabled" in chat_request.data:
            citations_enabled = bool(chat_request.data["citations_enabled"])

    # --- Producer: runs in thread, pushes events to queue ---
    def _producer() -> None:
        try:
            logger.info("Chat producer: thread started")
            engine = _get_engine()
            logger.info("Chat producer: _get_engine() returned, starting query_events")
            for event in engine.query_events(
                query_text,
                source_id=source_id,
                citations_enabled=citations_enabled,
                should_stop=stop_event.is_set,
            ):
                loop.call_soon_threadsafe(queue.put_nowait, event)
        except Exception as exc:
            logger.exception("query_events producer error: %s", exc)
            error_event = ErrorEvent(
                code="INTERNAL", message=f"{type(exc).__name__}: {exc}"
            )
            finish_event = FinishEvent(finish_reason="error")
            loop.call_soon_threadsafe(queue.put_nowait, error_event)
            loop.call_soon_threadsafe(queue.put_nowait, finish_event)
        finally:
            loop.call_soon_threadsafe(queue.put_nowait, _SENTINEL)

    # Start producer in thread
    producer_task = loop.run_in_executor(None, _producer)

    # --- Consumer: read events from queue, encode, yield ---
    # Use a timeout so we can yield keepalive bytes during long model loads and avoid client timeout
    first_event = True
    keepalive_interval = 8.0
    try:
        while True:
            try:
                event = await asyncio.wait_for(queue.get(), timeout=keepalive_interval)
            except asyncio.TimeoutError:
                yield " \n"  # keepalive (space+newline) so client can ignore, connection stays open
                continue
            if event is _SENTINEL:
                break
            if first_event:
                logger.info("Chat consumer: first event from producer (stream connected)")
                first_event = False

            chunk = _event_to_text_chunk(event)
            if chunk:
                yield chunk

    except asyncio.CancelledError:
        stop_event.set()
        logger.info("Chat stream consumer cancelled")
    except Exception as exc:
        logger.exception("Chat stream consumer error: %s", exc)
        yield f"Error: {exc}\n"
    finally:
        stop_event.set()
        # Wait for producer thread to finish
        try:
            await asyncio.wait_for(producer_task, timeout=5.0)
        except (asyncio.TimeoutError, Exception):
            pass


@app.post("/api/chat")
async def chat(request: Request, chat_request: ChatRequest):
    """Stream a chat response using the AI SDK Data Stream Protocol.

    Enforces single-user concurrency via ``chat_lock``. Returns 429 if
    another query is already in progress.
    """
    logger.info("Chat: request received")
    if _chat_lock.locked():
        return JSONResponse(
            status_code=429,
            content=http_error_body(
                "LOCK_BUSY", "Another query is already in progress"
            ),
        )

    stop_event = threading.Event()

    async def guarded_stream() -> AsyncGenerator[bytes, None]:
        """Stream with lock acquisition, disconnect detection, and cleanup."""
        async with _acquire_chat_lock():
            gen = _chat_stream_generator(request, chat_request, stop_event)
            try:
                async for line in gen:
                    # Periodically check for client disconnect
                    if await request.is_disconnected():
                        stop_event.set()
                        return
                    yield line.encode("utf-8")
            except asyncio.CancelledError:
                stop_event.set()
            finally:
                stop_event.set()

    @asynccontextmanager
    async def _acquire_chat_lock():
        await _chat_lock.acquire()
        try:
            yield
        finally:
            _chat_lock.release()

    return StreamingResponse(
        guarded_stream(),
        headers=TEXT_STREAM_HEADERS,
    )


async def _query_sse_event_generator(
    request: Request,
    query_request: QueryRequest,
    stop_event: threading.Event,
) -> AsyncGenerator[dict[str, str], None]:
    """Generate SSE events from ``RagEngine.query_events()``."""
    queue: asyncio.Queue = asyncio.Queue()
    loop = asyncio.get_running_loop()

    source_id = None
    if query_request.source_ids and len(query_request.source_ids) == 1:
        source_id = query_request.source_ids[0]

    def _producer() -> None:
        try:
            engine = _get_engine()
            for event in engine.query_events(
                query_request.query,
                source_id=source_id,
                citations_enabled=query_request.citations_enabled,
                should_stop=stop_event.is_set,
            ):
                loop.call_soon_threadsafe(queue.put_nowait, event)
        except Exception as exc:
            logger.exception("query sse producer error: %s", exc)
            error_event = ErrorEvent(
                code="INTERNAL", message=f"{type(exc).__name__}: {exc}"
            )
            finish_event = FinishEvent(finish_reason="error")
            loop.call_soon_threadsafe(queue.put_nowait, error_event)
            loop.call_soon_threadsafe(queue.put_nowait, finish_event)
        finally:
            loop.call_soon_threadsafe(queue.put_nowait, _SENTINEL)

    producer_task = loop.run_in_executor(None, _producer)

    try:
        while True:
            if await request.is_disconnected():
                stop_event.set()
                return

            event = await queue.get()
            if event is _SENTINEL:
                return

            if isinstance(event, StatusEvent):
                yield {
                    "event": "status",
                    "data": json.dumps({"message": event.status}),
                }
            elif isinstance(event, IntentEvent):
                yield {
                    "event": "intent",
                    "data": json.dumps(
                        {
                            "intent": event.intent,
                            "confidence": event.confidence,
                            "method": event.method,
                        }
                    ),
                }
            elif isinstance(event, SourcesEvent):
                yield {
                    "event": "sources",
                    "data": json.dumps({"source_ids": event.source_ids}),
                }
            elif isinstance(event, CitationListEvent):
                yield {
                    "event": "citations",
                    "data": json.dumps({"citations": event.citations}),
                }
            elif isinstance(event, TextTokenEvent):
                yield {
                    "event": "token",
                    "data": json.dumps({"text": event.token}),
                }
            elif isinstance(event, ErrorEvent):
                yield {
                    "event": "error",
                    "data": json.dumps({"code": event.code, "error": event.message}),
                }
            elif isinstance(event, FinishEvent):
                yield {
                    "event": "complete",
                    "data": json.dumps(
                        {
                            "finish_reason": event.finish_reason,
                            "completion_tokens": event.completion_tokens,
                            "prompt_tokens": event.prompt_tokens,
                        }
                    ),
                }
    finally:
        stop_event.set()
        try:
            await asyncio.wait_for(producer_task, timeout=5.0)
        except (asyncio.TimeoutError, Exception):
            pass


@app.post("/api/query", response_model=QueryResponse)
async def query_endpoint(request: Request, query_request: QueryRequest):
    """Query endpoint with optional SSE streaming.

    When ``stream=true``, returns text/event-stream events for status + tokens.
    Otherwise returns a single JSON payload.
    """
    if _chat_lock.locked():
        return JSONResponse(
            status_code=429,
            content=http_error_body(
                "LOCK_BUSY", "Another query is already in progress"
            ),
        )

    source_id = None
    if query_request.source_ids and len(query_request.source_ids) == 1:
        source_id = query_request.source_ids[0]

    if not query_request.stream:
        async with _chat_lock:
            result = await asyncio.to_thread(
                _get_engine().query,
                query_request.query,
                source_id=source_id,
                citations_enabled=query_request.citations_enabled,
            )
            return QueryResponse(
                answer=result.answer,
                citations=[],
                source_ids=result.source_ids,
                metrics={
                    "completion_tokens": 0,
                    "finish_reason": "stop",
                },
            )

    stop_event = threading.Event()

    async def guarded_event_stream() -> AsyncGenerator[dict[str, str], None]:
        async with _chat_lock:
            async for event in _query_sse_event_generator(
                request, query_request, stop_event
            ):
                yield event

    return EventSourceResponse(
        guarded_event_stream(),
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
        media_type="text/event-stream",
    )


# ---------------------------------------------------------------------------
# Source management endpoints
# ---------------------------------------------------------------------------


@app.get("/api/sources", response_model=SourceListResponse)
async def list_sources():
    """List all ingested sources with metadata."""
    engine = await asyncio.to_thread(_get_engine)
    storage = engine.storage  # type: ignore[union-attr]

    # Get full details from summaries table (schema v2)
    details = storage.get_source_details()
    # Also get source IDs from parent chunks (may include sources without summaries)
    all_ids = set(storage.list_source_ids())

    sources = []
    seen_ids = set()

    for detail in details:
        sid = detail["source_id"]
        seen_ids.add(sid)
        source_path = detail.get("source_path") or None
        snapshot_path = detail.get("snapshot_path") or None
        source_size_bytes = _safe_file_size(source_path)
        snapshot_size_bytes = _safe_file_size(snapshot_path)
        sources.append(SourceInfo(
            source_id=sid,
            summary=detail.get("summary") or None,
            source_path=source_path,
            snapshot_path=snapshot_path,
            source_size_bytes=source_size_bytes,
            content_size_bytes=source_size_bytes or snapshot_size_bytes,
        ))

    # Add any source IDs that have chunks but no summary
    for sid in sorted(all_ids - seen_ids):
        sources.append(SourceInfo(source_id=sid))

    return SourceListResponse(sources=sources)


@app.post("/api/sources/ingest", response_model=IngestResponse)
async def ingest_source(request: IngestRequest):
    """Ingest a document (PDF or Markdown) into the RAG store.

    Also creates a text snapshot for the ``/content`` endpoint.
    """
    from .source_cache import save_snapshot

    file_path = request.file_path
    source_id = request.source_id

    # Validate file exists
    if not Path(file_path).is_file():
        return JSONResponse(
            status_code=404,
            content=http_error_body(
                "SOURCE_NOT_FOUND",
                f"File not found: {file_path}",
            ),
        )

    try:
        async with _ingest_lock:
            engine = await asyncio.to_thread(_get_engine)

            # Run ingest in thread (blocking operation)
            result = await asyncio.to_thread(
                engine.ingest,  # type: ignore[union-attr]
                file_path,
                source_id=source_id,
                summarize=request.summarize,
            )

            # Create text snapshot for /content endpoint
            snapshot_path = ""
            try:
                # Collect parent texts for the snapshot
                storage = engine.storage  # type: ignore[union-attr]
                parent_texts = storage.get_parent_texts_by_source(source_id=source_id)
                if parent_texts:
                    full_text = "\n\n".join(parent_texts)
                    snapshot_path = await asyncio.to_thread(
                        save_snapshot, source_id, full_text
                    )
            except Exception as snap_exc:
                logger.warning("Failed to create snapshot for %s: %s", source_id, snap_exc)

            # Update the summary record with file paths (schema v2)
            try:
                storage = engine.storage  # type: ignore[union-attr]
                summaries = storage.get_source_summaries()
                summary_text = summaries.get(source_id, "")
                if summary_text:
                    storage.upsert_source_summary(
                        source_id=source_id,
                        summary=summary_text,
                        source_path=str(Path(file_path).resolve()),
                        snapshot_path=snapshot_path,
                    )
            except Exception as path_exc:
                logger.warning("Failed to update source paths for %s: %s", source_id, path_exc)

        return IngestResponse(
            source_id=result.source_id,
            parents_count=result.parents_count,
            children_count=result.children_count,
            summarized=result.summarized,
        )

    except Exception as exc:
        logger.exception("Ingest failed for %s: %s", source_id, exc)
        return JSONResponse(
            status_code=500,
            content=http_error_body(
                "INGEST_FAILED",
                f"Ingest failed: {type(exc).__name__}: {exc}",
            ),
        )


# Maximum upload size: 50 MB
_MAX_UPLOAD_BYTES = 50 * 1024 * 1024
_UPLOAD_DIR = Path(__file__).resolve().parent.parent / "data" / "uploads"
_ALLOWED_EXTENSIONS = {".pdf", ".md", ".markdown"}


def _sanitize_source_id(name: str) -> str:
    """Derive a safe source_id from a filename."""
    import re as _re
    stem = Path(name).stem
    # Replace non-alphanumeric (except hyphen/underscore) with underscore
    safe = _re.sub(r"[^\w\-]", "_", stem)
    # Collapse multiple underscores
    safe = _re.sub(r"_+", "_", safe).strip("_")
    return safe[:120] or "uploaded_doc"


@app.post("/api/sources/upload", response_model=IngestResponse)
async def upload_source(
    file: UploadFile = File(...),
    source_id: str = Form(""),
    summarize: bool = Form(True),
):
    """Upload and ingest a document (PDF or Markdown) from the browser.

    Accepts ``multipart/form-data`` with fields:
    - ``file``: The document file (.pdf, .md, .markdown)
    - ``source_id``: Optional custom ID (auto-generated from filename if empty)
    - ``summarize``: Whether to generate a summary (default true)
    """
    from .source_cache import save_snapshot

    # --- Validate file extension ---
    filename = file.filename or "unknown"
    ext = Path(filename).suffix.lower()
    if ext not in _ALLOWED_EXTENSIONS:
        return JSONResponse(
            status_code=422,
            content=http_error_body(
                "INVALID_FILE_TYPE",
                f"Unsupported file type '{ext}'. Allowed: {', '.join(sorted(_ALLOWED_EXTENSIONS))}",
            ),
        )

    # --- Read and validate size ---
    contents = await file.read()
    if len(contents) > _MAX_UPLOAD_BYTES:
        return JSONResponse(
            status_code=413,
            content=http_error_body(
                "FILE_TOO_LARGE",
                f"File exceeds {_MAX_UPLOAD_BYTES // (1024*1024)}MB limit ({len(contents) / (1024*1024):.1f}MB uploaded).",
            ),
        )
    if len(contents) == 0:
        return JSONResponse(
            status_code=422,
            content=http_error_body("INVALID_FILE", "Uploaded file is empty."),
        )

    # --- Derive source_id ---
    sid = source_id.strip() if source_id else _sanitize_source_id(filename)
    if not sid:
        sid = _sanitize_source_id(filename)

    # --- Save uploaded file to data/uploads/ ---
    _UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    dest = _UPLOAD_DIR / f"{sid}{ext}"
    dest.write_bytes(contents)
    logger.info("Saved upload to %s (%d bytes)", dest, len(contents))

    try:
        async with _ingest_lock:
            engine = await asyncio.to_thread(_get_engine)

            # Run ingest in thread (blocking operation)
            result = await asyncio.to_thread(
                engine.ingest,  # type: ignore[union-attr]
                str(dest),
                source_id=sid,
                summarize=summarize,
            )

            # Create text snapshot for /content endpoint
            snapshot_path = ""
            try:
                storage = engine.storage  # type: ignore[union-attr]
                parent_texts = storage.get_parent_texts_by_source(source_id=sid)
                if parent_texts:
                    full_text = "\n\n".join(parent_texts)
                    snapshot_path = await asyncio.to_thread(
                        save_snapshot, sid, full_text
                    )
            except Exception as snap_exc:
                logger.warning("Failed to create snapshot for %s: %s", sid, snap_exc)

            # Update the summary record with file paths (schema v2)
            try:
                storage = engine.storage  # type: ignore[union-attr]
                summaries = storage.get_source_summaries()
                summary_text = summaries.get(sid, "")
                if summary_text:
                    storage.upsert_source_summary(
                        source_id=sid,
                        summary=summary_text,
                        source_path=str(dest.resolve()),
                        snapshot_path=snapshot_path,
                    )
            except Exception as path_exc:
                logger.warning("Failed to update source paths for %s: %s", sid, path_exc)

        return IngestResponse(
            source_id=result.source_id,
            parents_count=result.parents_count,
            children_count=result.children_count,
            summarized=result.summarized,
        )

    except Exception as exc:
        logger.exception("Upload ingest failed for %s: %s", sid, exc)
        # Clean up the uploaded file on failure
        try:
            dest.unlink(missing_ok=True)
        except Exception:
            pass
        return JSONResponse(
            status_code=500,
            content=http_error_body(
                "INGEST_FAILED",
                f"Ingest failed: {type(exc).__name__}: {exc}",
            ),
        )


@app.delete("/api/sources/{source_id}", response_model=SourceDeleteResponse)
async def delete_source(source_id: str):
    """Delete a source and all its chunks, summary, and cached snapshot."""
    from .source_cache import delete_snapshot

    try:
        engine = await asyncio.to_thread(_get_engine)
        storage = engine.storage  # type: ignore[union-attr]

        # Get snapshot path before deleting from storage
        detail = storage.get_source_detail(source_id)
        snapshot_path = detail.get("snapshot_path", "") if detail else ""

        # Delete from storage (children, parents, summary)
        deleted = await asyncio.to_thread(storage.delete_source, source_id)

        # Delete snapshot file
        if snapshot_path:
            await asyncio.to_thread(delete_snapshot, snapshot_path)

        return SourceDeleteResponse(source_id=source_id, deleted=deleted)

    except Exception as exc:
        logger.exception("Delete failed for source %s: %s", source_id, exc)
        return JSONResponse(
            status_code=500,
            content=http_error_body("INTERNAL", f"Delete failed: {exc}"),
        )


def _detect_format(source_path: Optional[str]) -> str:
    """Derive content format from the source file extension."""
    if not source_path:
        return "text"
    from pathlib import Path as _P
    ext = _P(source_path).suffix.lower()
    if ext == ".pdf":
        return "pdf"
    if ext in {".md", ".markdown", ".mdx"}:
        return "markdown"
    return "text"


def _safe_file_size(path: Optional[str]) -> Optional[int]:
    if not path:
        return None
    try:
        p = Path(path)
        if not p.is_file():
            return None
        return p.stat().st_size
    except Exception:
        return None


@app.get("/api/sources/{source_id}/content", response_model=SourceContentResponse)
async def get_source_content(source_id: str):
    """Get the full text content of a source document.

    Resolution order: original file → cached snapshot → 404.
    """
    from .source_cache import resolve_content

    try:
        engine = await asyncio.to_thread(_get_engine)
        storage = engine.storage  # type: ignore[union-attr]

        detail = storage.get_source_detail(source_id)
        if detail is None:
            return JSONResponse(
                status_code=404,
                content=http_error_body(
                    "SOURCE_NOT_FOUND",
                    f"Source '{source_id}' not found",
                ),
            )

        source_path = detail.get("source_path", "") or None
        snapshot_path = detail.get("snapshot_path", "") or None
        fmt = _detect_format(source_path)

        result = await asyncio.to_thread(resolve_content, source_path, snapshot_path)
        if result is None:
            # Fallback: try to assemble content from parent texts
            parent_texts = storage.get_parent_texts_by_source(source_id=source_id)
            if parent_texts:
                return SourceContentResponse(
                    source_id=source_id,
                    content="\n\n".join(parent_texts),
                    content_source="summary",
                    format=fmt,
                )
            return JSONResponse(
                status_code=404,
                content=http_error_body(
                    "SOURCE_NOT_FOUND",
                    f"Content not available for source '{source_id}'",
                ),
            )

        content, content_source = result
        return SourceContentResponse(
            source_id=source_id,
            content=content,
            content_source=content_source,
            format=fmt,
        )

    except Exception as exc:
        logger.exception("Content fetch failed for source %s: %s", source_id, exc)
        return JSONResponse(
            status_code=500,
            content=http_error_body("INTERNAL", f"Content fetch failed: {exc}"),
        )


@app.get("/api/sources/{source_id}/chunk/{chunk_id}", response_model=ChunkDetailResponse)
async def get_chunk_detail(source_id: str, chunk_id: str):
    """Return full citation context for a single chunk.

    Used by the frontend to populate the document viewer modal with
    chunk text, page number, header path, and format so it can scroll
    and highlight the cited passage.
    """
    try:
        engine = await asyncio.to_thread(_get_engine)
        storage = engine.storage  # type: ignore[union-attr]

        children = storage.get_children_by_ids([chunk_id])
        if chunk_id not in children:
            return JSONResponse(
                status_code=404,
                content=http_error_body(
                    "SOURCE_NOT_FOUND",
                    f"Chunk '{chunk_id}' not found",
                ),
            )

        child = children[chunk_id]
        meta = child.get("metadata", {})
        if isinstance(meta, dict) and meta.get("source_id") != source_id:
            return JSONResponse(
                status_code=404,
                content=http_error_body(
                    "SOURCE_NOT_FOUND",
                    f"Chunk '{chunk_id}' does not belong to source '{source_id}'",
                ),
            )

        # Fetch parent text if available
        parent_text: Optional[str] = None
        parent_id = meta.get("parent_id") if isinstance(meta, dict) else None
        if parent_id:
            parent_text = storage.get_parent_text(parent_id)

        # Get source detail for format/source_path
        detail = storage.get_source_detail(source_id)
        sp = detail.get("source_path", "") if detail else ""
        fmt = _detect_format(sp or None)

        page_num_raw = meta.get("page_number") if isinstance(meta, dict) else None
        page_number = int(page_num_raw) if page_num_raw is not None else None

        return ChunkDetailResponse(
            source_id=source_id,
            chunk_id=chunk_id,
            chunk_text=str(child.get("text", "")),
            parent_text=parent_text,
            page_number=page_number,
            display_page=str(meta.get("display_page", "")) if isinstance(meta, dict) and meta.get("display_page") else None,
            header_path=str(meta.get("header_path", "")) if isinstance(meta, dict) else "",
            format=fmt,
            source_path=sp or None,
        )

    except Exception as exc:
        logger.exception("Chunk detail failed for %s/%s: %s", source_id, chunk_id, exc)
        return JSONResponse(
            status_code=500,
            content=http_error_body("INTERNAL", f"Chunk detail failed: {exc}"),
        )
