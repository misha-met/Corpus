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
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncGenerator, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from .rag_engine import RagEngine

from fastapi import FastAPI, File, Form, HTTPException, Query, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel

from . import config as app_config
from .api_schemas import (
    ChatRequest,
    ChunkBatchItem,
    ChunkBatchResponse,
    ChunkDetailResponse,
    ErrorResponse,
    HealthResponse,
    IngestRequest,
    IngestResponse,
    NERDiagnosticsResponse,
    PeopleListResponse,
    PeopleMergeRequest,
    PeopleMergeResponse,
    PersonMentionsResponse,
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
    ThinkingTokenEvent,
    TraceEvent,
)
from .phoenix_tracing import (
    SPAN_KIND_CHAIN,
    SPAN_KIND_LLM,
    annotate_span_feedback,
    get_phoenix_runtime_status,
    mark_span_error,
    set_llm_input_messages,
    set_llm_output_message,
    set_llm_token_counts,
    set_span_attributes,
    start_span,
    to_json,
)
from .stream_protocol import (
    annotation_error,
    annotation_error_with_metadata,
    annotation_citations,
    annotation_intent,
    annotation_sources,
    annotation_status,
    encode_annotation,
    encode_done,
    encode_error,
    encode_finish_message,
    encode_finish_step,
    encode_message_start,
    encode_reasoning_delta,
    encode_reasoning_end,
    encode_reasoning_start,
    encode_text_delta,
    encode_text_end,
    encode_text_start,
    http_error_body,
    STREAM_HEADERS,
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
# Tuning constants (were previously magic numbers scattered in handler bodies)
# ---------------------------------------------------------------------------

# SSE keepalive ping interval (seconds) — prevents client-side timeout
# during long model loads.
_KEEPALIVE_INTERVAL_S: float = 8.0

# Timeout for waiting on the producer thread to finish after the consumer
# exits or the client disconnects.
_PRODUCER_CLEANUP_TIMEOUT_S: float = 5.0

# Maximum upload size (bytes) — 50 MB
_MAX_UPLOAD_BYTES: int = 50 * 1024 * 1024

# ---------------------------------------------------------------------------
# Module-level state (set during lifespan)
# ---------------------------------------------------------------------------

_engine: RagEngine | None = None      # the single currently-loaded RagEngine
_engine_mode: str | None = None      # which mode that engine was built for
_engine_init_lock = threading.Lock()
_engine_loaded: bool = False

# Single-user concurrency guard: only one chat/freeform stream at a time.
_chat_lock = asyncio.Lock()

# Map frontend model IDs (from assistant-ui ModelSelector) to backend mode strings
_FRONTEND_MODE_MAP: dict[str, str] = {
    "regular": "regular",
    "deep-research": "deep-research",
}


def _get_engine(mode: str | None = None):
    """Return the RagEngine for the given mode.

    Only one engine is kept in memory at a time.  If the requested mode
    differs from the currently-loaded engine, the old engine is closed and a
    new one is initialised for the new mode.

    If *mode* is None (the default), the currently-loaded engine is returned
    as-is.  A new engine is only cold-started when no engine is loaded yet,
    in which case ``"regular"`` is used as the fallback mode.  This prevents
    non-chat endpoints (source lookups, ingest, etc.) from accidentally
    triggering a model swap back to ``"regular"`` while a different model is
    active.
    """
    global _engine, _engine_mode, _engine_loaded

    # Resolve mode: keep whatever is loaded; only default to "regular" when
    # nothing is loaded at all.
    if mode is None:
        mode = _engine_mode if _engine_mode is not None else "regular"

    # Fast path — already the right model.
    if _engine is not None and _engine_mode == mode:
        return _engine

    from .rag_engine import RagEngine, RagEngineConfig

    with _engine_init_lock:
        # Re-check under the lock in case another thread just swapped.
        if _engine is not None and _engine_mode == mode:
            return _engine

        # Unload the old engine if one is loaded.
        if _engine is not None:
            logger.info(
                "Swapping engine: unloading mode=%r, loading mode=%r",
                _engine_mode,
                mode,
            )
            try:
                _engine.close()  # type: ignore[union-attr]
            except Exception:
                logger.exception("Error closing old engine (mode=%r)", _engine_mode)
            _engine = None
            _engine_mode = None
            _engine_loaded = False

        try:
            logger.info("Initializing RagEngine (mode=%s)...", mode)
            config = RagEngineConfig(mode=mode)
            _engine = RagEngine(config)
            _engine_mode = mode
            _engine_loaded = True
            logger.info("RagEngine (mode=%s) initialized successfully", mode)
            return _engine
        except Exception:
            logger.exception("RagEngine initialization failed (mode=%s)", mode)
            raise


# ---------------------------------------------------------------------------
# Lifespan
# ---------------------------------------------------------------------------


_phoenix_process = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan: initialize engine on startup, cleanup on shutdown."""
    global _engine_loaded, _phoenix_process
    try:
        import os
        import sys
        import subprocess

        # Auto-start Phoenix server if tracing is enabled and we are running locally
        if os.environ.get("RAG_PHOENIX_ENABLED", "").strip() in ("1", "true", "yes", "on", "y"):
            logger.info("RAG_PHOENIX_ENABLED: Auto-starting local Phoenix tracing server on port 6006...")
            try:
                _phoenix_process = subprocess.Popen(
                    [sys.executable, "-m", "phoenix.server.main", "serve"],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
            except Exception as e:
                logger.warning("Failed to auto-start Phoenix server: %s", e)

        if os.environ.get("RAG_EAGER_LOAD", "").strip() == "1":
            logger.info("RAG_EAGER_LOAD=1: loading engine at startup...")
            await asyncio.to_thread(_get_engine)

    except Exception:
        logger.exception("Failed to initialize backend services at startup")
        
    yield
    
    global _engine, _engine_mode
    if _engine is not None:
        try:
            _engine.close()  # type: ignore[union-attr]
        except Exception:
            logger.warning("Error closing engine during shutdown", exc_info=True)
        _engine = None
        _engine_mode = None
    _engine_loaded = False

    if _phoenix_process is not None:
        logger.info("Shutting down local Phoenix tracing server...")
        try:
            _phoenix_process.terminate()
            _phoenix_process.wait(timeout=5)
        except Exception as e:
            logger.warning("Error during Phoenix server shutdown: %s", e)


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Corpus RAG API",
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
    from .config import get_system_ram_gb

    phoenix_status = get_phoenix_runtime_status()
    fts_policy: Optional[str] = None
    fts_dirty: Optional[bool] = None
    fts_pending_rows: Optional[int] = None
    if _engine is not None:
        try:
            fts_status = _engine.storage.get_fts_status()  # type: ignore[union-attr]
            fts_policy = fts_status.get("fts_policy")
            fts_dirty = bool(fts_status.get("fts_dirty"))
            fts_pending_rows = int(fts_status.get("fts_pending_rows", 0))
        except Exception:
            logger.debug("Unable to collect FTS status for health endpoint", exc_info=True)

    return HealthResponse(
        status="ok",
        engine_loaded=_engine_loaded,
        system_ram_gb=round(get_system_ram_gb(), 1),
        phoenix_configured=phoenix_status.configured,
        phoenix_active=phoenix_status.active,
        phoenix_project_name=phoenix_status.project_name,
        phoenix_endpoint=phoenix_status.endpoint,
        phoenix_error=phoenix_status.error,
        fts_policy=fts_policy,
        fts_dirty=fts_dirty,
        fts_pending_rows=fts_pending_rows,
    )


@app.get("/api/geo/status")
async def geo_status() -> dict:
    """Return geocoder readiness, counts, version info, and attribution."""
    from .geocoder import get_geocoder

    geocoder = get_geocoder()
    return geocoder.status()


@app.get("/api/geocode")
async def geocode_forward(
    q: str,
    threshold: int = 72,
    context: str = "",
) -> dict:
    """Forward geocode a place query using the offline GeoNames index."""
    from .geocoder import get_geocoder

    geocoder = get_geocoder()

    context_words = tuple(word.strip() for word in context.split(",") if word.strip())
    result = geocoder.forward(q, threshold=threshold, context_words=context_words)
    if result is None and not geocoder.is_available():
        raise HTTPException(503, detail="Geocoder not ready. Check /api/geo/status.")
    if not result:
        raise HTTPException(404, detail=f"No match found for '{q}'")

    return {
        **result.place.to_dict(),
        "matched_on": result.matched_on,
        "score": result.score,
        "confidence": result.confidence,
        "method": result.method.value,
        "ambiguous": result.ambiguous,
        "candidates": [place.to_dict() for place in result.candidates],
        "attribution": "GeoNames (https://www.geonames.org) - CC BY 4.0",
    }


@app.get("/api/geocode/near")
async def geocode_near(
    lat: float = Query(..., ge=-90.0, le=90.0),
    lon: float = Query(..., ge=-180.0, le=180.0),
    radius_km: float = Query(50.0, ge=0.0, le=1_000.0),
    limit: int = 50,
) -> dict:
    """Find canonical places near a coordinate using offline geocoder KDTree."""
    from .geocoder import get_geocoder

    limit = min(limit, 200)

    geocoder = get_geocoder()

    places = geocoder.find_near(lat, lon, radius_km=radius_km, max_results=limit)
    if not geocoder.is_available():
        raise HTTPException(503, detail="Geocoder not ready. Check /api/geo/status.")
    return {
        "lat": lat,
        "lon": lon,
        "radius_km": radius_km,
        "count": len(places),
        "results": [place.to_dict() for place in places],
        "attribution": "GeoNames (https://www.geonames.org) - CC BY 4.0",
    }


@app.get("/api/geocode/reverse")
async def geocode_reverse(
    lat: float,
    lon: float,
    k: int = Query(3, ge=1, le=10),
) -> dict:
    """Reverse geocode coordinates to nearest canonical places."""
    from .geocoder import get_geocoder

    geocoder = get_geocoder()

    results = geocoder.reverse(lat, lon, k=k)
    if not geocoder.is_available():
        raise HTTPException(503, detail="Geocoder not ready. Check /api/geo/status.")
    return {
        "lat": lat,
        "lon": lon,
        "results": [
            {**place.to_dict(), "distance_km": round(distance, 2)}
            for distance, place in results
        ],
        "attribution": "GeoNames (https://www.geonames.org) - CC BY 4.0",
    }


@app.get("/api/geo/mentions")
async def get_geo_mentions(
    request: Request,
    source_id: str | None = None,
    source_ids: list[str] | None = Query(default=None),
    q: str | None = Query(default=None),
    min_confidence: float = Query(0.75, ge=0.0, le=1.0),
    limit: int = Query(1000, ge=1, le=50_000),
    offset: int = Query(0, ge=0),
    detailed: bool = True,
) -> dict:
    """Return geocoded place mentions grouped by geonameid."""
    normalized_source_id = source_id.strip() if isinstance(source_id, str) else ""

    raw_source_ids = request.query_params.getlist("source_ids")
    source_ids_provided = "source_ids" in request.query_params
    normalized_source_ids: list[str] = []
    if source_ids_provided:
        for raw in raw_source_ids:
            sid = raw.strip()
            if sid and sid not in normalized_source_ids:
                normalized_source_ids.append(sid)

    effective_source_ids: list[str] | None = None
    if app_config.USE_SOURCE_IDS_FILTER:
        if source_ids_provided and not normalized_source_ids and not normalized_source_id:
            return {"count": 0, "mentions": []}

        union: list[str] = []
        if normalized_source_id:
            union.append(normalized_source_id)
        for sid in normalized_source_ids:
            if sid not in union:
                union.append(sid)
        if union:
            effective_source_ids = union

    engine = await asyncio.to_thread(_get_engine)
    storage = engine.storage  # type: ignore[union-attr]
    try:
        rows = await asyncio.to_thread(
            storage.get_geo_mentions,
            source_id=normalized_source_id or None,
            source_ids=effective_source_ids,
            q=q,
            min_confidence=min_confidence,
            limit=limit,
            offset=offset,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    groups: dict[int, dict] = {}
    seen_geo_keys: set[tuple[str, str, str]] = set()
    for row in rows:
        source_key = str(row.get("source_id", "")).strip()
        chunk_key = str(row.get("chunk_id", "")).strip()
        canonical_place = str(row.get("place_name", "")).strip()
        geo_identity_key = (source_key, chunk_key, canonical_place)
        if geo_identity_key in seen_geo_keys:
            continue
        seen_geo_keys.add(geo_identity_key)

        gid = row["geonameid"]
        if gid not in groups:
            group = {
                "place_name": row["place_name"],
                "geonameid": gid,
                "lat": row["lat"],
                "lon": row["lon"],
                "mention_count": 0,
                "max_confidence": 0.0,
            }
            if detailed:
                group.update(
                    {
                        "matched_inputs": [],
                        "source_ids": [],
                        "chunk_ids": [],
                        "mention_ids": [],
                        "mentions": [],
                    }
                )
            groups[gid] = group

        group = groups[gid]
        group["mention_count"] += 1
        group["max_confidence"] = max(group["max_confidence"], row["confidence"])
        if detailed:
            if row["matched_input"] not in group["matched_inputs"]:
                group["matched_inputs"].append(row["matched_input"])
            if row["source_id"] not in group["source_ids"]:
                group["source_ids"].append(row["source_id"])
            if row["chunk_id"] not in group["chunk_ids"]:
                group["chunk_ids"].append(row["chunk_id"])
            group["mention_ids"].append(row["id"])
            group["mentions"].append(
                {
                    "id": row["id"],
                    "source_id": row["source_id"],
                    "chunk_id": row["chunk_id"],
                    "matched_input": row["matched_input"],
                    "confidence": row["confidence"],
                    "method": row["method"],
                }
            )

    result = sorted(groups.values(), key=lambda item: item["mention_count"], reverse=True)
    return {"count": len(result), "mentions": result}


@app.delete("/api/geo/mentions/{mention_id}", status_code=204)
async def delete_geo_mention(mention_id: str) -> None:
    """Delete a single geo mention by UUID."""
    engine = await asyncio.to_thread(_get_engine)
    storage = engine.storage  # type: ignore[union-attr]
    await asyncio.to_thread(storage.delete_geo_mention, mention_id)


@app.get("/api/people", response_model=PeopleListResponse)
async def get_people(
    request: Request,
    source_id: str | None = None,
    source_ids: list[str] | None = Query(default=None),
    min_confidence: float = Query(0.0, ge=0.0, le=1.0),
    q: str | None = Query(default=None),
    limit: int = Query(200, ge=1, le=50_000),
    offset: int = Query(0, ge=0),
) -> PeopleListResponse:
    """Return canonical people grouped from mention-level rows."""
    normalized_source_id = source_id.strip() if isinstance(source_id, str) else ""

    raw_source_ids = request.query_params.getlist("source_ids")
    source_ids_provided = "source_ids" in request.query_params
    normalized_source_ids: list[str] = []
    if source_ids_provided:
        for raw in raw_source_ids:
            sid = raw.strip()
            if sid and sid not in normalized_source_ids:
                normalized_source_ids.append(sid)

    if source_ids_provided and not normalized_source_ids and not normalized_source_id:
        return PeopleListResponse(count=0, people=[])

    effective_source_ids: list[str] | None = None
    union: list[str] = []
    if normalized_source_id:
        union.append(normalized_source_id)
    for sid in normalized_source_ids:
        if sid not in union:
            union.append(sid)
    if union:
        effective_source_ids = union

    engine = await asyncio.to_thread(_get_engine)
    storage = engine.storage  # type: ignore[union-attr]

    try:
        rows = await asyncio.to_thread(
            storage.get_person_mentions,
            source_id=normalized_source_id or None,
            source_ids=effective_source_ids,
            min_confidence=min_confidence,
            q=q,
            limit=50_000,
            offset=0,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    grouped: dict[str, dict[str, object]] = {}
    seen_people_keys: set[tuple[str, str, str]] = set()
    for row in rows:
        source_key = str(row.get("source_id", "")).strip()
        chunk_key = str(row.get("chunk_id", "")).strip()
        raw_name = str(row.get("raw_name", "")).strip()
        if not raw_name:
            continue
        person_identity_key = (source_key, chunk_key, raw_name)
        if person_identity_key in seen_people_keys:
            continue
        seen_people_keys.add(person_identity_key)

        canonical = str(row.get("canonical_name", "")).strip()
        if not canonical:
            continue

        group = grouped.get(canonical)
        if group is None:
            group = {
                "canonical_name": canonical,
                "mention_count": 0,
                "source_ids": set(),
                "variants": set(),
                "roles": set(),
                "confidence_sum": 0.0,
            }
            grouped[canonical] = group

        group["mention_count"] = int(group["mention_count"]) + 1
        source_set = group["source_ids"]
        if isinstance(source_set, set):
            source_set.add(str(row.get("source_id", "")))

        variants_set = group["variants"]
        if isinstance(variants_set, set):
            if raw_name:
                variants_set.add(raw_name)

        roles_set = group["roles"]
        if isinstance(roles_set, set):
            role = row.get("role_hint")
            if role is not None and str(role).strip():
                roles_set.add(str(role).strip())

        try:
            conf = float(row.get("confidence", 0.0))
        except (TypeError, ValueError):
            conf = 0.0
        group["confidence_sum"] = float(group["confidence_sum"]) + conf

    ordered = sorted(
        grouped.values(),
        key=lambda item: (-int(item["mention_count"]), str(item["canonical_name"]).lower()),
    )

    paged = ordered[offset : offset + limit]
    people = [
        {
            "canonical_name": str(item["canonical_name"]),
            "mention_count": int(item["mention_count"]),
            "source_count": len(item["source_ids"]) if isinstance(item["source_ids"], set) else 0,
            "source_ids": sorted(item["source_ids"]) if isinstance(item["source_ids"], set) else [],
            "variants": sorted(item["variants"]) if isinstance(item["variants"], set) else [],
            "roles": sorted(item["roles"]) if isinstance(item["roles"], set) else [],
            "avg_confidence": round(
                float(item["confidence_sum"]) / max(1, int(item["mention_count"])),
                4,
            ),
        }
        for item in paged
    ]
    return PeopleListResponse(count=len(ordered), people=people)


@app.post("/api/people/merge", response_model=PeopleMergeResponse)
async def merge_people(request: PeopleMergeRequest) -> PeopleMergeResponse:
    """Merge one canonical person name into another and re-warm resolver state."""
    from .person_resolver import get_person_resolver

    source = request.source_canonical_name.strip()
    target = request.target_canonical_name.strip()
    if not source or not target:
        raise HTTPException(status_code=422, detail="source_canonical_name and target_canonical_name must be non-empty")

    engine = await asyncio.to_thread(_get_engine)
    storage = engine.storage  # type: ignore[union-attr]

    merged_count = await asyncio.to_thread(
        storage.merge_person_canonical_names,
        source,
        target,
    )

    resolver = get_person_resolver()
    rows = await asyncio.to_thread(storage.list_person_mentions_for_registry)
    await asyncio.to_thread(resolver.warm_from_rows, rows)

    return PeopleMergeResponse(
        source_canonical_name=source,
        target_canonical_name=target,
        merged_count=merged_count,
    )


@app.get("/api/people/mentions", response_model=PersonMentionsResponse)
async def get_people_mentions(
    request: Request,
    canonical_name: str = Query(..., min_length=1),
    source_id: str | None = None,
    source_ids: list[str] | None = Query(default=None),
    min_confidence: float = Query(0.0, ge=0.0, le=1.0),
    limit: int = Query(1000, ge=1, le=50_000),
    offset: int = Query(0, ge=0),
) -> PersonMentionsResponse:
    """Return mention rows for one canonical person."""
    normalized_source_id = source_id.strip() if isinstance(source_id, str) else ""

    raw_source_ids = request.query_params.getlist("source_ids")
    source_ids_provided = "source_ids" in request.query_params
    normalized_source_ids: list[str] = []
    if source_ids_provided:
        for raw in raw_source_ids:
            sid = raw.strip()
            if sid and sid not in normalized_source_ids:
                normalized_source_ids.append(sid)

    if source_ids_provided and not normalized_source_ids and not normalized_source_id:
        return PersonMentionsResponse(canonical_name=canonical_name, count=0, mentions=[])

    effective_source_ids: list[str] | None = None
    union: list[str] = []
    if normalized_source_id:
        union.append(normalized_source_id)
    for sid in normalized_source_ids:
        if sid not in union:
            union.append(sid)
    if union:
        effective_source_ids = union

    engine = await asyncio.to_thread(_get_engine)
    storage = engine.storage  # type: ignore[union-attr]
    try:
        mentions = await asyncio.to_thread(
            storage.get_person_mentions_by_canonical,
            canonical_name,
            source_id=normalized_source_id or None,
            source_ids=effective_source_ids,
            min_confidence=min_confidence,
            limit=limit,
            offset=offset,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    return PersonMentionsResponse(
        canonical_name=canonical_name,
        count=len(mentions),
        mentions=mentions,
    )


@app.delete("/api/people/mentions/{mention_id}", status_code=204)
async def delete_people_mention(mention_id: str) -> None:
    """Delete one person mention and eagerly update resolver registry."""
    from .person_resolver import get_person_resolver

    engine = await asyncio.to_thread(_get_engine)
    storage = engine.storage  # type: ignore[union-attr]

    mention = await asyncio.to_thread(storage.get_person_mention, mention_id)
    await asyncio.to_thread(storage.delete_person_mention, mention_id)

    if mention is not None:
        resolver = get_person_resolver()
        await asyncio.to_thread(
            resolver.remove_mention,
            canonical_name=str(mention.get("canonical_name", "")),
            raw_name=str(mention.get("raw_name", "")),
            source_id=str(mention.get("source_id", "")),
        )


# ---------------------------------------------------------------------------
# Speech-to-Text  (MLX Whisper — fully offline)
# ---------------------------------------------------------------------------


@app.post("/api/transcribe")
async def transcribe_audio(
    audio: UploadFile = File(...),
    sample_rate: int = Form(16000),
):
    """Accept a raw audio chunk from the frontend and return a transcript.

    The frontend records in small chunks (e.g. every 2–3 s) via
    ``MediaRecorder``.  Each chunk is sent as a multipart/form-data upload.
    MLX Whisper transcribes it locally — **no network calls**.

    Returns ``{"transcript": "...", "is_final": true}``.
    The ``is_final`` flag is always ``true`` here because each upload is an
    independent committed chunk; real-time interim display is handled on the
    client by accumulating chunks as they arrive.
    """
    from .transcription import transcribe_audio_bytes

    raw = await audio.read()
    if not raw:
        return JSONResponse({"transcript": "", "is_final": True})

    try:
        text = await asyncio.to_thread(transcribe_audio_bytes, raw, sample_rate)
    except RuntimeError as exc:
        # Model failed to load — tell the client so it can show a clear error
        logger.error("Transcription failed: %s", exc)
        return JSONResponse(
            status_code=503,
            content={"detail": str(exc), "code": "STT_UNAVAILABLE"},
        )
    except Exception:
        logger.exception("Unexpected transcription error")
        return JSONResponse(
            status_code=500,
            content={"detail": "Transcription failed.", "code": "STT_ERROR"},
        )

    return JSONResponse({"transcript": text, "is_final": True})


# ---------------------------------------------------------------------------
# Chat stream: AI SDK UI message stream protocol (SSE)
# ---------------------------------------------------------------------------


def _encode_event(event: QueryEvent) -> Optional[str]:
    """Convert a non-text QueryEvent to an AI SDK UI message stream line.

    ``TextTokenEvent`` is handled separately in the stream generator to manage
    the ``text-start`` / ``text-delta`` / ``text-end`` lifecycle.

    Returns None for events that don't produce a protocol line.
    """
    if isinstance(event, StatusEvent):
        return annotation_status(event.status)
    elif isinstance(event, IntentEvent):
        return annotation_intent(event.intent, event.confidence, event.method)
    elif isinstance(event, SourcesEvent):
        return annotation_sources(event.source_ids)
    elif isinstance(event, CitationListEvent):
        return annotation_citations(event.citations)
    elif isinstance(event, TextTokenEvent):
        # Should not reach here — handled statefully in the consumer loop.
        return None
    elif isinstance(event, ErrorEvent):
        if event.metadata:
            return (
                annotation_error_with_metadata(event.code, event.message, event.metadata)
                + encode_error(event.message)
            )
        return annotation_error(event.code, event.message) + encode_error(event.message)
    elif isinstance(event, FinishEvent):
        return (
            encode_finish_step(event.finish_reason)
            + encode_finish_message(event.finish_reason)
        )
    elif isinstance(event, TraceEvent):
        return encode_annotation([{
            "type": "trace-id",
            "trace_id": event.trace_id,
            "span_id": event.span_id,
        }])
    logger.warning("Unknown event type: %s", type(event).__name__)
    return None


# ---------------------------------------------------------------------------
# Chat (streaming with query_events)
# ---------------------------------------------------------------------------

# Sentinel value to signal end of event stream
_SENTINEL = object()


async def _stream_from_events(
    producer_fn,
    stop_event: threading.Event,
    *,
    keepalive: bool = True,
    request: Request | None = None,
) -> AsyncGenerator[str, None]:
    """Shared AI SDK v6 stream consumer.

    Runs *producer_fn* in a background thread.  *producer_fn(queue, loop)*
    must push ``QueryEvent`` instances into *queue* and finally push
    ``_SENTINEL``.

    Handles keepalive pings, reasoning/text block tracking, error recovery,
    and clean producer shutdown — eliminating the copy-pasted boilerplate
    that was in ``_chat_stream_generator`` and ``_query_stream_generator``.
    """
    queue: asyncio.Queue = asyncio.Queue()
    loop = asyncio.get_running_loop()

    def _wrapped_producer() -> None:
        producer_fn(queue, loop)

    producer_task = loop.run_in_executor(None, _wrapped_producer)

    text_id = "text-0"
    reasoning_id = "reasoning-0"
    in_text_block = False
    in_reasoning_block = False

    try:
        while True:
            if request is not None and await request.is_disconnected():
                stop_event.set()
                return

            try:
                timeout = _KEEPALIVE_INTERVAL_S if keepalive else None
                event = await asyncio.wait_for(queue.get(), timeout=timeout)
            except asyncio.TimeoutError:
                yield ": ping\n\n"
                continue

            if event is _SENTINEL:
                if in_reasoning_block:
                    yield encode_reasoning_end(reasoning_id)
                if in_text_block:
                    yield encode_text_end(text_id)
                break

            if isinstance(event, ThinkingTokenEvent):
                if not in_reasoning_block:
                    yield encode_reasoning_start(reasoning_id)
                    in_reasoning_block = True
                yield encode_reasoning_delta(event.token, reasoning_id)
            elif isinstance(event, TextTokenEvent):
                if in_reasoning_block:
                    yield encode_reasoning_end(reasoning_id)
                    in_reasoning_block = False
                if not in_text_block:
                    yield encode_text_start(text_id)
                    in_text_block = True
                yield encode_text_delta(event.token, text_id)
            else:
                if in_reasoning_block:
                    yield encode_reasoning_end(reasoning_id)
                    in_reasoning_block = False
                if in_text_block:
                    yield encode_text_end(text_id)
                    in_text_block = False
                chunk = _encode_event(event)
                if chunk:
                    yield chunk

    except asyncio.CancelledError:
        stop_event.set()
        logger.info("Stream consumer cancelled")
    except Exception as exc:
        logger.exception("Stream consumer error: %s", exc)
        if in_reasoning_block:
            yield encode_reasoning_end(reasoning_id)
        if in_text_block:
            yield encode_text_end(text_id)
        yield annotation_error("INTERNAL", str(exc))
        yield encode_error(str(exc))
        yield encode_finish_step("error")
        yield encode_finish_message("error")
    finally:
        stop_event.set()
        try:
            await asyncio.wait_for(producer_task, timeout=_PRODUCER_CLEANUP_TIMEOUT_S)
        except (asyncio.TimeoutError, Exception):
            pass


async def _chat_stream_generator(
    request: Request,
    chat_request: ChatRequest,
    stop_event: threading.Event,
) -> AsyncGenerator[str, None]:
    """Generate AI SDK UI message stream lines from query_events.

    Delegates the queue/producer/consumer/cleanup boilerplate to
    ``_stream_from_events``.
    """
    # --- Extract query parameters ---
    last_message = chat_request.messages[-1]
    query_text = last_message.get_text().strip()

    if not query_text:
        yield encode_message_start(f"msg-{uuid.uuid4()}")
        yield encode_error("Empty query")
        yield encode_finish_step("error")
        yield encode_finish_message("error")
        yield encode_done()
        return

    msg_id = f"msg-{uuid.uuid4()}"
    yield encode_message_start(msg_id)

    logger.info("Chat stream: sending initial status (engine may load next)")
    yield annotation_status("Loading RAG engine…")

    source_id = None
    citations_enabled = None
    request_mode = "regular"
    intent_override = None
    if chat_request.data:
        source_id_raw = chat_request.data.get("source_id")
        if isinstance(source_id_raw, str) and source_id_raw.strip():
            source_id = source_id_raw.strip()

        source_ids_raw = chat_request.data.get("source_ids")
        normalized_source_ids: list[str] = []
        if isinstance(source_ids_raw, list):
            for value in source_ids_raw:
                if value is None:
                    continue
                sid = str(value).strip()
                if sid and sid not in normalized_source_ids:
                    normalized_source_ids.append(sid)

        if source_id is None and normalized_source_ids:
            if len(normalized_source_ids) == 1:
                source_id = normalized_source_ids[0]
            else:
                logger.info(
                    "Chat request received %d source_ids; backend retrieval supports single-source pinning only, using all sources.",
                    len(normalized_source_ids),
                )

        if "citations_enabled" in chat_request.data:
            citations_enabled = bool(chat_request.data["citations_enabled"])
        raw_intent = chat_request.data.get("intent_override")
        if raw_intent and raw_intent != "auto":
            intent_override = str(raw_intent)

    frontend_config = (chat_request.model_extra or {}).get("config") or {}
    frontend_model_name = frontend_config.get("modelName", "regular")
    request_mode = _FRONTEND_MODE_MAP.get(frontend_model_name, "regular")
    mc_intent = frontend_config.get("intentOverride")
    if mc_intent and mc_intent != "auto":
        intent_override = str(mc_intent)

    raw_enable_thinking = frontend_config.get("enableThinking")
    if raw_enable_thinking is None:
        if chat_request.data:
            raw_enable_thinking = chat_request.data.get("enable_thinking")
    enable_thinking: Optional[bool] = None
    if raw_enable_thinking is True:
        enable_thinking = True
    elif raw_enable_thinking is False:
        enable_thinking = False

    logger.info("Chat request: frontend model=%r → backend mode=%r, intent_override=%r, enable_thinking=%r", frontend_model_name, request_mode, intent_override, enable_thinking)

    pinned_engine = await asyncio.to_thread(_get_engine, request_mode)

    def _producer(queue: asyncio.Queue, loop: asyncio.AbstractEventLoop) -> None:
        try:
            for event in pinned_engine.query_events(
                query_text,
                source_id=source_id,
                citations_enabled=citations_enabled,
                intent_override=intent_override,
                should_stop=lambda: stop_event.is_set(),
                enable_thinking=enable_thinking,
            ):
                loop.call_soon_threadsafe(queue.put_nowait, event)
        except Exception as exc:
            logger.exception("query_events producer error: %s", exc)
            loop.call_soon_threadsafe(queue.put_nowait, ErrorEvent(
                code="INTERNAL", message=f"{type(exc).__name__}: {exc}",
            ))
            loop.call_soon_threadsafe(queue.put_nowait, FinishEvent(finish_reason="error"))
        finally:
            loop.call_soon_threadsafe(queue.put_nowait, _SENTINEL)

    async for chunk in _stream_from_events(_producer, stop_event, keepalive=True, request=request):
        yield chunk

    yield encode_done()


@app.post("/api/chat")
async def chat(request: Request, chat_request: ChatRequest):
    """Stream a chat response using the AI SDK Data Stream Protocol.

    Uses ``_chat_lock`` to enforce single-user concurrency — only one
    chat stream may be active at a time.
    """
    if _chat_lock.locked():
        logger.warning("Chat: rejected — another request is in progress")
        return JSONResponse(
            status_code=429,
            content=http_error_body("LOCK_BUSY", "Another query is already in progress"),
        )

    if chat_request.data and "source_ids" in chat_request.data:
        source_id_raw = chat_request.data.get("source_id")
        has_source_id = isinstance(source_id_raw, str) and bool(source_id_raw.strip())
        source_ids_raw = chat_request.data.get("source_ids")
        normalized_source_ids: list[str] = []
        if isinstance(source_ids_raw, list):
            for value in source_ids_raw:
                if value is None:
                    continue
                sid = str(value).strip()
                if sid and sid not in normalized_source_ids:
                    normalized_source_ids.append(sid)

        if not has_source_id and not normalized_source_ids:
            return JSONResponse(
                status_code=422,
                content=http_error_body(
                    "NO_SOURCES_SELECTED",
                    "Select at least one source before sending a RAG query.",
                ),
            )

    logger.info("Chat: request received")

    stop_event = threading.Event()

    async def guarded_stream() -> AsyncGenerator[bytes, None]:
        """Stream with disconnect detection, cleanup, and single-user lock."""
        async with _chat_lock:
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

    return StreamingResponse(
        guarded_stream(),
        headers=STREAM_HEADERS,
    )


# ---------------------------------------------------------------------------
# Freeform chat  (non-RAG conversational mode)
# ---------------------------------------------------------------------------


class FreeformMessage(BaseModel):
    role: str  # "user" | "assistant" | "system"
    content: str


class FreeformChatRequest(BaseModel):
    messages: list[FreeformMessage]
    model: str = "regular"  # "regular" | "deep-research"
    enable_thinking: Optional[bool] = None  # tri-state: None=auto, True=on, False=off
    session_id: Optional[str] = None  # frontend session ID for Phoenix session tracking


_FREEFORM_SYSTEM = (
    "You are a knowledgeable and helpful research assistant. "
    "Answer questions clearly and concisely. "
    "You may draw on general knowledge. "
    "When uncertain, say so rather than guessing."
)


async def _freeform_stream_generator(
    request: Request,
    freeform_request: FreeformChatRequest,
    stop_event: threading.Event,
) -> AsyncGenerator[str, None]:
    """Stream freeform chat tokens as plain SSE events (no RAG retrieval)."""
    queue: asyncio.Queue = asyncio.Queue()
    loop = asyncio.get_running_loop()

    # Build message list: inject system message then conversation history.
    messages: list[dict[str, str]] = [
        {"role": "system", "content": _FREEFORM_SYSTEM},
    ]
    for m in freeform_request.messages:
        if m.role in ("user", "assistant"):
            messages.append({"role": m.role, "content": m.content})

    # Pin engine reference before spawning the producer thread (H2).
    freeform_mode = _FRONTEND_MODE_MAP.get(freeform_request.model, "regular")
    pinned_engine = await asyncio.to_thread(_get_engine, freeform_mode)

    # Resolve Phoenix tracer
    _tracer = getattr(pinned_engine, '_tracer', None) if pinned_engine else None

    freeform_request_id = f"ff-{uuid.uuid4().hex[:12]}"
    user_query = freeform_request.messages[-1].content if freeform_request.messages else ""
    model_name = getattr(pinned_engine, '_cfg', None)
    model_name = (model_name.model if model_name and hasattr(model_name, 'model') and model_name.model else freeform_request.model)

    def _producer() -> None:
        try:
            engine = pinned_engine
            gen = engine.ensure_generator()
            # Resolve thinking: explicit True/False from user, or default off for freeform
            use_thinking = freeform_request.enable_thinking if freeform_request.enable_thinking is not None else False
            if use_thinking:
                for event in gen.stream_chat_with_thinking(
                    messages,
                    should_stop=lambda: stop_event.is_set(),
                ):
                    loop.call_soon_threadsafe(queue.put_nowait, event)
            else:
                for token in gen.generate_chat_stream(
                    messages,
                    should_stop=lambda: stop_event.is_set(),
                ):
                    loop.call_soon_threadsafe(queue.put_nowait, {"type": "answer", "text": token})
        except Exception as exc:
            logger.exception("freeform producer error: %s", exc)
            loop.call_soon_threadsafe(queue.put_nowait, exc)
        finally:
            loop.call_soon_threadsafe(queue.put_nowait, _SENTINEL)

    producer_task = loop.run_in_executor(None, _producer)

    # Track token output for tracing
    chunk_event_count = 0
    all_answer_tokens: list[str] = []
    all_thinking_tokens: list[str] = []

    with start_span(
        _tracer,
        "freeform.chat",
        span_kind=SPAN_KIND_CHAIN,
        attributes={
            "input.value": user_query,
            "input.mime_type": "text/plain",
            "freeform.request_id": freeform_request_id,
            "session.id": freeform_request.session_id or freeform_request_id,
            "freeform.model": freeform_request.model,
            "freeform.message_count": len(messages),
            "freeform.enable_thinking": freeform_request.enable_thinking,
        },
    ) as chain_span:
        # Emit trace_id so the frontend can use it for feedback
        trace_id = ""
        span_id = ""
        if chain_span is not None:
            try:
                ctx = chain_span.get_span_context()
                trace_id = format(ctx.trace_id, '032x')
                span_id = format(ctx.span_id, '016x')
            except Exception:
                pass

        if trace_id:
            trace_json = json.dumps({"trace_id": trace_id, "span_id": span_id})
            yield f"event: trace_id\ndata: {trace_json}\n\n"

        with start_span(
            _tracer,
            "freeform.llm.generate",
            span_kind=SPAN_KIND_LLM,
            attributes={
                "input.value": user_query,
                "input.mime_type": "text/plain",
                "llm.model_name": model_name,
                "llm.input_messages": len(messages),
                "freeform.enable_thinking": freeform_request.enable_thinking,
            },
        ) as llm_span:
            set_llm_input_messages(llm_span, messages)

            try:
                while True:
                    if await request.is_disconnected():
                        stop_event.set()
                        mark_span_error(llm_span, "CLIENT_DISCONNECTED")
                        return
                    try:
                        item = await asyncio.wait_for(queue.get(), timeout=_KEEPALIVE_INTERVAL_S)
                    except asyncio.TimeoutError:
                        yield ": ping\n\n"
                        continue
                    if item is _SENTINEL:
                        break
                    if isinstance(item, Exception):
                        err_json = json.dumps({"error": str(item)})
                        yield f"event: error\ndata: {err_json}\n\n"
                        mark_span_error(llm_span, f"{type(item).__name__}: {item}")
                        return
                    # item is a dict {"type": "thinking"|"answer", "text": str}
                    chunk_event_count += 1
                    token_text = item.get("text", "")
                    if item.get("type") == "thinking":
                        all_thinking_tokens.append(token_text)
                    else:
                        all_answer_tokens.append(token_text)

                    if chunk_event_count == 1 and llm_span is not None:
                        try:
                            llm_span.add_event("first_token")
                        except Exception:
                            pass

                    sse_event = "thinking_token" if item.get("type") == "thinking" else "token"
                    token_json = json.dumps({"text": token_text})
                    yield f"event: {sse_event}\ndata: {token_json}\n\n"
            except asyncio.CancelledError:
                stop_event.set()
                mark_span_error(llm_span, "STREAM_CANCELLED")
            finally:
                stop_event.set()
                try:
                    await asyncio.wait_for(producer_task, timeout=_PRODUCER_CLEANUP_TIMEOUT_S)
                except (asyncio.TimeoutError, Exception):
                    pass

            # Finalize LLM span
            answer_text = "".join(all_answer_tokens)
            # Estimate token counts (rough char/4 heuristic)
            prompt_chars = sum(len(m.get("content", "")) for m in messages)
            prompt_token_est = max(1, prompt_chars // 4)
            completion_token_est = max(1, len(answer_text) // 4)

            if llm_span is not None:
                try:
                    llm_span.add_event("stream_complete")
                except Exception:
                    pass

            set_llm_output_message(llm_span, answer_text)
            set_llm_token_counts(
                llm_span,
                prompt_tokens=prompt_token_est,
                completion_tokens=completion_token_est,
                total_tokens=prompt_token_est + completion_token_est,
            )
            set_span_attributes(
                llm_span,
                {
                    "output.value": answer_text[:4096],
                    "output.mime_type": "text/plain",
                    "freeform.output_chars": len(answer_text),
                    "freeform.stream.chunk_events": chunk_event_count,
                    "freeform.thinking_chars": len("".join(all_thinking_tokens)),
                },
            )

        # Finalize chain span
        set_span_attributes(
            chain_span,
            {
                "output.value": f"stream finished: {chunk_event_count} chunks",
                "output.mime_type": "text/plain",
                "freeform.finish_reason": "stop" if chunk_event_count > 0 else "empty",
                "llm.token_count.prompt": prompt_token_est,
                "llm.token_count.completion": completion_token_est,
                "llm.token_count.total": prompt_token_est + completion_token_est,
            },
        )

    yield "event: complete\ndata: {}\n\n"


@app.post("/api/freeform/chat")
async def freeform_chat(request: Request, freeform_request: FreeformChatRequest):
    """Stream a freeform (non-RAG) chat response as plain SSE token events.

    Uses ``_chat_lock`` to enforce single-user concurrency (shared with
    the RAG chat endpoint).

    The SSE stream uses:
      event: token  / data: {"text": "..."}   — one per generated token
      event: complete / data: {}               — stream finished
      event: error  / data: {"error": "..."}  — on failure
    """
    if _chat_lock.locked():
        logger.warning("Freeform chat: rejected — another request is in progress")
        return JSONResponse(
            status_code=429,
            content=http_error_body("LOCK_BUSY", "Another query is already in progress"),
        )

    stop_event = threading.Event()

    async def guarded_stream() -> AsyncGenerator[bytes, None]:
        async with _chat_lock:
            gen = _freeform_stream_generator(request, freeform_request, stop_event)
            try:
                async for chunk in gen:
                    if await request.is_disconnected():
                        stop_event.set()
                        return
                    yield chunk.encode("utf-8")
            except asyncio.CancelledError:
                stop_event.set()
            finally:
                stop_event.set()

    return StreamingResponse(
        guarded_stream(),
        headers={
            "Content-Type": "text/event-stream; charset=utf-8",
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


async def _query_stream_generator(
    request: Request,
    query_request: QueryRequest,
    stop_event: threading.Event,
) -> AsyncGenerator[str, None]:
    """Generate AI SDK UI message stream lines from ``RagEngine.query_events()``."""
    source_id = None
    normalized_source_ids: list[str] = []
    if query_request.source_ids:
        for value in query_request.source_ids:
            if value is None:
                continue
            sid = str(value).strip()
            if sid and sid not in normalized_source_ids:
                normalized_source_ids.append(sid)
    if len(normalized_source_ids) == 1:
        source_id = normalized_source_ids[0]
    elif len(normalized_source_ids) > 1:
        logger.info(
            "Query stream received %d source_ids; backend retrieval supports single-source pinning only, using all sources.",
            len(normalized_source_ids),
        )

    request_mode = _FRONTEND_MODE_MAP.get(query_request.mode, query_request.mode) if query_request.mode else None
    pinned_engine = await asyncio.to_thread(_get_engine, request_mode)

    def _producer(queue: asyncio.Queue, loop: asyncio.AbstractEventLoop) -> None:
        try:
            _intent_override = query_request.intent_override
            if _intent_override == "auto":
                _intent_override = None
            for event in pinned_engine.query_events(
                query_request.query,
                source_id=source_id,
                citations_enabled=query_request.citations_enabled,
                intent_override=_intent_override,
                should_stop=lambda: stop_event.is_set(),
            ):
                loop.call_soon_threadsafe(queue.put_nowait, event)
        except Exception as exc:
            logger.exception("query sse producer error: %s", exc)
            loop.call_soon_threadsafe(queue.put_nowait, ErrorEvent(
                code="INTERNAL", message=f"{type(exc).__name__}: {exc}",
            ))
            loop.call_soon_threadsafe(queue.put_nowait, FinishEvent(finish_reason="error"))
        finally:
            loop.call_soon_threadsafe(queue.put_nowait, _SENTINEL)

    yield encode_message_start(f"msg-{uuid.uuid4()}")

    async for chunk in _stream_from_events(_producer, stop_event, request=request):
        yield chunk

    yield encode_done()


@app.post("/api/query", response_model=QueryResponse)
async def query_endpoint(request: Request, query_request: QueryRequest):
    """Query endpoint with optional AI SDK UI message stream output."""
    source_id = None
    normalized_source_ids: list[str] = []
    if query_request.source_ids is not None:
        for value in query_request.source_ids:
            if value is None:
                continue
            sid = str(value).strip()
            if sid and sid not in normalized_source_ids:
                normalized_source_ids.append(sid)

    if query_request.source_ids is not None and not normalized_source_ids:
        raise HTTPException(
            status_code=422,
            detail="Select at least one source before sending a RAG query.",
        )

    if len(normalized_source_ids) == 1:
        source_id = normalized_source_ids[0]
    elif len(normalized_source_ids) > 1:
        logger.info(
            "Query endpoint received %d source_ids; backend retrieval supports single-source pinning only, using all sources.",
            len(normalized_source_ids),
        )

    # Resolve mode: map frontend names, default to None (keep current engine)
    request_mode = _FRONTEND_MODE_MAP.get(query_request.mode, query_request.mode) if query_request.mode else None

    if not query_request.stream:
        intent_override = query_request.intent_override
        if intent_override == "auto":
            intent_override = None
        engine = await asyncio.to_thread(_get_engine, request_mode)
        result = await asyncio.to_thread(
            engine.query,
            query_request.query,
            source_id=source_id,
            citations_enabled=query_request.citations_enabled,
            intent_override=intent_override,
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

    async def guarded_event_stream() -> AsyncGenerator[bytes, None]:
        async for line in _query_stream_generator(
            request, query_request, stop_event
        ):
            yield line.encode("utf-8")

    return StreamingResponse(
        guarded_event_stream(),
        headers=STREAM_HEADERS,
    )


# ---------------------------------------------------------------------------
# Source management endpoints
# ---------------------------------------------------------------------------


@app.get("/api/sources", response_model=SourceListResponse)
async def list_sources():
    """List all ingested sources with metadata."""
    engine = await asyncio.to_thread(_get_engine)
    storage = engine.storage  # type: ignore[union-attr]

    # Get full details from summaries table
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
            page_offset=detail.get("page_offset") or 1,
            citation_reference=detail.get("citation_reference") or None,
        ))

    # Add any source IDs that have chunks but no summary
    for sid in sorted(all_ids - seen_ids):
        sources.append(SourceInfo(source_id=sid))

    return SourceListResponse(sources=sources)


async def _post_ingest_snapshot(
    engine: object,
    source_id: str,
    file_path: str,
    *,
    citation_reference: Optional[str] = None,
    page_offset: int = 1,
) -> None:
    """Create text snapshot and update summary record with file paths.

    Called after a successful ingest from both ``ingest_source()`` and
    ``upload_source()``.
    """
    from .source_cache import save_snapshot

    storage = engine.storage  # type: ignore[union-attr]

    # Create text snapshot for /content endpoint
    snapshot_path = ""
    try:
        parent_texts = storage.get_parent_texts_by_source(source_id=source_id)
        if parent_texts:
            full_text = "\n\n".join(parent_texts)
            snapshot_path = await asyncio.to_thread(
                save_snapshot, source_id, full_text
            )
    except Exception as snap_exc:
        logger.warning("Failed to create snapshot for %s: %s", source_id, snap_exc)

    resolved_source_path = str(Path(file_path).resolve())

    try:
        storage.persist_source_page_offset(
            source_id=source_id,
            page_offset=page_offset,
            source_path=resolved_source_path,
            snapshot_path=snapshot_path,
            citation_reference=citation_reference,
        )
    except Exception as path_exc:
        logger.warning("Failed to update source paths for %s: %s", source_id, path_exc)


def _to_ner_diagnostics_response(diag: object | None) -> Optional[NERDiagnosticsResponse]:
    """Best-effort conversion from ingest diagnostics objects to API schema."""
    if diag is None:
        return None

    method = getattr(diag, "method", None)
    if method is None:
        return None

    ner_available_raw = getattr(diag, "ner_available", False)
    warning_raw = getattr(diag, "warning", None)
    warning_value = str(warning_raw) if warning_raw is not None else None

    return NERDiagnosticsResponse(
        ner_available=bool(ner_available_raw),
        method=str(method),
        warning=warning_value,
    )


@app.post("/api/sources/ingest", response_model=IngestResponse)
async def ingest_source(request: IngestRequest):
    """Ingest a document (PDF or Markdown) into the RAG store.

    Also creates a text snapshot for the ``/content`` endpoint.
    """

    file_path = request.file_path
    source_id = request.source_id
    page_offset = request.page_offset

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
        engine = await asyncio.to_thread(_get_engine)

        ingest_kwargs = {
            "source_id": source_id,
            "summarize": request.summarize,
            "page_offset": page_offset,
            "citation_reference": request.citation_reference,
        }
        if request.geotag:
            ingest_kwargs["geotag"] = True
        if request.peopletag:
            ingest_kwargs["peopletag"] = True

        # Run ingest in thread (blocking operation)
        result = await asyncio.to_thread(
            engine.ingest,  # type: ignore[union-attr]
            file_path,
            **ingest_kwargs,
        )

        await _post_ingest_snapshot(
            engine,
            source_id,
            file_path,
            citation_reference=request.citation_reference,
            page_offset=page_offset,
        )

        return IngestResponse(
            source_id=result.source_id,
            parents_count=result.parents_count,
            children_count=result.children_count,
            summarized=result.summarized,
            geotag_ner=_to_ner_diagnostics_response(getattr(result, "geotag_ner", None)),
            peopletag_ner=_to_ner_diagnostics_response(getattr(result, "peopletag_ner", None)),
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


# _MAX_UPLOAD_BYTES is defined at top-of-file with other tuning constants.
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


def _source_id_exists(storage: object, source_id: str) -> bool:
    """Return True if a source_id already exists in storage."""
    sid = source_id.strip()
    if not sid:
        return False

    source_ids = storage.list_source_ids()  # type: ignore[attr-defined]
    if sid in source_ids:
        return True

    detail = storage.get_source_detail(sid)  # type: ignore[attr-defined]
    return detail is not None


@app.post("/api/sources/upload", response_model=IngestResponse)
async def upload_source(
    file: UploadFile = File(...),
    source_id: str = Form(""),
    summarize: bool = Form(True),
    geotag: bool = Form(False),
    peopletag: bool = Form(False),
    page_offset: int = Form(1),
    citation_reference: str = Form(""),
):
    """Upload and ingest a document (PDF or Markdown) from the browser.

    Accepts ``multipart/form-data`` with fields:
    - ``file``: The document file (.pdf, .md, .markdown)
    - ``source_id``: Optional custom ID (auto-generated from filename if empty)
    - ``summarize``: Whether to generate a summary (default true)
    """

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

    # --- Reject duplicate source IDs (prevent silent replacement) ---
    engine = await asyncio.to_thread(_get_engine)
    storage = engine.storage  # type: ignore[union-attr]
    exists = await asyncio.to_thread(_source_id_exists, storage, sid)
    if exists:
        return JSONResponse(
            status_code=409,
            content=http_error_body(
                "SOURCE_ALREADY_EXISTS",
                f"A source with ID '{sid}' already exists. Choose a different Source ID or delete the existing source first.",
            ),
        )

    # --- Save uploaded file to data/uploads/ ---
    _UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    dest = _UPLOAD_DIR / f"{sid}{ext}"
    dest.write_bytes(contents)
    logger.info("Saved upload to %s (%d bytes)", dest, len(contents))

    try:

        ingest_kwargs = {
            "source_id": sid,
            "summarize": summarize,
            "page_offset": page_offset,
            "citation_reference": citation_reference.strip() or None,
        }
        if geotag:
            ingest_kwargs["geotag"] = True
        if peopletag:
            ingest_kwargs["peopletag"] = True

        # Run ingest in thread (blocking operation)
        result = await asyncio.to_thread(
            engine.ingest,  # type: ignore[union-attr]
            str(dest),
            **ingest_kwargs,
        )

        await _post_ingest_snapshot(
            engine,
            sid,
            str(dest),
            citation_reference=citation_reference.strip() or None,
            page_offset=page_offset,
        )

        return IngestResponse(
            source_id=result.source_id,
            parents_count=result.parents_count,
            children_count=result.children_count,
            summarized=result.summarized,
            geotag_ner=_to_ner_diagnostics_response(getattr(result, "geotag_ner", None)),
            peopletag_ner=_to_ner_diagnostics_response(getattr(result, "peopletag_ner", None)),
        )

    except Exception as exc:
        logger.exception("Upload ingest failed for %s: %s", sid, exc)
        # Clean up the uploaded file on failure
        try:
            dest.unlink(missing_ok=True)
        except Exception:
            logger.warning("Failed to clean up uploaded file %s", dest, exc_info=True)
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
    from .person_resolver import get_person_resolver

    try:
        engine = await asyncio.to_thread(_get_engine)
        storage = engine.storage  # type: ignore[union-attr]

        # Get snapshot path before deleting from storage
        detail = storage.get_source_detail(source_id)
        snapshot_path = detail.get("snapshot_path", "") if detail else ""

        # Delete from storage (children, parents, summary)
        deleted = await asyncio.to_thread(storage.delete_source, source_id)

        # Keep in-memory resolver registry aligned with persisted table rows.
        try:
            resolver = get_person_resolver()
            rows = await asyncio.to_thread(storage.list_person_mentions_for_registry)
            await asyncio.to_thread(resolver.warm_from_rows, rows)
        except Exception:
            logger.warning("Failed to re-warm person resolver after source delete", exc_info=True)

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


def _coerce_positive_int(value: object) -> Optional[int]:
    if value is None:
        return None
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed >= 1 else None


@app.get("/api/sources/{source_id}/content", response_model=SourceContentResponse)
async def get_source_content(source_id: str):
    """Get the full text content of a source document.

    Resolution order: cached snapshot → original file → assembled parent texts → 404.
    The snapshot is preferred because it is built from the same parent-chunk texts
    stored in the vector DB, ensuring citation highlight needles match exactly.
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

        result = await asyncio.to_thread(
            resolve_content,
            source_path,
            snapshot_path,
            prefer_snapshot=True,
        )
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


@app.get("/api/sources/{source_id}/chunks", response_model=ChunkBatchResponse)
async def get_chunk_batch(source_id: str, ids: str):
    """Return an array of chunks in a single request.
    
    Tolerates missing chunks; returns only the ones found successfully.
    """
    chunk_ids = [cid.strip() for cid in ids.split(",") if cid.strip()]
    if not chunk_ids:
        return ChunkBatchResponse(chunks=[])

    try:
        engine = await asyncio.to_thread(_get_engine)
        storage = engine.storage  # type: ignore[union-attr]

        # Get source detail for format/source_path
        detail = storage.get_source_detail(source_id)
        sp = detail.get("source_path", "") if detail else ""
        fmt = _detect_format(sp or None)

        children = storage.get_children_by_ids(chunk_ids)

        chunks = []
        for cid in chunk_ids:
            child = children.get(cid)
            if not child:
                continue
            
            meta = child.get("metadata", {})
            if isinstance(meta, dict) and meta.get("source_id") != source_id:
                continue

            page_num_raw = meta.get("page_number") if isinstance(meta, dict) else None
            page_number = _coerce_positive_int(page_num_raw)
            start_page = _coerce_positive_int(meta.get("start_page")) if isinstance(meta, dict) else None
            end_page = _coerce_positive_int(meta.get("end_page")) if isinstance(meta, dict) else None
            display_page = str(meta.get("display_page", "")) if isinstance(meta, dict) and meta.get("display_page") else None
            header_path = str(meta.get("header_path", "")) if isinstance(meta, dict) else ""

            chunks.append(ChunkBatchItem(
                source_id=source_id,
                chunk_id=cid,
                chunk_text=str(child.get("text", "")),
                page_number=page_number,
                start_page=start_page,
                end_page=end_page,
                display_page=display_page,
                header_path=header_path,
                format=fmt,
                source_path=sp or None,
            ))

        return ChunkBatchResponse(chunks=chunks)

    except Exception as exc:
        logger.exception("Chunk batch failed for %s: %s", source_id, exc)
        return JSONResponse(
            status_code=500,
            content=http_error_body("INTERNAL", f"Chunk batch failed: {exc}"),
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
        page_number = _coerce_positive_int(page_num_raw)
        start_page = _coerce_positive_int(meta.get("start_page")) if isinstance(meta, dict) else None
        end_page = _coerce_positive_int(meta.get("end_page")) if isinstance(meta, dict) else None

        return ChunkDetailResponse(
            source_id=source_id,
            chunk_id=chunk_id,
            chunk_text=str(child.get("text", "")),
            parent_text=parent_text,
            page_number=page_number,
            start_page=start_page,
            end_page=end_page,
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


# ---------------------------------------------------------------------------
# User Feedback Annotations (Phoenix)
# ---------------------------------------------------------------------------


class FeedbackRequest(BaseModel):
    span_id: str
    trace_id: str
    label: str  # e.g. "👍", "👎"
    score: Optional[float] = None  # 1.0 = positive, 0.0 = negative
    comment: Optional[str] = None


@app.post("/api/feedback")
async def post_feedback(feedback: FeedbackRequest):
    """Log user feedback as a Phoenix annotation on a specific span."""
    ok = annotate_span_feedback(
        span_id=feedback.span_id,
        trace_id=feedback.trace_id,
        label=feedback.label,
        score=feedback.score,
        comment=feedback.comment,
    )
    if ok:
        return {"status": "ok", "message": "Feedback recorded"}
    return JSONResponse(
        status_code=503,
        content={"status": "unavailable", "message": "Phoenix annotation unavailable (arize-phoenix not installed or Phoenix unreachable)"},
    )
