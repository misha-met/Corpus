"""Decoupled RAG engine — reusable from CLI, Chainlit, or any other frontend.

This module extracts the full retrieval-augmented generation pipeline from
``cli.py`` into a stateful ``RagEngine`` class that returns structured results
instead of printing to stdout.
"""

from __future__ import annotations

import concurrent.futures
import gc
import json
import logging
import os
import re
import textwrap
import time as _time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Optional

from .config import CITATIONS_ENABLED_DEFAULT, ModelConfig, select_mode_config
from .generation import build_messages
from .generator import (
    BudgetPackResult,
    GenerationConfig,
    MlxGenerator,
    count_tokens,
    enforce_token_budget,
)
from .ingest import ingest_file_to_storage
from .intent import Intent, IntentClassifier, IntentResult, is_low_information_query
from .latency import LatencyProfiler
from .metrics import (
    BudgetMetrics,
    RetrievalMetrics,
    format_metrics_summary,
    log_metrics,
)
from .retrieval import (
    RetrievalEngine,
    RetrievalResult,
    build_source_legend,
    format_context_with_citations,
)
from .query_events import (
    ErrorEvent,
    FinishEvent,
    IntentEvent,
    QueryEvent,
    SourcesEvent,
    StatusEvent,
    TextTokenEvent,
)
from .storage import StorageConfig, StorageEngine

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Result data classes
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class IngestResult:
    """Structured result from a document ingestion."""

    parents_count: int
    children_count: int
    source_id: str
    summarized: bool


@dataclass(frozen=True)
class QueryResult:
    """Structured result from a RAG query."""

    answer: str
    intent: IntentResult
    citations_enabled: bool
    source_ids: list[str] = field(default_factory=list)
    retrieval_metrics: Optional[RetrievalMetrics] = None
    budget_metrics: Optional[BudgetMetrics] = None
    context: str = ""
    config: Optional[ModelConfig] = None
    raw_answer: str = ""


# ---------------------------------------------------------------------------
# Output sanitisation helpers (moved from cli.py)
# ---------------------------------------------------------------------------

_INSTRUCTION_PATTERNS = [
    re.compile(r"^\s*Important:.*$", re.IGNORECASE | re.MULTILINE),
    re.compile(r"^\s*Grounding rule:.*$", re.IGNORECASE | re.MULTILINE),
    re.compile(r"^\s*Base your answer.*$", re.IGNORECASE | re.MULTILINE),
    re.compile(r"^\s*Do not substitute.*$", re.IGNORECASE | re.MULTILINE),
    re.compile(r"^\s*Task:.*$", re.IGNORECASE | re.MULTILINE),
    re.compile(r"^\s*Format:.*$", re.IGNORECASE | re.MULTILINE),
    re.compile(r"^\s*Tone:.*$", re.IGNORECASE | re.MULTILINE),
]

_REPETITION_PATTERN = re.compile(
    r"(\d+\.\s+[A-Z][^.]{10,50}\.?)(\s*\1)+", re.IGNORECASE
)

_CHATTER_PHRASES = [
    "answer ends here",
    "this answer acknowledges",
    "this answer was generated",
    "this response reflects",
    "this response was",
    "this explanation covers",
    "this summary covers",
    "this analysis covers",
    "i hope this helps",
    "let me know if",
    "feel free to ask",
    "is there anything else",
    "the texts leave open the possibility",
    "future research may",
    "it remains to be seen",
    "only time will tell",
]

_RECURSIVE_DENIAL = re.compile(
    r"\n+\s*The provided context does not contain sufficient information[^\n]*$",
    re.IGNORECASE,
)

_INCOMPLETE_ENDING = re.compile(
    r"\b(the|a|an|to|of|in|for|and|or|but|is|are|was|were|that|this|with)\s*$",
    re.IGNORECASE,
)

_EXPANSION_TERMS: dict[Intent, list[str]] = {
    Intent.OVERVIEW: [],
    Intent.SUMMARIZE: ["main argument", "thesis", "conclusion", "key points"],
    Intent.EXPLAIN: [],
    Intent.ANALYZE: [
        "criticism",
        "critique",
        "debate",
        "objection",
        "response",
        "controversy",
    ],
    Intent.COMPARE: ["compare", "contrast", "difference", "similarity"],
    Intent.CRITIQUE: [
        "criticism",
        "critique",
        "debate",
        "objection",
        "weakness",
        "strength",
    ],
    Intent.FACTUAL: [],
    Intent.COLLECTION: [],
}


def _strip_chatter(text: str) -> str:
    """Remove trailing chatter phrases near end of text (last 20%)."""
    if not text:
        return text
    result = text
    if len(result) > 200:
        result = _RECURSIVE_DENIAL.sub("", result).rstrip()
    lowered = result.lower()
    for phrase in _CHATTER_PHRASES:
        idx = lowered.rfind(phrase)
        if idx != -1 and idx > len(result) * 0.8:
            result = result[:idx].rstrip()
            lowered = result.lower()
    return result


def _dedupe_repeated_blocks(text: str) -> str:
    """Remove duplicated halves when model repeats itself (>85% similarity)."""
    if not text or len(text) < 200:
        return text
    import difflib

    mid = len(text) // 2
    first_half, second_half = text[:mid].strip(), text[mid:].strip()
    if not first_half or not second_half:
        return text
    normalize = lambda s: re.sub(r"\s+", " ", s).strip().lower()
    if (
        difflib.SequenceMatcher(
            None, normalize(first_half), normalize(second_half)
        ).ratio()
        >= 0.85
    ):
        return first_half
    return text


def sanitize_output(text: str) -> str:
    """Post-process LLM output: remove instruction leakage, chatter, and incomplete sentences."""
    if not text:
        return text
    result = text
    for pattern in _INSTRUCTION_PATTERNS:
        result = pattern.sub("", result)
    result = _strip_chatter(result)
    result = _dedupe_repeated_blocks(result)
    result = _REPETITION_PATTERN.sub(r"\1", result)

    lines = result.rstrip().split("\n")
    if lines:
        last_line = lines[-1]
        if _INCOMPLETE_ENDING.search(last_line) or (
            last_line and last_line[-1] not in ".!?:"
        ):
            for sentinel in [". ", ".\n", "! ", "!\n", "? ", "?\n"]:
                truncate_pos = result.rfind(sentinel)
                if truncate_pos > len(result) * 0.5:
                    result = result[: truncate_pos + 1]
                    break
    return result.strip()


def _expand_query(query: str, intent: Intent) -> str:
    """Append intent-specific expansion terms to query (feature-flagged)."""
    terms = _EXPANSION_TERMS.get(intent, [])
    return f"{query} {' '.join(terms)}" if terms else query


def _dedupe_context(texts: Iterable[str]) -> str:
    seen: set[str] = set()
    unique_texts = []
    for text in texts:
        cleaned = text.strip()
        if cleaned and cleaned not in seen:
            seen.add(cleaned)
            unique_texts.append(cleaned)
    return "\n\n".join(unique_texts)


def _enable_offline_if_cached(config: ModelConfig) -> None:
    """Enable HF offline mode if all required models are already cached."""
    try:
        from huggingface_hub.constants import HF_HUB_CACHE

        cache_dir = Path(HF_HUB_CACHE)
    except ImportError:
        cache_dir = Path.home() / ".cache" / "huggingface" / "hub"

    if not cache_dir.exists():
        logger.debug("HF cache dir %s does not exist; staying online", cache_dir)
        return

    required_models = [config.llm_model, config.embedding_model, config.reranker_model]
    for model_id in required_models:
        cache_folder = f"models--{model_id.replace('/', '--')}"
        model_cache = cache_dir / cache_folder
        if not model_cache.exists():
            logger.debug("Model %s not cached; staying online", model_id)
            return
        snapshots_dir = model_cache / "snapshots"
        if not snapshots_dir.is_dir():
            return
        snapshot_dirs = [d for d in snapshots_dir.iterdir() if d.is_dir()]
        if not snapshot_dirs:
            return
        latest = max(snapshot_dirs, key=lambda d: d.stat().st_mtime)
        weight_exts = {".safetensors", ".bin", ".gguf", ".npz"}
        has_weights = any(
            f.suffix in weight_exts or f.name == "config.json"
            for f in latest.iterdir()
            if f.is_file() or f.is_symlink()
        )
        if not has_weights:
            return

    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    logger.info("All models cached — enabled offline mode")


def _release_mlx_cache() -> None:
    """Force MLX to release cached Metal buffers."""
    try:
        import mlx.core as mx

        mx.clear_cache()
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Engine configuration
# ---------------------------------------------------------------------------


@dataclass
class RagEngineConfig:
    """User-facing configuration for the RAG engine."""

    lance_dir: str = "data/lance"
    collection: str = "child_chunks"
    mode: Optional[str] = None
    model: Optional[str] = None
    fts_rebuild_policy: str = "deferred"
    fts_rebuild_batch_size: int = 0
    citations_enabled: Optional[bool] = None
    enable_query_expansion: bool = False
    intent_confidence_threshold: float = 0.6
    llm_fallback: bool = True
    llm_fallback_threshold: float = 0.70
    intent_model: str = "mlx-community/LFM2-8B-A1B-4bit"
    verbose: bool = False
    latency: bool = False


# ---------------------------------------------------------------------------
# Main Engine
# ---------------------------------------------------------------------------


class RagEngine:
    """Decoupled RAG engine that can be consumed by any frontend.

    Lifecycle::

        engine = RagEngine(config)   # loads models, opens storage
        result = engine.query("...")  # returns QueryResult
        engine.ingest("file.pdf", source_id="paper")  # returns IngestResult
        sources = engine.list_sources()
        engine.close()               # optional cleanup
    """

    def __init__(
        self,
        config: Optional[RagEngineConfig] = None,
        *,
        on_status: Optional[Callable[[str], None]] = None,
    ) -> None:
        self._cfg = config or RagEngineConfig()
        self._on_status = on_status or (lambda msg: None)

        # Suppress noisy loggers unless verbose
        if not self._cfg.verbose:
            for noisy in (
                "httpx",
                "huggingface_hub",
                "lancedb",
                "urllib3",
                "sentence_transformers",
                "filelock",
                "fsspec",
            ):
                logging.getLogger(noisy).setLevel(logging.WARNING)

        # Select configuration
        self._on_status("Selecting mode configuration...")
        self._model_config: ModelConfig = select_mode_config(
            manual_mode=self._cfg.mode
        )
        _enable_offline_if_cached(self._model_config)

        # Open storage
        self._on_status("Opening LanceDB storage...")
        self._storage = StorageEngine(
            StorageConfig(
                lance_dir=Path(self._cfg.lance_dir),
                lance_table=self._cfg.collection,
                fts_rebuild_policy=self._cfg.fts_rebuild_policy,
                fts_rebuild_batch_size=self._cfg.fts_rebuild_batch_size,
            )
        )

        # Lazy-initialised models (loaded on first use)
        self._embedding_model: Any = None
        self._reranker: Any = None
        self._generator: Optional[MlxGenerator] = None

    # -- properties --------------------------------------------------------

    @property
    def model_config(self) -> ModelConfig:
        return self._model_config

    @property
    def storage(self) -> StorageEngine:
        return self._storage

    # -- model loading -----------------------------------------------------

    def _ensure_embedding_model(self) -> Any:
        if self._embedding_model is not None:
            return self._embedding_model
        logger.info("_ensure_embedding_model: loading...")
        self._on_status("Loading embedding model...")
        from sentence_transformers import SentenceTransformer

        self._embedding_model = SentenceTransformer(
            self._model_config.embedding_model,
            device=self._model_config.embedding_device,
        )
        logger.info("_ensure_embedding_model: done")
        return self._embedding_model

    def _ensure_reranker(self) -> Any:
        if self._reranker is not None:
            return self._reranker
        logger.info("_ensure_reranker: loading...")
        self._on_status("Loading reranker model...")
        from .reranker import JinaRerankerMLX

        self._reranker = JinaRerankerMLX(model_id=self._model_config.reranker_model)
        logger.info("_ensure_reranker: done")
        return self._reranker

    def _ensure_generator(self) -> MlxGenerator:
        if self._generator is not None:
            return self._generator
        model_id = self._cfg.model or self._model_config.llm_model
        logger.info("_ensure_generator: loading LLM %s...", model_id.split("/")[-1])
        self._on_status(f"Loading LLM ({model_id.split('/')[-1]})...")
        try:
            import mlx.core as mx

            mx.set_cache_limit(0)
        except Exception:
            pass
        self._generator = MlxGenerator(model_id)
        logger.info("_ensure_generator: done")
        return self._generator

    def load_retrieval_models(self) -> None:
        """Pre-load embedding + reranker in parallel (call once at startup)."""
        self._on_status("Loading retrieval models (embedding + reranker)...")
        from sentence_transformers import SentenceTransformer

        from .reranker import JinaRerankerMLX

        embed_result: list[Any] = [None]
        reranker_result: list[Any] = [None]

        def _load_embed() -> None:
            embed_result[0] = SentenceTransformer(
                self._model_config.embedding_model,
                device=self._model_config.embedding_device,
            )

        def _load_reranker() -> None:
            reranker_result[0] = JinaRerankerMLX(
                model_id=self._model_config.reranker_model
            )

        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
            pool.submit(_load_embed)
            pool.submit(_load_reranker)
            pool.shutdown(wait=True)

        self._embedding_model = embed_result[0]
        self._reranker = reranker_result[0]
        self._on_status("Retrieval models loaded.")

    def _release_retrieval_models(self) -> None:
        """Free embedding + reranker memory before LLM generation."""
        self._embedding_model = None
        self._reranker = None
        gc.collect()
        _release_mlx_cache()
        logger.debug(
            "Released reranker and embedding model to free memory for LLM generation"
        )

    # -- public API --------------------------------------------------------

    def list_sources(self) -> list[str]:
        """Return sorted list of ingested source IDs."""
        return self._storage.list_source_ids()

    def ingest(
        self,
        file_path: str,
        *,
        source_id: str,
        page_number: Optional[int] = None,
        summarize: bool = True,
    ) -> IngestResult:
        """Ingest a document (PDF or Markdown) into the RAG store."""
        self._on_status(f"Ingesting {Path(file_path).name}...")
        embedding_model = self._ensure_embedding_model()

        generator: Optional[MlxGenerator] = None
        if summarize:
            generator = self._ensure_generator()

        parents_count, children_count = ingest_file_to_storage(
            file_path,
            source_id=source_id,
            page_number=page_number,
            storage=self._storage,
            embedding_model=embedding_model,
            summarize=summarize,
            summary_generator=generator,
        )

        self._on_status(
            f"Ingested {parents_count} parents, {children_count} children."
        )
        return IngestResult(
            parents_count=parents_count,
            children_count=children_count,
            source_id=source_id,
            summarized=summarize,
        )

    def query(
        self,
        query_text: str,
        *,
        source_id: Optional[str] = None,
        intent_override: Optional[str] = None,
        citations_enabled: Optional[bool] = None,
        no_generate: bool = False,
        enable_query_expansion: Optional[bool] = None,
    ) -> QueryResult:
        """Execute the full RAG pipeline and return a structured result.

        Parameters
        ----------
        query_text : str
            The user's question.
        source_id : str | None
            Restrict retrieval to a specific source document.
        intent_override : str | None
            Force a specific intent (e.g. ``"analyze"``).
        citations_enabled : bool | None
            Override citation mode. ``None`` uses the engine default.
        no_generate : bool
            If True, skip LLM generation and return only retrieved context.
        enable_query_expansion : bool | None
            Override query expansion setting.
        """
        config = self._model_config
        model_id = self._cfg.model or config.llm_model
        profiler = LatencyProfiler(enabled=self._cfg.latency)
        profiler.start_wall()

        # -- resolve citation mode -----------------------------------------
        if citations_enabled is not None:
            cite = citations_enabled
        elif self._cfg.citations_enabled is not None:
            cite = self._cfg.citations_enabled
        else:
            cite = CITATIONS_ENABLED_DEFAULT
        logger.info(
            "Citations mode: %s",
            "ENABLED (Academic Mode)" if cite else "DISABLED (Casual Mode)",
        )

        expansion = (
            enable_query_expansion
            if enable_query_expansion is not None
            else self._cfg.enable_query_expansion
        )

        # -- load retrieval models -----------------------------------------
        self._on_status("Preparing retrieval models...")
        embedding_model = self._ensure_embedding_model()
        reranker = self._ensure_reranker()

        retrieval_engine = RetrievalEngine(
            storage=self._storage,
            embedding_model=embedding_model,
            reranker=reranker,
            config=config,
        )

        # -- intent classification -----------------------------------------
        self._on_status("Classifying intent...")
        intent_result: Optional[IntentResult] = None
        if intent_override:
            intent_map = {
                "overview": Intent.OVERVIEW,
                "summarize": Intent.SUMMARIZE,
                "explain": Intent.EXPLAIN,
                "analyze": Intent.ANALYZE,
                "compare": Intent.COMPARE,
                "critique": Intent.CRITIQUE,
                "factual": Intent.FACTUAL,
                "collection": Intent.COLLECTION,
            }
            intent_result = IntentResult(
                intent=intent_map[intent_override], confidence=1.0, method="manual"
            )
        else:
            with profiler.span("Intent classification"):
                llm_fallback_enabled = not no_generate and self._cfg.llm_fallback
                llm_model_id = self._cfg.intent_model if llm_fallback_enabled else None
                classifier = IntentClassifier(
                    confidence_threshold=self._cfg.intent_confidence_threshold,
                    llm_model_id=llm_model_id,
                    llm_fallback_threshold=self._cfg.llm_fallback_threshold,
                    eager_load_llm=False,
                )
                intent_result = classifier.classify(query_text)
                del classifier
                gc.collect()
                _release_mlx_cache()

        logger.info(
            "Classified intent: %s (confidence=%.2f, method=%s)",
            intent_result.intent.value,
            intent_result.confidence,
            intent_result.method,
        )

        # -- query expansion -----------------------------------------------
        search_query = query_text
        if expansion:
            search_query = _expand_query(query_text, intent_result.intent)

        extra_instructions: Optional[str] = None
        bypass_retrieval = is_low_information_query(query_text) and not no_generate

        # -- retrieval paths -----------------------------------------------
        context = ""
        results: list[RetrievalResult] = []
        source_ids: list[str] = []
        parent_texts: list[str] = []
        generator: Optional[MlxGenerator] = None

        if bypass_retrieval:
            self._on_status("Query is unclear — skipping retrieval...")
            logger.info(
                "Skipping retrieval for low-information query; delegating to generation model"
            )
            if cite:
                logger.info(
                    "Auto-disabling citations: no retrieval context for unclear query"
                )
                cite = False
            extra_instructions = (
                "The user query is unclear or nonsensical. Ask a concise clarifying "
                "question first, and optionally suggest 2-3 concrete ways they can rephrase it."
            )

        elif intent_result.intent == Intent.COLLECTION:
            self._on_status("Fetching collection summaries...")
            context, source_ids, cite = self._handle_collection(
                config=config,
                model_id=model_id,
                no_generate=no_generate,
                citations_enabled=cite,
            )
            if context is None:
                # No docs — return early
                return QueryResult(
                    answer="No documents found in the database.",
                    intent=intent_result,
                    citations_enabled=False,
                    config=config,
                )

        elif intent_result.intent == Intent.SUMMARIZE and not source_id:
            self._on_status("Checking multi-document summarise path...")
            multi_result = self._handle_multi_doc_summarize(
                config=config,
                model_id=model_id,
                no_generate=no_generate,
                citations_enabled=cite,
                profiler=profiler,
                retrieval_engine=retrieval_engine,
                search_query=search_query,
                source_id=source_id,
            )
            context = multi_result["context"]
            results = multi_result["results"]
            source_ids = multi_result["source_ids"]
            parent_texts = multi_result["parent_texts"]
            cite = multi_result["citations_enabled"]
            extra_instructions = multi_result.get("extra_instructions")

        else:
            self._on_status("Searching knowledge base...")
            with profiler.span("Retrieval (hybrid search + rerank)"):
                results = retrieval_engine.search(
                    search_query, source_id=source_id
                )
            source_ids = sorted(
                {
                    r.metadata.get("source_id")
                    for r in results
                    if r.metadata.get("source_id")
                }
            )
            parent_texts = [r.parent_text for r in results if r.parent_text]

        retrieval_metrics: Optional[RetrievalMetrics] = (
            results[0].metrics if results and results[0].metrics else None
        )

        # -- release retrieval models for LLM headroom ---------------------
        with profiler.span("Memory cleanup (gc)"):
            self._release_retrieval_models()

        # -- token budget packing ------------------------------------------
        budget_metrics: Optional[BudgetMetrics] = None
        source_legend: Optional[str] = None
        result_metadatas: list[dict] = (
            [r.metadata for r in results if r.parent_text]
            if parent_texts and results
            else []
        )

        if not no_generate and parent_texts:
            if generator is None:
                generator = self._ensure_generator()

            self._on_status("Packing token budget...")
            with profiler.span("Budget packing"):
                pack_start = _time.perf_counter()
                pack_result = enforce_token_budget(
                    docs=parent_texts,
                    max_tokens=config.retrieval_budget,
                    tokenizer=generator.tokenizer,
                    consecutive_fail_threshold=3,
                    allow_truncation=True,
                    log=logger,
                )
                pack_time_ms = (_time.perf_counter() - pack_start) * 1000

            packed_metadatas = [
                result_metadatas[i]
                for i in pack_result.packed_indices
                if i < len(result_metadatas)
            ]

            if cite and packed_metadatas:
                context, source_mapping = format_context_with_citations(
                    texts=pack_result.packed_docs, metadatas=packed_metadatas
                )
                source_legend = build_source_legend(source_mapping)
            elif cite:
                logger.info(
                    "Auto-disabling citations: packed context missing metadata"
                )
                cite = False
            else:
                context = "\n\n".join(pack_result.packed_docs)

            budget_metrics = BudgetMetrics(
                budget_tokens=config.retrieval_budget,
                used_tokens=pack_result.used_tokens,
                utilization_pct=(
                    100 * pack_result.used_tokens / config.retrieval_budget
                    if config.retrieval_budget > 0
                    else 0
                ),
                avg_doc_tokens=(
                    pack_result.used_tokens / len(pack_result.packed_docs)
                    if pack_result.packed_docs
                    else 0
                ),
                docs_packed=len(pack_result.packed_docs),
                docs_skipped=pack_result.skipped_count,
                docs_truncated=pack_result.truncated_count,
            )

        elif parent_texts:
            if cite and result_metadatas:
                context, source_mapping = format_context_with_citations(
                    texts=parent_texts, metadatas=result_metadatas
                )
                source_legend = build_source_legend(source_mapping)
            elif cite:
                cite = False
            else:
                context = _dedupe_context(parent_texts)

        # -- no-generate path (context only) --------------------------------
        if no_generate:
            profiler.end_wall()
            return QueryResult(
                answer="",
                intent=intent_result,
                citations_enabled=cite,
                source_ids=source_ids,
                retrieval_metrics=retrieval_metrics,
                budget_metrics=budget_metrics,
                context=context,
                config=config,
            )

        # -- build prompt ---------------------------------------------------
        self._on_status("Building prompt...")
        with profiler.span("Build prompt / messages"):
            messages = build_messages(
                context,
                query_text,
                intent=intent_result.intent,
                extra_instructions=extra_instructions,
                citations_enabled=cite,
                source_legend=source_legend,
                mode=config.mode,
            )

        # -- generate -------------------------------------------------------
        if generator is None:
            generator = self._ensure_generator()

        gen_config = GenerationConfig(
            max_tokens=600 if cite else 1200,
            context_window=config.context_window,
        )

        self._on_status("Generating answer...")
        with profiler.span("LLM generation"):
            raw_answer = generator.generate_chat(messages, config=gen_config)

        with profiler.span("Output sanitisation"):
            answer = sanitize_output(raw_answer)

        profiler.end_wall()

        return QueryResult(
            answer=answer,
            intent=intent_result,
            citations_enabled=cite,
            source_ids=source_ids,
            retrieval_metrics=retrieval_metrics,
            budget_metrics=budget_metrics,
            context=context,
            config=config,
            raw_answer=raw_answer,
        )

    # -- streaming query (event generator) ----------------------------------

    def query_events(
        self,
        query_text: str,
        *,
        source_id: Optional[str] = None,
        intent_override: Optional[str] = None,
        citations_enabled: Optional[bool] = None,
        enable_query_expansion: Optional[bool] = None,
        should_stop: Optional[Callable[[], bool]] = None,
    ) -> Iterable[QueryEvent]:
        """Execute the RAG pipeline yielding structured events.

        This is the streaming counterpart of :meth:`query`.  It yields
        :class:`QueryEvent` objects at each pipeline stage, enabling the
        API layer to stream status updates and text tokens to the client.

        The existing :meth:`query` method is unchanged.

        Parameters
        ----------
        query_text : str
            The user's question.
        source_id : str | None
            Restrict retrieval to a specific source document.
        intent_override : str | None
            Force a specific intent.
        citations_enabled : bool | None
            Override citation mode.
        enable_query_expansion : bool | None
            Override query expansion setting.
        should_stop : callable | None
            Called periodically; if it returns True, generation is
            aborted and a ``STREAM_CANCELLED`` error event is yielded.
        """
        _stop = should_stop or (lambda: False)

        try:
            yield from self._query_events_impl(
                query_text,
                source_id=source_id,
                intent_override=intent_override,
                citations_enabled=citations_enabled,
                enable_query_expansion=enable_query_expansion,
                should_stop=_stop,
            )
        except Exception as exc:
            logger.exception("query_events error: %s", exc)
            yield ErrorEvent(code="INTERNAL", message=f"{type(exc).__name__}: {exc}")
            yield FinishEvent(finish_reason="error")

    def _query_events_impl(
        self,
        query_text: str,
        *,
        source_id: Optional[str],
        intent_override: Optional[str],
        citations_enabled: Optional[bool],
        enable_query_expansion: Optional[bool],
        should_stop: Callable[[], bool],
    ) -> Iterable[QueryEvent]:
        """Internal implementation of query_events (unwrapped from error handling)."""
        logger.info("query_events_impl: started")
        config = self._model_config
        model_id = self._cfg.model or config.llm_model

        # -- resolve citation mode -----------------------------------------
        if citations_enabled is not None:
            cite = citations_enabled
        elif self._cfg.citations_enabled is not None:
            cite = self._cfg.citations_enabled
        else:
            cite = CITATIONS_ENABLED_DEFAULT

        expansion = (
            enable_query_expansion
            if enable_query_expansion is not None
            else self._cfg.enable_query_expansion
        )

        # -- load retrieval models -----------------------------------------
        logger.info("query_events_impl: yielding 'Preparing retrieval models', then loading embedding")
        yield StatusEvent(status="Preparing retrieval models...")
        embedding_model = self._ensure_embedding_model()
        logger.info("query_events_impl: embedding ready, loading reranker")
        reranker = self._ensure_reranker()
        logger.info("query_events_impl: reranker ready")

        if should_stop():
            yield ErrorEvent(code="STREAM_CANCELLED", message="Cancelled")
            yield FinishEvent(finish_reason="error")
            return

        retrieval_engine = RetrievalEngine(
            storage=self._storage,
            embedding_model=embedding_model,
            reranker=reranker,
            config=config,
        )

        # -- intent classification -----------------------------------------
        yield StatusEvent(status="Classifying intent...")
        intent_result: Optional[IntentResult] = None
        if intent_override:
            intent_map = {
                "overview": Intent.OVERVIEW,
                "summarize": Intent.SUMMARIZE,
                "explain": Intent.EXPLAIN,
                "analyze": Intent.ANALYZE,
                "compare": Intent.COMPARE,
                "critique": Intent.CRITIQUE,
                "factual": Intent.FACTUAL,
                "collection": Intent.COLLECTION,
            }
            intent_result = IntentResult(
                intent=intent_map[intent_override], confidence=1.0, method="manual"
            )
        else:
            llm_fallback_enabled = self._cfg.llm_fallback
            llm_model_id = self._cfg.intent_model if llm_fallback_enabled else None
            classifier = IntentClassifier(
                confidence_threshold=self._cfg.intent_confidence_threshold,
                llm_model_id=llm_model_id,
                llm_fallback_threshold=self._cfg.llm_fallback_threshold,
                eager_load_llm=False,
            )
            intent_result = classifier.classify(query_text)
            del classifier
            gc.collect()
            _release_mlx_cache()

        yield IntentEvent(
            intent=intent_result.intent.value,
            confidence=intent_result.confidence,
            method=intent_result.method,
        )

        if should_stop():
            yield ErrorEvent(code="STREAM_CANCELLED", message="Cancelled")
            yield FinishEvent(finish_reason="error")
            return

        # -- query expansion -----------------------------------------------
        search_query = query_text
        if expansion:
            search_query = _expand_query(query_text, intent_result.intent)

        extra_instructions: Optional[str] = None
        bypass_retrieval = is_low_information_query(query_text)

        # -- retrieval paths -----------------------------------------------
        context = ""
        results: list[RetrievalResult] = []
        source_ids: list[str] = []
        parent_texts: list[str] = []

        if bypass_retrieval:
            yield StatusEvent(status="Query is unclear — asking for clarification...")
            if cite:
                cite = False
            extra_instructions = (
                "The user query is unclear or nonsensical. Ask a concise clarifying "
                "question first, and optionally suggest 2-3 concrete ways they can rephrase it."
            )

        elif intent_result.intent == Intent.COLLECTION:
            yield StatusEvent(status="Fetching collection summaries...")
            context, source_ids, cite = self._handle_collection(
                config=config,
                model_id=model_id,
                no_generate=False,
                citations_enabled=cite,
            )
            if context is None:
                yield TextTokenEvent(token="No documents found in the database.")
                yield FinishEvent(finish_reason="stop")
                return

        elif intent_result.intent == Intent.SUMMARIZE and not source_id:
            yield StatusEvent(status="Checking multi-document summarise path...")
            profiler = LatencyProfiler(enabled=False)
            multi_result = self._handle_multi_doc_summarize(
                config=config,
                model_id=model_id,
                no_generate=False,
                citations_enabled=cite,
                profiler=profiler,
                retrieval_engine=retrieval_engine,
                search_query=search_query,
                source_id=source_id,
            )
            context = multi_result["context"]
            results = multi_result["results"]
            source_ids = multi_result["source_ids"]
            parent_texts = multi_result["parent_texts"]
            cite = multi_result["citations_enabled"]
            extra_instructions = multi_result.get("extra_instructions")

        else:
            yield StatusEvent(status="Searching knowledge base...")
            results = retrieval_engine.search(search_query, source_id=source_id)
            source_ids = sorted(
                {
                    r.metadata.get("source_id")
                    for r in results
                    if r.metadata.get("source_id")
                }
            )
            parent_texts = [r.parent_text for r in results if r.parent_text]

        # Emit sources
        if source_ids:
            yield SourcesEvent(source_ids=source_ids)

        if should_stop():
            yield ErrorEvent(code="STREAM_CANCELLED", message="Cancelled")
            yield FinishEvent(finish_reason="error")
            return

        # -- release retrieval models for LLM headroom ---------------------
        self._release_retrieval_models()

        # -- token budget packing ------------------------------------------
        source_legend: Optional[str] = None
        result_metadatas: list[dict] = (
            [r.metadata for r in results if r.parent_text]
            if parent_texts and results
            else []
        )
        generator: Optional[MlxGenerator] = None

        if parent_texts:
            generator = self._ensure_generator()

            yield StatusEvent(status="Packing token budget...")
            pack_result = enforce_token_budget(
                docs=parent_texts,
                max_tokens=config.retrieval_budget,
                tokenizer=generator.tokenizer,
                consecutive_fail_threshold=3,
                allow_truncation=True,
                log=logger,
            )

            packed_metadatas = [
                result_metadatas[i]
                for i in pack_result.packed_indices
                if i < len(result_metadatas)
            ]

            if cite and packed_metadatas:
                context, source_mapping = format_context_with_citations(
                    texts=pack_result.packed_docs, metadatas=packed_metadatas
                )
                source_legend = build_source_legend(source_mapping)
            elif cite:
                cite = False
            else:
                context = "\n\n".join(pack_result.packed_docs)

        elif not context and not bypass_retrieval and intent_result.intent != Intent.COLLECTION:
            # No context retrieved and not bypassing — try with existing context
            pass

        # -- build prompt ---------------------------------------------------
        yield StatusEvent(status="Building prompt...")
        messages = build_messages(
            context,
            query_text,
            intent=intent_result.intent,
            extra_instructions=extra_instructions,
            citations_enabled=cite,
            source_legend=source_legend,
            mode=config.mode,
        )

        # -- generate with streaming tokens --------------------------------
        if generator is None:
            generator = self._ensure_generator()

        gen_config = GenerationConfig(
            max_tokens=600 if cite else 1200,
            context_window=config.context_window,
        )

        yield StatusEvent(status="Generating answer...")

        token_count = 0
        accumulated_answer = []

        for token in generator.generate_chat_stream(
            messages, config=gen_config, should_stop=should_stop
        ):
            if should_stop():
                yield ErrorEvent(code="STREAM_CANCELLED", message="Cancelled during generation")
                yield FinishEvent(finish_reason="error")
                return
            token_count += 1
            accumulated_answer.append(token)
            yield TextTokenEvent(token=token)

        # -- finish ---------------------------------------------------------
        yield FinishEvent(
            finish_reason="stop",
            completion_tokens=token_count,
        )

    # -- internal pipeline helpers -----------------------------------------

    def _handle_collection(
        self,
        *,
        config: ModelConfig,
        model_id: str,
        no_generate: bool,
        citations_enabled: bool,
    ) -> tuple[Optional[str], list[str], bool]:
        """Handle COLLECTION intent — return context from source summaries.

        Returns ``(context, source_ids, citations_enabled)`` or
        ``(None, [], False)`` when no documents exist.
        """
        sources = self._storage.list_source_ids()
        if not sources:
            return None, [], False

        summaries = self._storage.get_source_summaries()
        missing = [s for s in sources if s not in summaries]

        if missing and no_generate:
            lines = ["Some sources are missing summaries. Re-ingest with --summarize."]
            lines.extend(f"- {s}" for s in missing)
            return "\n".join(lines), sources, False

        if missing:
            generator = self._ensure_generator()
            for source in missing:
                p_texts = self._storage.get_parent_texts_by_source(source_id=source)
                context_text = "\n\n".join(p_texts)
                if len(context_text) > 12_000:
                    context_text = context_text[:12_000]
                summary_messages = build_messages(
                    context=context_text,
                    question="Summarize this document.",
                    intent=Intent.SUMMARIZE,
                    mode=config.mode,
                )
                summary_text = generator.generate_chat(summary_messages)
                self._storage.upsert_source_summary(
                    source_id=source, summary=summary_text
                )
            summaries = self._storage.get_source_summaries()

        summary_blocks = [
            f"Source: {source}\nSummary: {summaries[source]}"
            for source in sources
            if source in summaries
        ]
        context = "\n\n".join(summary_blocks)
        if citations_enabled:
            logger.info(
                "Auto-disabling citations: context is from document summaries"
            )
            citations_enabled = False
        return context, sources, citations_enabled

    def _handle_multi_doc_summarize(
        self,
        *,
        config: ModelConfig,
        model_id: str,
        no_generate: bool,
        citations_enabled: bool,
        profiler: LatencyProfiler,
        retrieval_engine: RetrievalEngine,
        search_query: str,
        source_id: Optional[str],
    ) -> dict[str, Any]:
        """Handle SUMMARIZE intent across multiple sources."""
        sources = self._storage.list_source_ids()

        if len(sources) > 1:
            summaries = self._storage.get_source_summaries()
            missing = [s for s in sources if s not in summaries]

            if missing and no_generate:
                return {
                    "context": "Multiple documents available but some lack summaries.",
                    "results": [],
                    "source_ids": sources,
                    "parent_texts": [],
                    "citations_enabled": False,
                    "extra_instructions": None,
                }

            if missing:
                generator = self._ensure_generator()
                for source in missing:
                    p_texts = self._storage.get_parent_texts_by_source(
                        source_id=source
                    )
                    context_text = "\n\n".join(p_texts)
                    if len(context_text) > 12_000:
                        context_text = context_text[:12_000]
                    summary_messages = build_messages(
                        context=context_text,
                        question="Summarize this document.",
                        intent=Intent.SUMMARIZE,
                        mode=config.mode,
                    )
                    summary_text = generator.generate_chat(summary_messages)
                    self._storage.upsert_source_summary(
                        source_id=source, summary=summary_text
                    )
                summaries = self._storage.get_source_summaries()

            summary_blocks = [
                f"Source: {source}\nSummary: {summaries[source]}"
                for source in sources
                if source in summaries
            ]
            if citations_enabled:
                logger.info(
                    "Auto-disabling citations: context from document summaries"
                )
                citations_enabled = False
            return {
                "context": "\n\n".join(summary_blocks),
                "results": [],
                "source_ids": sources,
                "parent_texts": [],
                "citations_enabled": citations_enabled,
                "extra_instructions": (
                    "Provide a single consolidated answer addressing the user's question. "
                    "Do not output per-source summaries or repeat points."
                ),
            }

        # Single source — fall through to normal retrieval
        with profiler.span("Retrieval (hybrid search + rerank)"):
            results = retrieval_engine.search(search_query, source_id=source_id)
        source_ids = sorted(
            {
                r.metadata.get("source_id")
                for r in results
                if r.metadata.get("source_id")
            }
        )
        parent_texts = [r.parent_text for r in results if r.parent_text]
        return {
            "context": "",
            "results": results,
            "source_ids": source_ids,
            "parent_texts": parent_texts,
            "citations_enabled": citations_enabled,
            "extra_instructions": None,
        }

    # -- cleanup -----------------------------------------------------------

    def close(self) -> None:
        """Release all resources."""
        self._embedding_model = None
        self._reranker = None
        self._generator = None
        gc.collect()
        _release_mlx_cache()
