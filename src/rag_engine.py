"""Decoupled RAG engine — reusable from CLI and other frontends.

This module extracts the full retrieval-augmented generation pipeline from
``cli.py`` into a stateful ``RagEngine`` class that returns structured results
instead of printing to stdout.
"""

from __future__ import annotations

import concurrent.futures
import difflib
import gc
import logging
import os
import re
import threading
import uuid
from contextlib import nullcontext
from dataclasses import dataclass, field, replace as _dc_replace
from pathlib import Path
from typing import Any, Callable, Iterable, Optional

from .config import (
    CITATIONS_ENABLED_DEFAULT,
    IntentGenerationParams,
    ModelConfig,
    ResolvedRetrievalParams,
    resolve_generation_params,
    resolve_retrieval_params,
    select_mode_config,
)
from .generation import build_messages
from .embeddings import MlxEmbeddingModel
from .generator import (
    BudgetPackResult,
    GenerationConfig,
    MlxGenerator,
    count_tokens,
    enforce_token_budget,
)
from .ingest import ingest_file_to_storage
from .intent import (
    Intent,
    IntentClassifier,
    IntentResult,
    is_low_information_query,
    is_source_selection_query,
)
from .latency import LatencyProfiler
from .metrics import (
    BudgetMetrics,
    RetrievalMetrics,
)
from .phoenix_tracing import (
    SPAN_KIND_CHAIN,
    SPAN_KIND_LLM,
    SPAN_KIND_RETRIEVER,
    PhoenixTracingStatus,
    format_openinference_document,
    get_phoenix_tracer,
    mark_span_error,
    set_llm_input_messages,
    set_llm_output_message,
    set_llm_token_counts,
    set_retrieval_documents,
    set_span_attribute,
    set_span_attributes,
    start_span,
    to_json,
)
from .retrieval import (
    RetrievalEngine,
    RetrievalResult,
    build_source_legend,
    format_context_with_citations,
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
    latency_report: str = ""
    context: str = ""
    config: Optional[ModelConfig] = None
    raw_answer: str = ""
    prompt_messages: Optional[list[dict[str, str]]] = None


# ---------------------------------------------------------------------------
# Internal pipeline step results (private — not part of the public API)
# ---------------------------------------------------------------------------


@dataclass
class _ClassifyResult:
    """Output of intent classification + parameter resolution step."""

    intent_result: IntentResult
    before_guard_intent: IntentResult  # original intent before collection guard
    retrieval_params: ResolvedRetrievalParams
    generation_params: IntentGenerationParams
    force_collection: bool
    bypass_retrieval: bool


@dataclass
class _RetrieveResult:
    """Output of the retrieval step."""

    # context is None only when COLLECTION intent finds no documents (early exit)
    context: Optional[str]
    results: list[RetrievalResult]
    source_ids: list[str]
    context_docs: list[str]
    cite: bool
    extra_instructions: Optional[str]
    retrieval_metrics: Optional[RetrievalMetrics]
    generator_preload_future: Optional[concurrent.futures.Future[MlxGenerator]]


@dataclass
class _PackResult:
    """Output of the token budget packing + citation formatting step."""

    context: str
    cite: bool
    source_legend: Optional[str]
    result_metadatas: list[dict[str, Any]]
    budget_metrics: Optional[BudgetMetrics]
    citation_list: list[dict[str, Any]]
    packed_retrieval_results: list[RetrievalResult]
    pack_result: Optional[BudgetPackResult]
    packed_metadatas: list[dict[str, Any]]


# ---------------------------------------------------------------------------
# Output sanitisation helpers
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


def _dedupe_context(texts: Iterable[str]) -> str:
    seen: set[str] = set()
    unique_texts = []
    for text in texts:
        cleaned = text.strip()
        if cleaned and cleaned not in seen:
            seen.add(cleaned)
            unique_texts.append(cleaned)
    return "\n\n".join(unique_texts)


def _build_openinference_retrieval_documents(
    results: Iterable[RetrievalResult],
    *,
    limit: int = 8,
    max_content_chars: int = 400,
) -> list[dict[str, Any]]:
    docs: list[dict[str, Any]] = []
    for result in results:
        if len(docs) >= max(1, limit):
            break

        docs.append(
            format_openinference_document(
                document_id=result.child_id,
                content=result.text or "",
                score=result.score,
                metadata=result.metadata,
                max_content_chars=max_content_chars,
            )
        )
    return docs


def _estimate_prompt_tokens(
    *,
    generator: Any,
    messages: list[dict[str, str]],
    enable_thinking: bool,
) -> int:
    """Estimate prompt token count using tokenizer chat template when available."""

    tokenizer = getattr(generator, "tokenizer", None)
    if tokenizer is None:
        return 0

    prompt_text: Optional[str] = None
    if hasattr(tokenizer, "apply_chat_template"):
        try:
            prompt_text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=enable_thinking,
            )
        except TypeError:
            try:
                prompt_text = tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
            except Exception:
                prompt_text = None
        except Exception:
            prompt_text = None

    if prompt_text is None:
        prompt_text = "\n\n".join(
            f"{msg.get('role', 'user')}: {msg.get('content', '')}"
            for msg in messages
            if isinstance(msg, dict)
        )

    return count_tokens(prompt_text, tokenizer)


def _check_novel_proper_nouns(output: str, context: str) -> None:
    """Log a warning if the output contains proper nouns absent from context.

    A proper noun is defined as a capitalized word that does not start a
    sentence.  If more than 3 such words appear in the output but not
    anywhere in the context, there is a risk of hallucination.
    """
    # Split output into sentences to identify sentence-start words
    sentences = re.split(r'(?<=[.!?])\s+', output)
    sentence_start_words: set[str] = set()
    for sent in sentences:
        words = sent.split()
        if words:
            sentence_start_words.add(words[0])

    # Find capitalized words in output that are NOT at sentence starts
    output_words = output.split()
    proper_nouns_in_output: set[str] = set()
    for word in output_words:
        # Strip punctuation for comparison
        clean = re.sub(r'[^\w]', '', word)
        if not clean:
            continue
        if clean[0].isupper() and clean not in sentence_start_words and len(clean) > 1:
            proper_nouns_in_output.add(clean)

    if not proper_nouns_in_output:
        return

    # Check which proper nouns are absent from context
    context_lower = context.lower()
    novel = [pn for pn in proper_nouns_in_output if pn.lower() not in context_lower]

    if len(novel) > 3:
        logger.warning(
            "Output contains %d proper nouns not found in context — possible hallucination: %s",
            len(novel),
            ", ".join(sorted(novel)[:10]),
        )


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

    required_models = [config.llm_model, config.embedding_model]
    if config.reranker_enabled:
        required_models.append(config.reranker_model)
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

    intent_confidence_threshold: float = 0.6
    llm_fallback: bool = True
    llm_fallback_threshold: float = 0.70
    intent_model: str = "mlx-community/LFM2-8B-A1B-4bit"
    verbose: bool = False
    latency: bool = False

    # Phoenix / Arize tracing (OpenTelemetry)
    phoenix_enabled: Optional[bool] = None
    phoenix_project_name: Optional[str] = None
    phoenix_endpoint: Optional[str] = None
    phoenix_api_key: Optional[str] = None
    phoenix_auto_instrument: Optional[bool] = None
    phoenix_batch: Optional[bool] = None


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
                "filelock",
                "fsspec",
            ):
                logging.getLogger(noisy).setLevel(logging.WARNING)

        # Select configuration
        self._on_status("Selecting mode configuration...")
        self._model_config: ModelConfig = select_mode_config(
            manual_mode=self._cfg.mode
        )

        self._tracer, self._phoenix_status = get_phoenix_tracer(
            __name__,
            enabled=self._cfg.phoenix_enabled,
            project_name=self._cfg.phoenix_project_name,
            endpoint=self._cfg.phoenix_endpoint,
            api_key=self._cfg.phoenix_api_key,
            auto_instrument=self._cfg.phoenix_auto_instrument,
            batch=self._cfg.phoenix_batch,
        )
        if self._phoenix_status.error:
            logger.warning(self._phoenix_status.error)

        self._system_ram_gb = float(self._model_config.system_ram_gb or 0.0)
        self._memory_constrained = (
            self._system_ram_gb > 0.0 and self._system_ram_gb <= 40.0
        )
        self._generation_max_tokens = 700 if self._memory_constrained else 1200

        if self._memory_constrained:
            logger.info(
                "Memory-aware mode enabled (%.1fGB RAM): disabling speculative preload and intent LLM fallback; unloading retrieval models before generation.",
                self._system_ram_gb,
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
        self._generator_load_lock = threading.Lock()
        self._preload_executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)

    # -- properties --------------------------------------------------------

    @property
    def model_config(self) -> ModelConfig:
        return self._model_config

    @property
    def storage(self) -> StorageEngine:
        return self._storage

    @property
    def tracer(self) -> Any:
        """OpenTelemetry tracer used by this engine (or ``None`` when disabled)."""

        return self._tracer

    @property
    def phoenix_status(self) -> PhoenixTracingStatus:
        """Current Phoenix tracing runtime status for diagnostics/health endpoints."""

        return self._phoenix_status

    # -- model loading -----------------------------------------------------

    def _ensure_embedding_model(self) -> Any:
        if self._embedding_model is not None:
            return self._embedding_model
        logger.info("_ensure_embedding_model: loading...")
        self._on_status("Loading embedding model...")
        self._embedding_model = MlxEmbeddingModel(
            self._model_config.embedding_model,
            batch_size=16,
            max_length=512,
        )
        self._validate_embedding_storage_compatibility(self._embedding_model)
        logger.info("_ensure_embedding_model: done")
        return self._embedding_model

    def _validate_embedding_storage_compatibility(self, embedding_model: Any) -> None:
        existing_dim = self._storage.get_child_vector_dimension()
        if existing_dim is None:
            return
        try:
            sample = embedding_model.encode(["dimension probe"], normalize_embeddings=True)
            model_dim = len(sample[0]) if sample else None
        except Exception as exc:
            logger.warning("Unable to probe embedding dimension: %s", exc)
            return

        if model_dim is None or model_dim == existing_dim:
            return

        logger.warning(
            "Embedding dimension changed (existing=%d, new=%d). Existing LanceDB vectors are incompatible; resetting tables for re-ingest.",
            existing_dim,
            model_dim,
        )
        self._storage.reset_all_tables()

    def _ensure_reranker(self) -> Any:
        if not self._model_config.reranker_enabled:
            return None
        if self._reranker is not None:
            return self._reranker
        logger.info("_ensure_reranker: loading...")
        self._on_status("Loading reranker model...")
        from .reranker import JinaRerankerMLX

        self._reranker = JinaRerankerMLX(model_id=self._model_config.reranker_model)
        logger.info("_ensure_reranker: done")
        return self._reranker

    def ensure_generator(self) -> MlxGenerator:
        """Return the LLM generator, loading it lazily if needed."""
        if self._generator is not None:
            return self._generator
        with self._generator_load_lock:
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

    def _start_generator_preload(self) -> Optional[concurrent.futures.Future[MlxGenerator]]:
        """Speculatively begin loading the LLM in the background.

        Used to overlap model load latency with retrieval/reranking work.
        """
        if self._memory_constrained:
            logger.info(
                "Skipping speculative LLM preload in memory-aware mode"
            )
            return None
        if self._generator is not None:
            return None
        logger.info("Starting speculative LLM preload during retrieval")
        return self._preload_executor.submit(self.ensure_generator)

    def _consume_preloaded_generator(
        self,
        preload_future: Optional[concurrent.futures.Future[MlxGenerator]],
    ) -> MlxGenerator:
        if preload_future is None:
            return self.ensure_generator()
        try:
            return preload_future.result()
        except Exception:
            logger.exception(
                "Speculative LLM preload failed; falling back to synchronous load"
            )
            return self.ensure_generator()

    def load_retrieval_models(self) -> None:
        """Pre-load embedding + reranker in parallel (call once at startup)."""
        if not self._model_config.reranker_enabled:
            self._on_status("Loading retrieval models (embedding only)...")
            self._ensure_embedding_model()
            self._on_status("Retrieval models loaded.")
            return

        self._on_status("Loading retrieval models (embedding + reranker)...")

        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
            pool.submit(self._ensure_embedding_model)
            pool.submit(self._ensure_reranker)
            pool.shutdown(wait=True)

        self._on_status("Retrieval models loaded.")

    def _release_retrieval_models(self) -> None:
        """Free unreferenced MLX memory before LLM generation.

        The embedding model and reranker stay loaded on the instance so
        subsequent queries skip the reload.  Only orphaned Metal buffers
        and Python garbage are collected here.
        """
        if self._memory_constrained:
            self._embedding_model = None
            self._reranker = None
            logger.info(
                "Memory-aware mode: unloaded embedding/reranker before generation"
            )
        gc.collect()
        _release_mlx_cache()
        logger.debug(
            "Ran gc.collect + MLX cache clear before LLM generation"
        )

    def _release_generator_model(self) -> None:
        if self._generator is None:
            return
        self._generator = None
        gc.collect()
        _release_mlx_cache()
        logger.info("Released generation model for retrieval-phase memory headroom")

    def _classify_intent(
        self,
        *,
        query_text: str,
        intent_override: Optional[str],
        no_generate: bool,
    ) -> IntentResult:
        if intent_override:
            intent_map = {
                "overview": Intent.OVERVIEW,
                "summarize": Intent.SUMMARIZE,
                "summarise": Intent.SUMMARIZE,
                "explain": Intent.EXPLAIN,
                "analyze": Intent.ANALYZE,
                "compare": Intent.COMPARE,
                "critique": Intent.CRITIQUE,
                "factual": Intent.FACTUAL,
                "collection": Intent.COLLECTION,
                "extract": Intent.EXTRACT,
                "timeline": Intent.TIMELINE,
                "how_to": Intent.HOW_TO,
                "quote_evidence": Intent.QUOTE_EVIDENCE,
            }
            normalized_override = intent_override.strip().lower()
            manual_intent = intent_map.get(normalized_override)
            if manual_intent is None:
                logger.warning(
                    "Unknown intent_override='%s'; falling back to classifier",
                    intent_override,
                )
            else:
                return IntentResult(
                    intent=manual_intent, confidence=1.0, method="manual"
                )

        def _run() -> IntentResult:
            llm_fallback_enabled = (
                not no_generate
                and self._cfg.llm_fallback
                and not self._memory_constrained
            )
            if self._cfg.llm_fallback and self._memory_constrained:
                logger.info(
                    "Memory-aware mode: disabling intent LLM fallback to reduce peak RAM"
                )
            llm_model_id = self._cfg.intent_model if llm_fallback_enabled else None
            classifier = IntentClassifier(
                confidence_threshold=self._cfg.intent_confidence_threshold,
                llm_model_id=llm_model_id,
                llm_fallback_threshold=self._cfg.llm_fallback_threshold,
                eager_load_llm=False,
            )
            result = classifier.classify(query_text)
            del classifier
            gc.collect()
            _release_mlx_cache()
            return result

        return _run()

    def _apply_collection_guard(
        self,
        *,
        query_text: str,
        source_id: Optional[str],
        intent_override: Optional[str],
        intent_result: IntentResult,
    ) -> tuple[IntentResult, bool]:
        force_collection = (
            source_id is None
            and not intent_override
            and len(self._storage.list_source_ids()) > 1
            and is_source_selection_query(query_text)
        )
        if force_collection and intent_result.intent != Intent.COLLECTION:
            logger.info(
                "Forcing COLLECTION summary routing for source-selection query: %s",
                query_text,
            )
            intent_result = IntentResult(
                intent=Intent.COLLECTION,
                confidence=max(intent_result.confidence, 0.85),
                method=f"{intent_result.method}+collection_guard",
            )
        return intent_result, force_collection

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
        page_offset: int = 1,
    ) -> IngestResult:
        """Ingest a document (PDF or Markdown) into the RAG store."""
        self._on_status(f"Ingesting {Path(file_path).name}...")
        ingest_request_id = f"ing-{uuid.uuid4().hex[:12]}"
        with start_span(
            self._tracer,
            "rag.ingest",
            span_kind=SPAN_KIND_CHAIN,
            attributes={
                "rag.request_id": ingest_request_id,
                "input.value": source_id,
                "rag.ingest.file_path": str(file_path),
                "rag.ingest.page_number": page_number,
                "rag.ingest.page_offset": page_offset,
                "rag.ingest.summarize": summarize,
            },
        ) as ingest_span:
            try:
                embedding_model = self._ensure_embedding_model()

                generator: Optional[MlxGenerator] = None
                if summarize:
                    generator = self.ensure_generator()

                parents_count, children_count = ingest_file_to_storage(
                    file_path,
                    source_id=source_id,
                    page_number=page_number,
                    storage=self._storage,
                    embedding_model=embedding_model,
                    summarize=summarize,
                    summary_generator=generator,
                    page_offset=page_offset,
                    tracer=self._tracer,
                )

                self._on_status(
                    f"Ingested {parents_count} parents, {children_count} children."
                )
                set_span_attributes(
                    ingest_span,
                    {
                        "output.value": f"{source_id}: {parents_count} parents, {children_count} children",
                        "rag.ingest.parents_count": parents_count,
                        "rag.ingest.children_count": children_count,
                    },
                )
                return IngestResult(
                    parents_count=parents_count,
                    children_count=children_count,
                    source_id=source_id,
                    summarized=summarize,
                )
            except Exception as exc:
                mark_span_error(ingest_span, f"{type(exc).__name__}: {exc}")
                raise

    def query(
        self,
        query_text: str,
        *,
        source_id: Optional[str] = None,
        intent_override: Optional[str] = None,
        citations_enabled: Optional[bool] = None,
        citation_output_mode: str = "default",
        no_generate: bool = False,
        dump_prompt: bool = False,
    ) -> QueryResult:
        """Execute the full RAG pipeline and return a structured result.

        This is the *synchronous* counterpart of :meth:`query_events`.
        Both share the same pipeline steps (``_step_classify``,
        ``_step_retrieve``, ``_step_pack_budget``) but differ in the
        orchestration layer: ``query()`` returns a complete result,
        ``query_events()`` yields incremental events for streaming.

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
        dump_prompt : bool
            If True, build the full prompt and return it in ``prompt_messages``
            without running LLM generation. Useful for benchmarking.
        """
        config = self._model_config
        profiler = LatencyProfiler(enabled=self._cfg.latency)
        profiler.start_wall()
        query_request_id = f"qry-{uuid.uuid4().hex[:12]}"
        with start_span(
            self._tracer,
            "rag.query",
            span_kind=SPAN_KIND_CHAIN,
            attributes={
                "rag.request_id": query_request_id,
                "session.id": query_request_id,
                "input.value": query_text,
                "input.mime_type": "text/plain",
                "rag.source_id": source_id,
                "rag.intent_override": intent_override,
                "rag.no_generate": no_generate,
                "rag.dump_prompt": dump_prompt,
                "rag.mode": config.mode,
                "rag.collection": self._cfg.collection,
                "rag.retrieval_budget": config.retrieval_budget,
                "rag.context_window": config.context_window,
                "rag.model.llm": (self._cfg.model or config.llm_model),
                "rag.model.embedding": config.embedding_model,
                "rag.model.reranker": config.reranker_model,
                "rag.model.reranker_enabled": config.reranker_enabled,
                "rag.memory_constrained": self._memory_constrained,
                "rag.system_ram_gb": self._system_ram_gb,
            },
        ) as query_span:
            try:
                # -- resolve citation mode ---------------------------------
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
                set_span_attribute(query_span, "rag.citations_requested", cite)

                if self._memory_constrained and self._generator is not None:
                    self._release_generator_model()

                # -- load retrieval models ---------------------------------
                self._on_status("Preparing retrieval models...")
                embedding_model = self._ensure_embedding_model()
                reranker = self._ensure_reranker()

                retrieval_engine = RetrievalEngine(
                    storage=self._storage,
                    embedding_model=embedding_model,
                    reranker=reranker,
                    config=config,
                    tracer=self._tracer,
                )

                # -- intent classification + parameter resolution ----------
                self._on_status("Classifying intent...")
                # For dump_prompt we want full intent classification (inc. LLM fallback),
                # so suppress no_generate so it doesn't disable the intent LLM fallback.
                classify_no_generate = no_generate and not dump_prompt
                classified = self._step_classify(
                    query_text, source_id, intent_override, classify_no_generate, profiler=profiler
                )
                intent_result = classified.intent_result

                # -- retrieval (bypass / collection / hybrid) --------------
                if classified.bypass_retrieval:
                    self._on_status("Query is unclear — skipping retrieval...")
                elif classified.intent_result.intent == Intent.COLLECTION:
                    self._on_status("Fetching collection summaries...")
                else:
                    self._on_status("Searching knowledge base...")

                retrieved = self._step_retrieve(
                    query_text, source_id, classified, retrieval_engine, cite,
                    no_generate=(no_generate or dump_prompt), profiler=profiler,
                )
                cite = retrieved.cite
                set_span_attributes(
                    query_span,
                    {
                        "rag.intent": intent_result.intent.value,
                        "rag.citations_enabled": cite,
                        "rag.retrieval_results_count": len(retrieved.results),
                        "rag.sources_count": len(retrieved.source_ids),
                    },
                )

                if retrieved.context is None:
                    # COLLECTION with no documents → early exit
                    profiler.end_wall()
                    set_span_attributes(
                        query_span,
                        {
                            "rag.finish_reason": "collection_empty",
                            "rag.latency_wall_ms": profiler.wall_ms,
                            "output.mime_type": "text/plain",
                            "output.value": "No documents found in the database.",
                        },
                    )
                    return QueryResult(
                        answer="No documents found in the database.",
                        intent=intent_result,
                        citations_enabled=False,
                        config=config,
                    )

                # -- release retrieval models for LLM headroom -------------
                with profiler.span("Memory cleanup (gc)"):
                    retrieval_engine = None
                    self._release_retrieval_models()

                # -- no-generate path (context only, no token budgeting) ---
                if no_generate:
                    packed = self._step_pack_budget(retrieved, config, generator=None)
                    profiler.end_wall()
                    set_span_attributes(
                        query_span,
                        {
                            "rag.finish_reason": "no_generate",
                            "rag.context_chars": len(packed.context),
                            "rag.latency_wall_ms": profiler.wall_ms,
                            "output.mime_type": "text/plain",
                            "output.value": f"Context-only response ({len(packed.context)} chars)",
                        },
                    )
                    return QueryResult(
                        answer="",
                        intent=intent_result,
                        citations_enabled=packed.cite,
                        source_ids=retrieved.source_ids,
                        retrieval_metrics=retrieved.retrieval_metrics,
                        budget_metrics=None,
                        latency_report=profiler.format_report(),
                        context=packed.context,
                        config=config,
                    )

                # -- token budget packing ----------------------------------
                generator = (
                    None if dump_prompt
                    else self._consume_preloaded_generator(retrieved.generator_preload_future)
                )
                self._on_status("Packing token budget...")
                with profiler.span("Budget packing"):
                    packed = self._step_pack_budget(retrieved, config, generator)
                cite = packed.cite

                # -- build prompt -------------------------------------------
                self._on_status("Building prompt...")
                with profiler.span("Build prompt / messages"):
                    with start_span(
                        self._tracer,
                        "rag.prompt.build",
                        span_kind=SPAN_KIND_CHAIN,
                        attributes={
                            "rag.intent": intent_result.intent.value,
                            "rag.citations_enabled": cite,
                            "rag.context_chars": len(packed.context),
                            "rag.prompt.query_chars": len(query_text),
                        },
                    ) as prompt_span:
                        messages = build_messages(
                            packed.context,
                            query_text,
                            intent=intent_result.intent,
                            extra_instructions=retrieved.extra_instructions,
                            citations_enabled=cite,
                            source_legend=packed.source_legend,
                            mode=config.mode,
                            retrieval_budget=config.retrieval_budget,
                            citation_output_mode=citation_output_mode,
                        )
                        prompt_chars = sum(
                            len(str(msg.get("content", "")))
                            for msg in messages
                            if isinstance(msg, dict)
                        )
                        set_span_attributes(
                            prompt_span,
                            {
                                "rag.prompt.message_count": len(messages),
                                "rag.prompt.messages_chars": prompt_chars,
                                "rag.prompt.has_source_legend": bool(packed.source_legend),
                            },
                        )

                # -- dump-prompt path (no LLM generation) ------------------
                if dump_prompt:
                    profiler.end_wall()
                    set_span_attributes(
                        query_span,
                        {
                            "rag.finish_reason": "dump_prompt",
                            "rag.context_chars": len(packed.context),
                            "rag.prompt.message_count": len(messages),
                            "rag.latency_wall_ms": profiler.wall_ms,
                            "output.mime_type": "text/plain",
                            "output.value": f"Prompt dump with {len(messages)} messages",
                        },
                    )
                    return QueryResult(
                        answer="",
                        intent=intent_result,
                        citations_enabled=cite,
                        source_ids=retrieved.source_ids,
                        retrieval_metrics=retrieved.retrieval_metrics,
                        budget_metrics=None,
                        latency_report=profiler.format_report(),
                        context=packed.context,
                        config=config,
                        prompt_messages=messages,
                    )

                # -- generate -----------------------------------------------
                gen_config = GenerationConfig(
                    max_tokens=self._generation_max_tokens,
                    context_window=config.context_window,
                    temperature=classified.generation_params.temperature,
                    top_p=classified.generation_params.top_p,
                    top_k=classified.generation_params.top_k,
                    min_p=classified.generation_params.min_p,
                    presence_penalty=classified.generation_params.presence_penalty,
                    repetition_penalty=classified.generation_params.repetition_penalty if classified.generation_params.repetition_penalty != 1.0 else None,
                )
                prompt_token_count = _estimate_prompt_tokens(
                    generator=generator,
                    messages=messages,
                    enable_thinking=False,
                )

                self._on_status("Generating answer...")
                with start_span(
                    self._tracer,
                    "rag.llm.generate",
                    span_kind=SPAN_KIND_LLM,
                    attributes={
                        "input.mime_type": "text/plain",
                        "input.value": query_text,
                        "rag.intent": intent_result.intent.value,
                        "llm.model_name": (self._cfg.model or config.llm_model),
                        "llm.invocation_parameters": to_json(
                            {
                                "temperature": gen_config.temperature,
                                "top_p": gen_config.top_p,
                                "top_k": gen_config.top_k,
                                "max_tokens": gen_config.max_tokens,
                                "context_window": gen_config.context_window,
                            }
                        ),
                        "llm.token_count.prompt": prompt_token_count,
                        "llm.temperature": gen_config.temperature,
                        "llm.top_p": gen_config.top_p,
                        "llm.top_k": gen_config.top_k,
                        "llm.max_tokens": gen_config.max_tokens,
                        "llm.input_messages": len(messages),
                        "llm.input_chars": prompt_chars,
                    },
                ) as generation_span:
                    try:
                        set_llm_input_messages(generation_span, messages)
                        with profiler.span("LLM generation"):
                            raw_answer = generator.generate_chat(
                                messages,
                                config=gen_config,
                            )
                        completion_token_count = count_tokens(raw_answer, generator.tokenizer)
                        total_token_count = prompt_token_count + completion_token_count
                        set_llm_output_message(generation_span, raw_answer)
                        set_llm_token_counts(
                            generation_span,
                            prompt_tokens=prompt_token_count,
                            completion_tokens=completion_token_count,
                            total_tokens=total_token_count,
                        )
                        set_span_attributes(
                            generation_span,
                            {
                                "output.value": raw_answer,
                                "output.mime_type": "text/plain",
                                "rag.output_chars": len(raw_answer),
                            },
                        )
                    except Exception as exc:
                        mark_span_error(generation_span, f"{type(exc).__name__}: {exc}")
                        raise

                logger.info(
                    "INTENT_AWARE | intent=%s | temperature=%.2f | top_p=%.2f | tokens_generated=%d",
                    intent_result.intent.value,
                    classified.generation_params.temperature,
                    classified.generation_params.top_p,
                    completion_token_count,
                )

                with profiler.span("Output sanitisation"):
                    answer = sanitize_output(raw_answer)

                if answer and packed.context:
                    _check_novel_proper_nouns(answer, packed.context)

                profiler.end_wall()
                set_span_attributes(
                    query_span,
                    {
                        "rag.finish_reason": "stop",
                        "rag.context_chars": len(packed.context),
                        "rag.answer_chars": len(answer),
                        "rag.output_chars": len(raw_answer),
                        "llm.token_count.prompt": prompt_token_count,
                        "llm.token_count.completion": completion_token_count,
                        "llm.token_count.total": total_token_count,
                        "rag.latency_wall_ms": profiler.wall_ms,
                        "output.mime_type": "text/plain",
                        "output.value": answer,
                    },
                )
                return QueryResult(
                    answer=answer,
                    intent=intent_result,
                    citations_enabled=cite,
                    source_ids=retrieved.source_ids,
                    retrieval_metrics=retrieved.retrieval_metrics,
                    budget_metrics=packed.budget_metrics,
                    latency_report=profiler.format_report(),
                    context=packed.context,
                    config=config,
                    raw_answer=raw_answer,
                )
            except Exception as exc:
                mark_span_error(query_span, f"{type(exc).__name__}: {exc}")
                raise

    # -- streaming query (event generator) ----------------------------------

    def query_events(
        self,
        query_text: str,
        *,
        source_id: Optional[str] = None,
        intent_override: Optional[str] = None,
        citations_enabled: Optional[bool] = None,
        citation_output_mode: str = "default",
        should_stop: Optional[Callable[[], bool]] = None,
        enable_thinking: Optional[bool] = None,
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
        should_stop : callable | None
            Called periodically; if it returns True, generation is
            aborted and a ``STREAM_CANCELLED`` error event is yielded.
        enable_thinking : bool | None
            Tri-state thinking override: ``True`` = user forced on,
            ``False`` = user forced off, ``None`` = auto/intent-driven.
        """
        _stop = should_stop or (lambda: False)
        stream_request_id = f"str-{uuid.uuid4().hex[:12]}"
        finish_reason = "error"
        prompt_tokens = 0
        completion_tokens = 0

        with start_span(
            self._tracer,
            "rag.query.stream",
            span_kind=SPAN_KIND_CHAIN,
            attributes={
                "rag.request_id": stream_request_id,
                "session.id": stream_request_id,
                "input.value": query_text,
                "input.mime_type": "text/plain",
                "rag.source_id": source_id,
                "rag.intent_override": intent_override,
                "rag.citations_requested": citations_enabled,
                "rag.enable_thinking": enable_thinking,
            },
        ) as stream_span:
            try:
                for event in self._query_events_impl(
                    query_text,
                    source_id=source_id,
                    intent_override=intent_override,
                    citations_enabled=citations_enabled,
                    citation_output_mode=citation_output_mode,
                    should_stop=_stop,
                    enable_thinking=enable_thinking,
                ):
                    if isinstance(event, FinishEvent):
                        finish_reason = event.finish_reason
                        prompt_tokens = event.prompt_tokens
                        completion_tokens = event.completion_tokens
                    yield event

                set_span_attributes(
                    stream_span,
                    {
                        "rag.finish_reason": finish_reason,
                        "output.mime_type": "text/plain",
                        "output.value": f"stream finished: {finish_reason}",
                        "llm.token_count.prompt": prompt_tokens,
                        "llm.token_count.completion": completion_tokens,
                        "llm.token_count.total": (prompt_tokens + completion_tokens),
                    },
                )
            except Exception as exc:
                mark_span_error(stream_span, f"{type(exc).__name__}: {exc}")
                logger.exception("query_events error: %s", exc)
                yield ErrorEvent(
                    code="INTERNAL",
                    message=str(exc),
                    metadata={
                        "exception_type": type(exc).__name__,
                        "query": query_text[:100],
                    },
                )
                yield FinishEvent(finish_reason="error")
                set_span_attributes(
                    stream_span,
                    {
                        "rag.finish_reason": "error",
                        "output.mime_type": "text/plain",
                        "output.value": "stream finished: error",
                        "llm.token_count.prompt": prompt_tokens,
                        "llm.token_count.completion": completion_tokens,
                        "llm.token_count.total": (prompt_tokens + completion_tokens),
                    },
                )

    def _query_events_impl(
        self,
        query_text: str,
        *,
        source_id: Optional[str],
        intent_override: Optional[str],
        citations_enabled: Optional[bool],
        citation_output_mode: str,
        should_stop: Callable[[], bool],
        enable_thinking: Optional[bool] = None,
    ) -> Iterable[QueryEvent]:
        """Internal implementation of query_events (unwrapped from error handling)."""
        logger.info("query_events_impl: started")
        config = self._model_config

        # -- resolve citation mode -----------------------------------------
        if citations_enabled is not None:
            cite = citations_enabled
        elif self._cfg.citations_enabled is not None:
            cite = self._cfg.citations_enabled
        else:
            cite = CITATIONS_ENABLED_DEFAULT

        if self._memory_constrained and self._generator is not None:
            self._release_generator_model()

        if self._tracer is not None:
            try:
                from opentelemetry import trace
                span = trace.get_current_span()
                if span:
                    ctx = span.get_span_context()
                    if ctx.is_valid:
                        yield TraceEvent(
                            trace_id=format(ctx.trace_id, '032x'),
                            span_id=format(ctx.span_id, '016x'),
                        )
            except Exception as e:
                logger.debug("Failed to extract trace_id for TraceEvent: %s", e)

        # -- load retrieval models -----------------------------------------
        yield StatusEvent(status="Preparing retrieval models...")
        embedding_model = self._ensure_embedding_model()
        reranker = self._ensure_reranker()

        if should_stop():
            yield ErrorEvent(code="STREAM_CANCELLED", message="Cancelled")
            yield FinishEvent(finish_reason="error")
            return

        retrieval_engine = RetrievalEngine(
            storage=self._storage,
            embedding_model=embedding_model,
            reranker=reranker,
            config=config,
            tracer=self._tracer,
        )

        # -- intent classification + parameter resolution ------------------
        if intent_override and intent_override != "auto":
            normalized_override = intent_override.strip().lower()
            display_override = (
                "summarise"
                if normalized_override in {"summarize", "summarise"}
                else normalized_override
            )
            yield StatusEvent(status=f"Using intent: {display_override.replace('_', ' ').title()}...")
        else:
            yield StatusEvent(status="Classifying intent...")
        classified = self._step_classify(
            query_text, source_id, intent_override, no_generate=False
        )
        intent_result = classified.intent_result

        yield IntentEvent(
            intent=intent_result.intent.value,
            confidence=intent_result.confidence,
            method=intent_result.method,
        )

        # Re-emit if collection guard altered the intent
        if classified.force_collection and (
            classified.intent_result.intent != classified.before_guard_intent.intent
            or classified.intent_result.method != classified.before_guard_intent.method
            or classified.intent_result.confidence != classified.before_guard_intent.confidence
        ):
            yield IntentEvent(
                intent=intent_result.intent.value,
                confidence=intent_result.confidence,
                method=intent_result.method,
            )

        if should_stop():
            yield ErrorEvent(code="STREAM_CANCELLED", message="Cancelled")
            yield FinishEvent(finish_reason="error")
            return

        # -- retrieval (bypass / collection / hybrid) ----------------------
        if classified.bypass_retrieval:
            yield StatusEvent(status="Query is unclear — asking for clarification...")
        elif classified.intent_result.intent == Intent.COLLECTION:
            yield StatusEvent(status="Fetching collection summaries...")
        else:
            yield StatusEvent(status="Searching knowledge base...")

        retrieved = self._step_retrieve(
            query_text, source_id, classified, retrieval_engine, cite, no_generate=False
        )
        cite = retrieved.cite

        if retrieved.context is None:
            yield TextTokenEvent(token="No documents found in the database.")
            yield FinishEvent(finish_reason="stop")
            return

        # -- emit detailed retrieval step statuses -------------------------
        if retrieved.results:
            m = retrieved.results[0].metrics
            if m:
                t = m.timing
                d = m.deduplication
                th = m.threshold
                n_raw = d.children_before_dedup
                yield StatusEvent(
                    status=f"Hybrid search: {n_raw} results in {t.hybrid_search_ms:.0f}ms"
                )
                if d.parents_deduplicated > 0:
                    yield StatusEvent(
                        status=(
                            f"Deduplication: {n_raw} → {d.children_after_dedup} docs"
                            f" ({d.parents_deduplicated} duplicates removed)"
                        )
                    )
                if m.reranker.items_reranked > 0:
                    yield StatusEvent(
                        status=(
                            f"Reranker: scored {m.reranker.items_reranked} docs in {t.rerank_ms:.0f}ms"
                            f" → {th.items_after_threshold} passed"
                            + (" (safety net)" if th.safety_net_triggered else "")
                        )
                    )
                yield StatusEvent(
                    status=(
                        f"Retrieved {len(retrieved.results)} relevant passage"
                        f"{'s' if len(retrieved.results) != 1 else ''}"
                    )
                )

        if retrieved.source_ids:
            yield SourcesEvent(source_ids=retrieved.source_ids)

        if should_stop():
            yield ErrorEvent(code="STREAM_CANCELLED", message="Cancelled after retrieval")
            yield FinishEvent(finish_reason="error")
            return

        # -- release retrieval models for LLM headroom ---------------------
        retrieval_engine = None
        self._release_retrieval_models()

        if should_stop():
            yield ErrorEvent(code="STREAM_CANCELLED", message="Cancelled after model cleanup")
            yield FinishEvent(finish_reason="error")
            return

        # -- token budget packing ------------------------------------------
        generator = self._consume_preloaded_generator(retrieved.generator_preload_future)
        yield StatusEvent(status="Packing token budget...")
        packed = self._step_pack_budget(retrieved, config, generator)
        cite = packed.cite

        # -- budget summary status -----------------------------------------
        if packed.pack_result is not None:
            pr = packed.pack_result
            n_packed = len(pr.packed_docs)
            used_tok = pr.used_tokens
            budget_pct = (
                f"{100 * used_tok / config.retrieval_budget:.0f}%"
                if config.retrieval_budget
                else "n/a"
            )
            _budget_extras = []
            if pr.skipped_count:
                _budget_extras.append(f"{pr.skipped_count} skipped")
            if pr.truncated_count:
                _budget_extras.append(f"{pr.truncated_count} truncated")
            _extra_str = f" ({', '.join(_budget_extras)})" if _budget_extras else ""
            yield StatusEvent(
                status=(
                    f"Budget: {n_packed} docs, {used_tok:,} tokens"
                    f" ({budget_pct} of budget){_extra_str}"
                )
            )

        if packed.citation_list:
            yield CitationListEvent(citations=packed.citation_list)

        # -- build prompt ---------------------------------------------------
        if should_stop():
            yield ErrorEvent(code="STREAM_CANCELLED", message="Cancelled before prompt build")
            yield FinishEvent(finish_reason="error")
            return

        yield StatusEvent(status="Building prompt...")
        with start_span(
            self._tracer,
            "rag.prompt.build.stream",
            span_kind=SPAN_KIND_CHAIN,
            attributes={
                "rag.intent": intent_result.intent.value,
                "rag.citations_enabled": cite,
                "rag.context_chars": len(packed.context),
                "rag.prompt.query_chars": len(query_text),
            },
        ) as prompt_span:
            messages = build_messages(
                packed.context,
                query_text,
                intent=intent_result.intent,
                extra_instructions=retrieved.extra_instructions,
                citations_enabled=cite,
                source_legend=packed.source_legend,
                mode=config.mode,
                retrieval_budget=config.retrieval_budget,
                citation_output_mode=citation_output_mode,
            )
            stream_prompt_chars = sum(
                len(str(msg.get("content", "")))
                for msg in messages
                if isinstance(msg, dict)
            )
            set_span_attributes(
                prompt_span,
                {
                    "rag.prompt.message_count": len(messages),
                    "rag.prompt.messages_chars": stream_prompt_chars,
                    "rag.prompt.has_source_legend": bool(packed.source_legend),
                },
            )

        # -- generate with streaming tokens --------------------------------
        gen_params = classified.generation_params

        # Apply user thinking override (tri-state)
        if enable_thinking is True:
            # User forced thinking on — switch to thinking-mode sampling
            gen_params = _dc_replace(gen_params, enable_thinking=True, temperature=1.0, top_p=0.95)
        elif enable_thinking is False:
            # User forced thinking off — ensure non-thinking sampling
            gen_params = _dc_replace(gen_params, enable_thinking=False)
        # else: None = auto, use intent-resolved value

        # RAM gate: disable thinking if insufficient RAM
        if gen_params.enable_thinking and self._system_ram_gb > 0 and self._system_ram_gb < 48:
            logger.info(
                "Thinking mode disabled: requires 48GB+ RAM (detected %.0fGB)",
                self._system_ram_gb,
            )
            gen_params = _dc_replace(gen_params, enable_thinking=False)

        # Determine token budgets based on thinking state and RAM tier
        if gen_params.enable_thinking:
            if self._system_ram_gb >= 64:
                max_internal = 20480
                max_visible = 4096
            else:
                max_internal = 16384
                max_visible = 2048
        else:
            max_internal = self._generation_max_tokens
            max_visible = self._generation_max_tokens

        gen_config = GenerationConfig(
            max_tokens=max_visible,
            max_internal_tokens=max_internal if gen_params.enable_thinking else None,
            context_window=config.context_window,
            temperature=gen_params.temperature,
            top_p=gen_params.top_p,
            top_k=gen_params.top_k,
            min_p=gen_params.min_p,
            presence_penalty=gen_params.presence_penalty,
            repetition_penalty=gen_params.repetition_penalty if gen_params.repetition_penalty != 1.0 else None,
        )
        prompt_token_count = _estimate_prompt_tokens(
            generator=generator,
            messages=messages,
            enable_thinking=gen_params.enable_thinking,
        )
        yield StatusEvent(status="Generating answer...")
        chunk_event_count = 0
        answer_tokens: list[str] = []
        thinking_tokens: list[str] = []

        with start_span(
            self._tracer,
            "rag.llm.generate.stream",
            span_kind=SPAN_KIND_LLM,
            attributes={
                "input.mime_type": "text/plain",
                "input.value": query_text,
                "rag.intent": intent_result.intent.value,
                "llm.model_name": (self._cfg.model or config.llm_model),
                "llm.invocation_parameters": to_json(
                    {
                        "temperature": gen_config.temperature,
                        "top_p": gen_config.top_p,
                        "top_k": gen_config.top_k,
                        "max_tokens": gen_config.max_tokens,
                        "max_internal_tokens": gen_config.max_internal_tokens,
                        "context_window": gen_config.context_window,
                        "enable_thinking": gen_params.enable_thinking,
                    }
                ),
                "llm.token_count.prompt": prompt_token_count,
                "rag.thinking_enabled": gen_params.enable_thinking,
                "llm.temperature": gen_config.temperature,
                "llm.top_p": gen_config.top_p,
                "llm.top_k": gen_config.top_k,
                "llm.max_tokens": gen_config.max_tokens,
                "llm.max_internal_tokens": gen_config.max_internal_tokens,
                "llm.input_messages": len(messages),
                "llm.input_chars": stream_prompt_chars,
            },
        ) as generation_span:
            set_llm_input_messages(generation_span, messages)
            if gen_params.enable_thinking:
                thinking_budget_exhausted = False
                for event in generator.stream_chat_with_thinking(
                    messages,
                    config=gen_config,
                    should_stop=should_stop,
                ):
                    if should_stop():
                        mark_span_error(generation_span, "STREAM_CANCELLED")
                        yield ErrorEvent(code="STREAM_CANCELLED", message="Cancelled during generation")
                        yield FinishEvent(finish_reason="error")
                        return
                    chunk_event_count += 1
                    if chunk_event_count == 1 and generation_span is not None:
                        try:
                            generation_span.add_event("first_token")
                        except Exception:
                            pass
                    if event["type"] == "answer":
                        answer_tokens.append(event["text"])
                        yield TextTokenEvent(token=event["text"])
                    elif event["type"] == "thinking":
                        thinking_tokens.append(event["text"])
                        yield ThinkingTokenEvent(token=event["text"])
                    elif event["type"] == "error" and event.get("text") == "THINKING_BUDGET_EXHAUSTED":
                        thinking_budget_exhausted = True
                if thinking_budget_exhausted:
                    mark_span_error(generation_span, "THINKING_BUDGET_EXHAUSTED")
                    yield ErrorEvent(
                        code="THINKING_BUDGET_EXHAUSTED",
                        message="The model's reasoning phase consumed the entire token budget. Please try a simpler question or disable thinking mode.",
                    )
                    yield FinishEvent(finish_reason="error")
                    return
            else:
                for token in generator.generate_chat_stream(
                    messages,
                    config=gen_config,
                    should_stop=should_stop,
                ):
                    if should_stop():
                        mark_span_error(generation_span, "STREAM_CANCELLED")
                        yield ErrorEvent(code="STREAM_CANCELLED", message="Cancelled during generation")
                        yield FinishEvent(finish_reason="error")
                        return
                    chunk_event_count += 1
                    if chunk_event_count == 1 and generation_span is not None:
                        try:
                            generation_span.add_event("first_token")
                        except Exception:
                            pass
                    answer_tokens.append(token)
                    yield TextTokenEvent(token=token)

            if generation_span is not None:
                try:
                    generation_span.add_event("stream_complete")
                except Exception:
                    pass

            answer_text = "".join(answer_tokens)
            completion_token_count = count_tokens(answer_text, generator.tokenizer)
            total_token_count = prompt_token_count + completion_token_count
            set_llm_output_message(generation_span, answer_text)
            set_llm_token_counts(
                generation_span,
                prompt_tokens=prompt_token_count,
                completion_tokens=completion_token_count,
                total_tokens=total_token_count,
            )

            thinking_token_count = (
                count_tokens("".join(thinking_tokens), generator.tokenizer)
                if thinking_tokens
                else 0
            )

            set_span_attributes(
                generation_span,
                {
                    "output.value": answer_text,
                    "output.mime_type": "text/plain",
                    "rag.output_chars": len(answer_text),
                    "rag.thinking_chars": len("".join(thinking_tokens)),
                    "rag.thinking_token_count": thinking_token_count,
                    "rag.stream.chunk_events": chunk_event_count,
                },
            )

        # -- end of generation -----------------------------------------------

        logger.info(
            "INTENT_AWARE | intent=%s | temperature=%.2f | top_p=%.2f | thinking=%s | tokens_generated=%d",
            intent_result.intent.value,
            gen_params.temperature,
            gen_params.top_p,
            gen_params.enable_thinking,
            completion_token_count,
        )

        yield FinishEvent(
            finish_reason="stop",
            prompt_tokens=prompt_token_count,
            completion_tokens=completion_token_count,
        )

    # -- shared pipeline step methods --------------------------------------

    def _step_classify(
        self,
        query_text: str,
        source_id: Optional[str],
        intent_override: Optional[str],
        no_generate: bool,
        profiler: Optional[LatencyProfiler] = None,
    ) -> _ClassifyResult:
        """Classify intent, apply collection guard, resolve retrieval/gen params."""
        config = self._model_config
        _span = profiler.span if profiler is not None else lambda _: nullcontext()

        with start_span(
            self._tracer,
            "rag.intent.classify",
            span_kind=SPAN_KIND_CHAIN,
            attributes={
                "input.value": query_text,
                "rag.source_id": source_id,
                "rag.intent_override": intent_override,
                "rag.no_generate": no_generate,
            },
        ) as classify_span:
            with _span("Intent classification"):
                intent_result = self._classify_intent(
                    query_text=query_text,
                    intent_override=intent_override,
                    no_generate=no_generate,
                )

            logger.info(
                "Classified intent: %s (confidence=%.2f, method=%s)",
                intent_result.intent.value,
                intent_result.confidence,
                intent_result.method,
            )

            before_guard = intent_result
            intent_result, force_collection = self._apply_collection_guard(
                query_text=query_text,
                source_id=source_id,
                intent_override=intent_override,
                intent_result=intent_result,
            )

            retrieval_params = resolve_retrieval_params(config, intent_result.intent.value)
            generation_params = resolve_generation_params(intent_result.intent.value, config.mode)

            logger.info(
                "INTENT_CLASSIFIED | intent=%s | confidence=%.2f | method=%s | mode=%s",
                intent_result.intent.value,
                intent_result.confidence,
                intent_result.method,
                config.mode,
            )

            bypass_retrieval = is_low_information_query(query_text)

            set_span_attributes(
                classify_span,
                {
                    "output.value": intent_result.intent.value,
                    "rag.intent": intent_result.intent.value,
                    "rag.intent_confidence": intent_result.confidence,
                    "rag.intent_method": intent_result.method,
                    "rag.force_collection": force_collection,
                    "rag.bypass_retrieval": bypass_retrieval,
                    "rag.retrieval.top_k_fused": retrieval_params.top_k_fused,
                    "rag.retrieval.top_k_rerank": retrieval_params.top_k_rerank,
                    "rag.retrieval.top_k_final": retrieval_params.top_k_final,
                    "rag.retrieval.threshold": retrieval_params.reranker_threshold,
                },
            )

            return _ClassifyResult(
                intent_result=intent_result,
                before_guard_intent=before_guard,
                retrieval_params=retrieval_params,
                generation_params=generation_params,
                force_collection=force_collection,
                bypass_retrieval=bypass_retrieval,
            )

    def _step_retrieve(
        self,
        query_text: str,
        source_id: Optional[str],
        classified: _ClassifyResult,
        retrieval_engine: Any,
        cite: bool,
        no_generate: bool,
        profiler: Optional[LatencyProfiler] = None,
    ) -> _RetrieveResult:
        """Execute the retrieval phase — bypass, collection summary, or hybrid search."""
        config = self._model_config
        _span = profiler.span if profiler is not None else lambda _: nullcontext()

        with start_span(
            self._tracer,
            "rag.retrieve",
            span_kind=SPAN_KIND_RETRIEVER,
            attributes={
                "input.value": query_text,
                "input.mime_type": "text/plain",
                "rag.intent": classified.intent_result.intent.value,
                "rag.source_id": source_id,
                "rag.no_generate": no_generate,
                "rag.citations_enabled": cite,
            },
        ) as retrieve_span:
            extra_instructions: Optional[str] = None
            context = ""
            results: list = []
            source_ids: list = []
            context_docs: list = []
            retrieval_metrics = None
            generator_preload_future = None

            if classified.bypass_retrieval:
                if cite:
                    logger.info(
                        "Auto-disabling citations: no retrieval context for unclear query"
                    )
                    cite = False
                extra_instructions = (
                    "The user query is unclear or nonsensical. Ask a concise clarifying "
                    "question first, and optionally suggest 2-3 concrete ways they can rephrase it."
                )

            elif classified.intent_result.intent == Intent.COLLECTION:
                context, source_ids, cite = self._handle_collection(
                    config=config,
                    no_generate=no_generate,
                    citations_enabled=cite,
                )
                if classified.force_collection:
                    extra_instructions = (
                        "If the question asks which source/document matches a criterion, "
                        "name the matching source IDs explicitly in the first sentence, "
                        "then justify briefly from the summaries. If none match, say that clearly."
                    )
                if context is None:
                    set_span_attributes(
                        retrieve_span,
                        {
                            "rag.retrieve.path": "collection",
                            "rag.retrieve.results_count": 0,
                            "rag.retrieve.sources_count": 0,
                            "output.mime_type": "text/plain",
                            "output.value": "No collection documents found",
                            "rag.citations_enabled": cite,
                        },
                    )
                    set_retrieval_documents(retrieve_span, [])
                    return _RetrieveResult(
                        context=None,
                        results=[],
                        source_ids=[],
                        context_docs=[],
                        cite=cite,
                        extra_instructions=extra_instructions,
                        retrieval_metrics=None,
                        generator_preload_future=None,
                    )

            else:
                if not no_generate:
                    generator_preload_future = self._start_generator_preload()

                with _span("Retrieval (hybrid search + rerank)"):
                    results = retrieval_engine.search(
                        query_text,
                        source_id=source_id,
                        params=classified.retrieval_params,
                        retrieval_budget=config.retrieval_budget,
                        intent=classified.intent_result.intent.value,
                    )

                source_ids = sorted(
                    {
                        r.metadata.get("source_id")
                        for r in results
                        if r.metadata.get("source_id")
                    }
                )
                context_docs = [
                    (r.parent_text if r.parent_text else r.text)
                    for r in results
                    if (r.parent_text if r.parent_text else r.text)
                ]

            retrieval_metrics = results[0].metrics if results and results[0].metrics else None

            _scores = [r.score for r in results] if results else []
            top_score = max(_scores) if _scores else 0.0
            low_score = min(_scores) if _scores else 0.0
            mean_score = (sum(_scores) / len(_scores)) if _scores else 0.0
            top_results_preview = []
            for rank, result in enumerate(results[:5], start=1):
                metadata = result.metadata if isinstance(result.metadata, dict) else {}
                top_results_preview.append(
                    {
                        "rank": rank,
                        "source_id": metadata.get("source_id"),
                        "display_page": metadata.get("display_page"),
                        "page_number": metadata.get("page_number"),
                        "chunk_id": result.child_id,
                        "score": round(float(result.score), 6),
                    }
                )
            retrieval_documents = _build_openinference_retrieval_documents(results)
            logger.info(
                "INTENT_AWARE | intent=%s | retrieval_results=%d | top_score=%.4f | low_score=%.4f | "
                "params: top_k_dense=%d top_k_fused=%d top_k_rerank=%d top_k_final=%d threshold=%.4f",
                classified.intent_result.intent.value,
                len(results),
                top_score,
                low_score,
                classified.retrieval_params.top_k_dense,
                classified.retrieval_params.top_k_fused,
                classified.retrieval_params.top_k_rerank,
                classified.retrieval_params.top_k_final,
                classified.retrieval_params.reranker_threshold,
            )

            set_span_attributes(
                retrieve_span,
                {
                    "rag.retrieve.path": (
                        "bypass"
                        if classified.bypass_retrieval
                        else "collection"
                        if classified.intent_result.intent == Intent.COLLECTION
                        else "hybrid"
                    ),
                    "rag.retrieve.results_count": len(results),
                    "rag.retrieve.sources_count": len(source_ids),
                    "rag.retrieve.context_docs_count": len(context_docs),
                    "rag.retrieve.top_score": top_score,
                    "rag.retrieve.low_score": low_score,
                    "rag.retrieve.mean_score": mean_score,
                    "rag.retrieval.top_k_dense": classified.retrieval_params.top_k_dense,
                    "rag.retrieval.top_k_fused": classified.retrieval_params.top_k_fused,
                    "rag.retrieval.top_k_rerank": classified.retrieval_params.top_k_rerank,
                    "rag.retrieval.top_k_final": classified.retrieval_params.top_k_final,
                    "rag.retrieval.threshold": classified.retrieval_params.reranker_threshold,
                    "rag.retrieve.preload_started": generator_preload_future is not None,
                    "rag.retrieve.extra_instructions": bool(extra_instructions),
                    "rag.retrieve.top_results": top_results_preview,
                    "output.mime_type": "text/plain",
                    "output.value": f"{len(results)} retrieved passages",
                    "rag.citations_enabled": cite,
                },
            )
            set_retrieval_documents(retrieve_span, retrieval_documents)
            if retrieval_metrics is not None:
                set_span_attributes(
                    retrieve_span,
                    {
                        "rag.retrieve.timing_total_ms": retrieval_metrics.timing.total_ms,
                        "rag.retrieve.timing_hybrid_ms": retrieval_metrics.timing.hybrid_search_ms,
                        "rag.retrieve.timing_rerank_ms": retrieval_metrics.timing.rerank_ms,
                        "rag.retrieve.threshold_value": retrieval_metrics.threshold.threshold_value,
                        "rag.retrieve.threshold_items_after": retrieval_metrics.threshold.items_after_threshold,
                        "rag.retrieve.threshold_safety_net": retrieval_metrics.threshold.safety_net_triggered,
                        "rag.retrieve.parents_deduplicated": retrieval_metrics.deduplication.parents_deduplicated,
                    },
                )

            return _RetrieveResult(
                context=context,
                results=results,
                source_ids=source_ids,
                context_docs=context_docs,
                cite=cite,
                extra_instructions=extra_instructions,
                retrieval_metrics=retrieval_metrics,
                generator_preload_future=generator_preload_future,
            )

    def _step_pack_budget(
        self,
        retrieved: _RetrieveResult,
        config: ModelConfig,
        generator: Optional[Any],
    ) -> _PackResult:
        """Pack docs into the token budget and format citations.

        When *generator* is ``None`` (the ``no_generate`` path) the docs are
        formatted without token budgeting — using :func:`_dedupe_context` so
        both code paths get deduplication.
        """
        with start_span(
            self._tracer,
            "rag.budget.pack",
            span_kind=SPAN_KIND_CHAIN,
            attributes={
                "rag.retrieval.docs_in": len(retrieved.context_docs),
                "rag.citations_enabled": retrieved.cite,
                "rag.retrieval_budget": config.retrieval_budget,
            },
        ) as budget_span:
            source_legend: Optional[str] = None
            result_metadatas: list = (
                [
                    r.metadata
                    for r in retrieved.results
                    if (r.parent_text if r.parent_text else r.text)
                ]
                if retrieved.context_docs and retrieved.results
                else []
            )
            citation_list: list = []
            packed_retrieval_results: list = []
            pack_result_obj = None
            packed_metadatas: list = []
            context = retrieved.context
            cite = retrieved.cite
            budget_metrics = None

            if generator is not None and retrieved.context_docs:
                pack_result_obj = enforce_token_budget(
                    docs=retrieved.context_docs,
                    max_tokens=config.retrieval_budget,
                    tokenizer=generator.tokenizer,
                    consecutive_fail_threshold=3,
                    allow_truncation=True,
                    log=logger,
                )
                packed_metadatas = [
                    result_metadatas[i]
                    for i in pack_result_obj.packed_indices
                    if i < len(result_metadatas)
                ]
                packed_chunk_ids = [
                    (
                        retrieved.results[i].child_id
                        if i < len(retrieved.results)
                        else ""
                    )
                    for i in pack_result_obj.packed_indices
                ]

                if cite and packed_metadatas:
                    context, source_mapping = format_context_with_citations(
                        texts=pack_result_obj.packed_docs,
                        metadatas=packed_metadatas,
                        chunk_ids=packed_chunk_ids,
                    )
                    source_legend = build_source_legend(source_mapping)

                    for ci, pack_idx in enumerate(pack_result_obj.packed_indices):
                        if pack_idx >= len(retrieved.results):
                            continue
                        r = retrieved.results[pack_idx]
                        packed_retrieval_results.append(r)
                        citation_list.append(
                            {
                                "index": ci + 1,
                                "source_id": r.metadata.get("source_id", ""),
                                "chunk_id": r.child_id,
                                "page_number": r.metadata.get("page_number"),
                                "display_page": r.metadata.get("display_page"),
                                "start_page": r.metadata.get("start_page"),
                                "end_page": r.metadata.get("end_page"),
                                "header_path": r.metadata.get("header_path", ""),
                                "chunk_text": r.text if r.text else "",
                            }
                        )
                elif cite:
                    logger.info(
                        "Auto-disabling citations: packed context missing metadata"
                    )
                    cite = False
                else:
                    context = "\n\n".join(pack_result_obj.packed_docs)

                budget_metrics = BudgetMetrics(
                    budget_tokens=config.retrieval_budget,
                    used_tokens=pack_result_obj.used_tokens,
                    utilization_pct=(
                        100 * pack_result_obj.used_tokens / config.retrieval_budget
                        if config.retrieval_budget > 0
                        else 0
                    ),
                    avg_doc_tokens=(
                        pack_result_obj.used_tokens / len(pack_result_obj.packed_docs)
                        if pack_result_obj.packed_docs
                        else 0
                    ),
                    docs_packed=len(pack_result_obj.packed_docs),
                    docs_skipped=pack_result_obj.skipped_count,
                    docs_truncated=pack_result_obj.truncated_count,
                )

            elif retrieved.context_docs:
                # no_generate=True path: format context without token budgeting.
                if cite and result_metadatas:
                    context_chunk_ids = [r.child_id for r in retrieved.results]
                    context, source_mapping = format_context_with_citations(
                        texts=retrieved.context_docs,
                        metadatas=result_metadatas,
                        chunk_ids=context_chunk_ids,
                    )
                    source_legend = build_source_legend(source_mapping)
                elif cite:
                    cite = False
                else:
                    context = _dedupe_context(retrieved.context_docs)

            if packed_retrieval_results:
                packed_results_for_preview = packed_retrieval_results
            elif pack_result_obj is not None and retrieved.results:
                packed_results_for_preview = [
                    retrieved.results[idx]
                    for idx in pack_result_obj.packed_indices
                    if idx < len(retrieved.results)
                ]
            else:
                packed_results_for_preview = list(retrieved.results)

            packed_source_ids = []
            packed_chunk_ids = []
            for result in packed_results_for_preview:
                if isinstance(result.child_id, str) and result.child_id:
                    packed_chunk_ids.append(result.child_id)
                metadata = result.metadata if isinstance(result.metadata, dict) else {}
                source_value = metadata.get("source_id")
                if isinstance(source_value, str) and source_value:
                    packed_source_ids.append(source_value)

            unique_packed_source_ids = sorted(set(packed_source_ids))
            unique_packed_chunk_ids = list(dict.fromkeys(packed_chunk_ids))
            packed_retrieval_documents = _build_openinference_retrieval_documents(
                packed_results_for_preview,
                limit=8,
                max_content_chars=220,
            )

            set_span_attributes(
                budget_span,
                {
                    "rag.citations_enabled": cite,
                    "rag.retrieval.docs_out": (
                        len(pack_result_obj.packed_docs)
                        if pack_result_obj is not None
                        else len(retrieved.context_docs)
                    ),
                    "rag.budget.docs_skipped": (
                        pack_result_obj.skipped_count if pack_result_obj is not None else 0
                    ),
                    "rag.budget.docs_truncated": (
                        pack_result_obj.truncated_count if pack_result_obj is not None else 0
                    ),
                    "rag.budget.used_tokens": (
                        pack_result_obj.used_tokens if pack_result_obj is not None else 0
                    ),
                    "rag.budget.utilization_pct": (
                        budget_metrics.utilization_pct if budget_metrics is not None else 0.0
                    ),
                    "rag.budget.citation_count": len(citation_list),
                    "rag.budget.has_source_legend": bool(source_legend),
                    "rag.budget.source_ids_count": len(retrieved.source_ids),
                    "rag.budget.packed_results_count": len(packed_results_for_preview),
                    "rag.budget.packed_source_ids": unique_packed_source_ids[:10],
                    "rag.budget.packed_chunk_ids": unique_packed_chunk_ids[:10],
                    "output.mime_type": "text/plain",
                    "output.value": f"Packed {len(packed_results_for_preview)} passages",
                    "rag.context_chars": len(context or ""),
                },
            )
            set_retrieval_documents(budget_span, packed_retrieval_documents)

            return _PackResult(
                context=context,
                cite=cite,
                source_legend=source_legend,
                result_metadatas=result_metadatas,
                budget_metrics=budget_metrics,
                citation_list=citation_list,
                packed_retrieval_results=packed_retrieval_results,
                pack_result=pack_result_obj,
                packed_metadatas=packed_metadatas,
            )

    # -- internal pipeline helpers -----------------------------------------

    def _handle_collection(
        self,
        *,
        config: ModelConfig,
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
            generator = self.ensure_generator()
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

    # -- cleanup -----------------------------------------------------------

    def close(self) -> None:
        """Release all resources."""
        self._embedding_model = None
        self._reranker = None
        self._generator = None
        self._preload_executor.shutdown(wait=False)
        gc.collect()
        _release_mlx_cache()
