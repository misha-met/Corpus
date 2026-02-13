from __future__ import annotations

import argparse
import gc
import json
import logging
import os
import re
import textwrap
import warnings
from pathlib import Path
from typing import Iterable, Optional

warnings.filterwarnings("ignore", message=r"urllib3 v2 only supports OpenSSL.*")

from .config import CITATIONS_ENABLED_DEFAULT, ModelConfig, select_mode_config
from .generation import build_messages
from .generator import GenerationConfig, MlxGenerator, count_tokens, enforce_token_budget
from .ingest import ingest_file_to_storage
from .intent import Intent, IntentClassifier, IntentResult, is_low_information_query
from .latency import LatencyProfiler
from .metrics import BudgetMetrics, RetrievalMetrics, ThresholdMetrics, format_metrics_summary, log_metrics
from .retrieval import RetrievalEngine, build_source_legend, format_context_with_citations
from .storage import StorageConfig, StorageEngine

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger(__name__)

_VALID_FTS_POLICIES = ("immediate", "deferred", "batch")


def _get_fts_policy_default() -> str:
    raw = os.getenv("RAG_FTS_REBUILD_POLICY", "deferred").strip().lower()
    if raw in _VALID_FTS_POLICIES:
        return raw
    if raw:
        logger.warning(
            "Invalid RAG_FTS_REBUILD_POLICY='%s'; falling back to 'deferred'",
            raw,
        )
    return "deferred"


def _get_fts_batch_size_default() -> int:
    raw = os.getenv("RAG_FTS_REBUILD_BATCH_SIZE", "0").strip()
    if not raw:
        return 0
    try:
        parsed = int(raw)
        if parsed < 0:
            raise ValueError
        return parsed
    except ValueError:
        logger.warning(
            "Invalid RAG_FTS_REBUILD_BATCH_SIZE='%s'; falling back to 0",
            raw,
        )
        return 0


def _dedupe_context(texts: Iterable[str]) -> str:
    seen: set[str] = set()
    unique_texts = []
    for text in texts:
        cleaned = text.strip()
        if cleaned and cleaned not in seen:
            seen.add(cleaned)
            unique_texts.append(cleaned)
    return "\n\n".join(unique_texts)


_INSTRUCTION_PATTERNS = [
    re.compile(r"^\s*Important:.*$", re.IGNORECASE | re.MULTILINE),
    re.compile(r"^\s*Grounding rule:.*$", re.IGNORECASE | re.MULTILINE),
    re.compile(r"^\s*Base your answer.*$", re.IGNORECASE | re.MULTILINE),
    re.compile(r"^\s*Do not substitute.*$", re.IGNORECASE | re.MULTILINE),
    re.compile(r"^\s*Task:.*$", re.IGNORECASE | re.MULTILINE),
    re.compile(r"^\s*Format:.*$", re.IGNORECASE | re.MULTILINE),
    re.compile(r"^\s*Tone:.*$", re.IGNORECASE | re.MULTILINE),
]

_REPETITION_PATTERN = re.compile(r"(\d+\.\s+[A-Z][^.]{10,50}\.?)(\s*\1)+", re.IGNORECASE)

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


def _strip_chatter(text: str) -> str:
    """Remove trailing chatter phrases near end of text (last 20%)."""
    if not text:
        return text
    result = text
    # Strip recursive denial: substantive answer followed by "insufficient info" disclaimer
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
    if difflib.SequenceMatcher(None, normalize(first_half), normalize(second_half)).ratio() >= 0.85:
        return first_half
    return text


_INCOMPLETE_ENDING = re.compile(r"\b(the|a|an|to|of|in|for|and|or|but|is|are|was|were|that|this|with)\s*$", re.IGNORECASE)


def _sanitize_output(text: str) -> str:
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
        if _INCOMPLETE_ENDING.search(last_line) or (last_line and last_line[-1] not in ".!?:"):
            for sentinel in [". ", ".\n", "! ", "!\n", "? ", "?\n"]:
                truncate_pos = result.rfind(sentinel)
                if truncate_pos > len(result) * 0.5:
                    result = result[:truncate_pos + 1]
                    break
    return result.strip()


_EXPANSION_TERMS: dict[Intent, list[str]] = {
    Intent.OVERVIEW: [],
    Intent.SUMMARIZE: ["main argument", "thesis", "conclusion", "key points"],
    Intent.EXPLAIN: [],
    Intent.ANALYZE: ["criticism", "critique", "debate", "objection", "response", "controversy"],
    Intent.COMPARE: ["compare", "contrast", "difference", "similarity"],
    Intent.CRITIQUE: ["criticism", "critique", "debate", "objection", "weakness", "strength"],
    Intent.FACTUAL: [],
    Intent.COLLECTION: [],
}


def _expand_query(query: str, intent: Intent) -> str:
    """Append intent-specific expansion terms to query (feature-flagged)."""
    terms = _EXPANSION_TERMS.get(intent, [])
    return f"{query} {' '.join(terms)}" if terms else query


def _log_query(original: str, expanded: Optional[str], intent: IntentResult, expansion_enabled: bool) -> None:
    logger.info(f"Query log: {json.dumps({'original_query': original, 'expanded_query': expanded, 'intent': intent.intent.value, 'intent_confidence': intent.confidence, 'intent_method': intent.method, 'expansion_enabled': expansion_enabled})}")


def _enable_offline_if_cached(config: ModelConfig) -> None:
    """Enable HF offline mode if all models for the active config are cached.

    Must be called **after** mode selection so the model list is accurate.
    Uses ``huggingface_hub.constants.HF_HUB_CACHE`` to honour custom cache paths.
    """
    try:
        from huggingface_hub.constants import HF_HUB_CACHE
        cache_dir = Path(HF_HUB_CACHE)
    except ImportError:
        cache_dir = Path.home() / ".cache" / "huggingface" / "hub"

    if not cache_dir.exists():
        logger.debug("HF cache dir %s does not exist; staying online", cache_dir)
        return

    required_models = [
        config.llm_model,
        config.embedding_model,
        config.reranker_model,
    ]

    for model_id in required_models:
        cache_folder = f"models--{model_id.replace('/', '--')}"
        model_cache = cache_dir / cache_folder
        if not model_cache.exists():
            logger.debug("Model %s not cached; staying online", model_id)
            return

        # Verify that at least one snapshot with actual files exists.
        snapshots_dir = model_cache / "snapshots"
        if not snapshots_dir.is_dir():
            logger.debug("Model %s cached but missing snapshots/; staying online", model_id)
            return
        snapshot_dirs = [d for d in snapshots_dir.iterdir() if d.is_dir()]
        if not snapshot_dirs:
            logger.debug("Model %s has empty snapshots/; staying online", model_id)
            return
        # Check that the latest snapshot contains at least one weight file.
        latest = max(snapshot_dirs, key=lambda d: d.stat().st_mtime)
        weight_exts = {".safetensors", ".bin", ".gguf", ".npz"}
        has_weights = any(
            f.suffix in weight_exts or f.name == "config.json"
            for f in latest.iterdir()
            if f.is_file() or f.is_symlink()
        )
        if not has_weights:
            logger.debug(
                "Model %s snapshot %s has no weight files; staying online",
                model_id, latest.name,
            )
            return

    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    logger.info("All models cached — enabled offline mode")


def run() -> None:
    logging.getLogger("transformers.tokenization_utils_base").setLevel(logging.ERROR)

    parser = argparse.ArgumentParser(description="Offline RAG CLI")
    fts_policy_default = _get_fts_policy_default()
    fts_batch_size_default = _get_fts_batch_size_default()
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging (show httpx, huggingface_hub, and detailed retrieval metrics)",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    ingest_parser = subparsers.add_parser("ingest", help="Ingest a Markdown file")
    ingest_parser.add_argument("file", help="Markdown file path")
    ingest_parser.add_argument("--source-id", required=True, help="Source identifier")
    ingest_parser.add_argument("--page-number", type=int, default=None, help="Page number")
    ingest_parser.add_argument("--lance", default="data/lance", help="LanceDB directory")
    ingest_parser.add_argument("--collection", default="child_chunks", help="LanceDB table name")
    ingest_parser.add_argument(
        "--fts-rebuild-policy",
        choices=list(_VALID_FTS_POLICIES),
        default=fts_policy_default,
        help=(
            "FTS index rebuild policy after ingest writes: immediate, deferred, or batch "
            "(default: %(default)s). Deferred rebuilds on the next query and may add a one-time "
            "search latency spike while the index is refreshed."
        ),
    )
    ingest_parser.add_argument(
        "--fts-rebuild-batch-size",
        type=int,
        default=fts_batch_size_default,
        help=(
            "Row threshold for --fts-rebuild-policy=batch (default: %(default)s). "
            "Ignored for other policies."
        ),
    )
    ingest_parser.add_argument(
        "--mode",
        choices=["regular", "power-deep-research"],
        default=None,
        help="Operating mode: regular (auto-scales to RAM), power-deep-research (80B model for deep research)",
    )
    # Allow verbosity flag after the subcommand as well
    ingest_parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose logging (show httpx, huggingface_hub, and detailed retrieval metrics)")
    ingest_parser.add_argument(
        "--model",
        default=None,
        help="Path or Hugging Face ID for mlx-lm model (used for summaries)",
    )
    _summarize_group = ingest_parser.add_mutually_exclusive_group()
    _summarize_group.add_argument(
        "--summarize",
        action="store_true",
        dest="summarize",
        default=True,
        help="Generate and store a per-source summary during ingest (default: enabled)",
    )
    _summarize_group.add_argument(
        "--no-summarize",
        action="store_false",
        dest="summarize",
        help="Disable automatic summary generation during ingest",
    )

    query_parser = subparsers.add_parser("query", help="Query the RAG system")
    query_parser.add_argument("query", help="User query")
    query_parser.add_argument("--lance", default="data/lance", help="LanceDB directory")
    query_parser.add_argument("--collection", default="child_chunks", help="LanceDB table name")
    query_parser.add_argument(
        "--fts-rebuild-policy",
        choices=list(_VALID_FTS_POLICIES),
        default=fts_policy_default,
        help=(
            "FTS index rebuild policy after ingest writes: immediate, deferred, or batch "
            "(default: %(default)s). Deferred rebuilds on the next query and may add a one-time "
            "search latency spike while the index is refreshed."
        ),
    )
    query_parser.add_argument(
        "--fts-rebuild-batch-size",
        type=int,
        default=fts_batch_size_default,
        help=(
            "Row threshold for --fts-rebuild-policy=batch (default: %(default)s). "
            "Ignored for other policies."
        ),
    )
    query_parser.add_argument(
        "--mode",
        choices=["regular", "power-deep-research"],
        default=None,
        help="Operating mode: regular (auto-scales to RAM), power-deep-research (80B model for deep research)",
    )
    # Allow verbosity flag after the subcommand as well
    query_parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose logging (show httpx, huggingface_hub, and detailed retrieval metrics)")
    query_parser.add_argument(
        "--source-id",
        default=None,
        help="Restrict retrieval to a specific source document",
    )
    query_parser.add_argument(
        "--list-sources",
        action="store_true",
        help="List available source documents and exit",
    )
    query_parser.add_argument(
        "--model",
        default=None,
        help="Path or Hugging Face ID for mlx-lm model",
    )
    query_parser.add_argument(
        "--no-generate",
        action="store_true",
        help="Only show retrieved context without LLM generation",
    )
    query_parser.add_argument(
        "--intent",
        choices=["overview", "summarize", "explain", "analyze", "compare", "critique", "factual", "collection"],
        default=None,
        help="Override automatic intent classification",
    )
    query_parser.add_argument(
        "--intent-confidence-threshold", type=float, default=0.6,
        help="Minimum confidence for intent classification (default: 0.6)",
    )
    query_parser.add_argument(
        "--llm-fallback", action="store_true",
        help="Deprecated compatibility flag. LLM intent fallback is now enabled by default.",
    )
    query_parser.add_argument(
        "--no-llm-fallback", action="store_true",
        help="Disable LLM intent fallback and use heuristic-only intent classification.",
    )
    query_parser.add_argument(
        "--llm-fallback-threshold", type=float, default=0.70,
        help="Heuristic confidence below this triggers LLM fallback (default: 0.70).",
    )
    query_parser.add_argument(
        "--intent-model",
        default="mlx-community/LFM2-8B-A1B-4bit",
        help="Model ID for LLM intent fallback (default: mlx-community/LFM2-8B-A1B-4bit).",
    )
    query_parser.add_argument(
        "--enable-query-expansion", action="store_true",
        help="Enable intent-based query expansion for retrieval (experimental)",
    )
    query_parser.add_argument(
        "--cite", action="store_true", default=None,
        help="Enable inline citations in output (Academic Mode). Formats context with source/page markers and requires [SourceID, p. X] citations.",
    )
    query_parser.add_argument(
        "--no-cite",
        action="store_true",
        default=None,
        help="Disable inline citations (overrides CITATIONS_ENABLED default).",
    )
    query_parser.add_argument(
        "--latency",
        action="store_true",
        help="Enable detailed latency profiling for every pipeline stage.",
    )

    args = parser.parse_args()

    if getattr(args, "cite", None) and getattr(args, "no_cite", None):
        parser.error("Conflicting flags: use only one of --cite or --no-cite.")
    if getattr(args, "fts_rebuild_batch_size", 0) < 0:
        parser.error("--fts-rebuild-batch-size must be >= 0.")

    # ---- logging verbosity ----
    verbose = getattr(args, "verbose", False)
    if not verbose:
        # Suppress noisy third-party loggers in normal mode
        for noisy in ("httpx", "huggingface_hub", "lancedb", "urllib3",
                       "sentence_transformers", "filelock", "fsspec"):
            logging.getLogger(noisy).setLevel(logging.WARNING)

    # ---- latency profiler ----
    latency_enabled = getattr(args, "latency", False) and args.command == "query"
    profiler = LatencyProfiler(enabled=latency_enabled)
    profiler.start_wall()

    def _print_latency_report(metrics: Optional[RetrievalMetrics]) -> None:
        profiler.end_wall()
        if profiler.enabled:
            if metrics:
                t = metrics.timing
                profiler.record("    └ Hybrid search (LanceDB)", t.hybrid_search_ms)
                profiler.record("    └ Deduplication", t.dedup_ms)
                profiler.record("    └ Reranking", t.rerank_ms)
            print(profiler.format_report())

    with profiler.span("Config / mode selection"):
        config = select_mode_config(manual_mode=getattr(args, 'mode', None))
        _enable_offline_if_cached(config)

    mode_source = "CLI" if getattr(args, 'mode', None) else "env" if os.getenv("RAG_MODE") else "auto"
    print(f"\n[Hardware: {config.system_ram_gb:.0f}GB | Mode: {config.mode} ({mode_source})]")
    print(f"[LLM: {config.llm_model} | Quant: {config.quantization}]")
    print(f"[Context: {config.context_window:,} | Budget: {config.retrieval_budget:,}]\n")

    try:
        from sentence_transformers import SentenceTransformer
    except Exception as exc:  # pragma: no cover - dependency runtime
        raise RuntimeError("sentence-transformers is required for embeddings.") from exc

    from .reranker import JinaRerankerMLX

    # ---- parallel model loading (embedding + reranker overlap) ----
    import concurrent.futures
    import time as _time

    _embed_result: list = [None]
    _reranker_result: list = [None]
    _embed_time_ms: list = [0.0]
    _reranker_time_ms: list = [0.0]

    def _load_embedding():
        t0 = _time.perf_counter()
        _embed_result[0] = SentenceTransformer(config.embedding_model, device=config.embedding_device)
        _embed_time_ms[0] = (_time.perf_counter() - t0) * 1000

    def _load_reranker():
        t0 = _time.perf_counter()
        _reranker_result[0] = JinaRerankerMLX(model_id=config.reranker_model)
        _reranker_time_ms[0] = (_time.perf_counter() - t0) * 1000

    # Only parallelize for query (not ingest); ingest doesn't need reranker.
    if args.command == "query" and not getattr(args, "list_sources", False):
        with profiler.span("Load models (parallel: embedding + reranker)"):
            with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
                pool.submit(_load_embedding)
                pool.submit(_load_reranker)
                pool.shutdown(wait=True)
        embedding_model = _embed_result[0]
        if profiler.enabled:
            profiler.record("  \u2514 Embedding model", _embed_time_ms[0], config.embedding_model.split("/")[-1])
            profiler.record("  \u2514 Reranker model", _reranker_time_ms[0], config.reranker_model.split("/")[-1])
    else:
        with profiler.span("Load embedding model"):
            embedding_model = SentenceTransformer(config.embedding_model, device=config.embedding_device)

    with profiler.span("Open storage / LanceDB"):
        storage = StorageEngine(
            StorageConfig(
                lance_dir=Path(args.lance),
                lance_table=args.collection,
                fts_rebuild_policy=args.fts_rebuild_policy,
                fts_rebuild_batch_size=args.fts_rebuild_batch_size,
            )
        )

    if args.command == "ingest":
        do_summarize = args.summarize
        generator: Optional[MlxGenerator] = None
        if do_summarize:
            model_id = args.model or config.llm_model
            generator = MlxGenerator(model_id)
        parents_count, children_count = ingest_file_to_storage(
            args.file,
            source_id=args.source_id,
            page_number=args.page_number,
            storage=storage,
            embedding_model=embedding_model,
            summarize=do_summarize,
            summary_generator=generator,
        )
        print(f"Ingested {parents_count} parents and {children_count} children.")
        if do_summarize:
            print(f"Stored summary for source: {args.source_id}")
        return

    if args.command == "query" and args.list_sources:
        sources = storage.list_source_ids()
        if not sources:
            print("No sources found in the database.")
        else:
            print("Available sources:")
            for source in sources:
                print(f"- {source}")
        return

    # Reranker was already loaded in parallel above
    reranker = _reranker_result[0] if _reranker_result[0] is not None else JinaRerankerMLX(model_id=config.reranker_model)

    retrieval = RetrievalEngine(
        storage=storage,
        embedding_model=embedding_model,
        reranker=reranker,
        config=config,
    )

    model_id = args.model or config.llm_model
    generator: Optional[MlxGenerator] = None
    
    if args.cite is True:
        citations_enabled = True
    elif args.no_cite is True:
        citations_enabled = False
    else:
        citations_enabled = CITATIONS_ENABLED_DEFAULT
    logger.info(f"Citations mode: {'ENABLED (Academic Mode)' if citations_enabled else 'DISABLED (Casual Mode)'}")
    
    intent_result: Optional[IntentResult] = None
    if args.intent:
        intent_map = {
            "overview": Intent.OVERVIEW, "summarize": Intent.SUMMARIZE,
            "explain": Intent.EXPLAIN, "analyze": Intent.ANALYZE,
            "compare": Intent.COMPARE, "critique": Intent.CRITIQUE,
            "factual": Intent.FACTUAL, "collection": Intent.COLLECTION,
        }
        intent_result = IntentResult(intent=intent_map[args.intent], confidence=1.0, method="manual")
        logger.info(f"Using manual intent override: {intent_result.intent.value}")
    else:
        with profiler.span("Intent classification"):
            llm_fallback_enabled = not args.no_generate and not getattr(args, "no_llm_fallback", False)
            llm_model_id = args.intent_model if llm_fallback_enabled else None
            llm_fallback_threshold = getattr(args, "llm_fallback_threshold", 0.70)

            classifier = IntentClassifier(
                confidence_threshold=args.intent_confidence_threshold,
                llm_model_id=llm_model_id,
                llm_fallback_threshold=llm_fallback_threshold,
                eager_load_llm=False,  # Lazy-load only if heuristic is uncertain
            )
            intent_result = classifier.classify(args.query)
            # Immediately release intent classifier to free ~4-5GB before LLM generation
            del classifier
            gc.collect()
            try:
                import mlx.core as mx
                mx.clear_cache()
            except Exception:
                pass
        logger.info(f"Classified intent: {intent_result.intent.value} (confidence={intent_result.confidence:.2f}, method={intent_result.method})")

    search_query = args.query
    if args.enable_query_expansion:
        search_query = _expand_query(args.query, intent_result.intent)
        logger.info(f"Query expansion enabled: original='{args.query}' -> expanded='{search_query}'")

    _log_query(args.query, search_query if args.enable_query_expansion else None, intent_result, args.enable_query_expansion)

    extra_instructions: Optional[str] = None
    query_looks_unclear = is_low_information_query(args.query)
    bypass_retrieval_unclear_query = (
        query_looks_unclear
        and not args.no_generate
    )

    if bypass_retrieval_unclear_query:
        logger.info(
            "Skipping retrieval for low-information query; delegating directly to generation model"
        )
        context = ""
        results = []
        source_ids = []
        parent_texts = []
        if citations_enabled:
            logger.info("Auto-disabling citations: no retrieval context for unclear query")
            citations_enabled = False
        extra_instructions = (
            "The user query is unclear or nonsensical. Ask a concise clarifying question first, "
            "and optionally suggest 2-3 concrete ways they can rephrase it."
        )

    # --- COLLECTION intent: always use document summaries ---
    if bypass_retrieval_unclear_query:
        pass
    elif intent_result.intent == Intent.COLLECTION:
        sources = storage.list_source_ids()
        if not sources:
            print("No documents found in the database.")
            del reranker, embedding_model, retrieval
            gc.collect()
            try:
                import mlx.core as mx
                mx.clear_cache()
            except Exception:
                pass
            return

        summaries = storage.get_source_summaries()
        missing = [source for source in sources if source not in summaries]
        if missing:
            if args.no_generate:
                print("Some sources are missing summaries. Re-run without --no-generate.")
                for source in missing:
                    print(f"- {source}")
                del reranker, embedding_model, retrieval
                gc.collect()
                try:
                    import mlx.core as mx
                    mx.clear_cache()
                except Exception:
                    pass
                return

            if generator is None:
                generator = MlxGenerator(model_id)

            for source in missing:
                p_texts = storage.get_parent_texts_by_source(source_id=source)
                context_text = "\n\n".join(p_texts)
                if len(context_text) > 12000:
                    context_text = context_text[:12000]
                summary_messages = build_messages(
                    context=context_text,
                    question="Summarize this document.",
                    intent=Intent.SUMMARIZE,
                    mode=config.mode,
                )
                summary_text = generator.generate_chat(summary_messages)
                storage.upsert_source_summary(source_id=source, summary=summary_text)
            summaries = storage.get_source_summaries()

        summary_blocks = [
            f"Source: {source}\nSummary: {summaries[source]}"
            for source in sources
            if source in summaries
        ]
        context = "\n\n".join(summary_blocks)
        results = []
        source_ids = sources
        parent_texts = []
        if citations_enabled:
            logger.info("Auto-disabling citations: context is built from document summaries (no chunk markers)")
            citations_enabled = False

    # --- SUMMARIZE with multiple sources (no --source-id): use summaries ---
    elif intent_result.intent == Intent.SUMMARIZE and not args.source_id:
        sources = storage.list_source_ids()
        if len(sources) > 1:
            summaries = storage.get_source_summaries()
            missing = [source for source in sources if source not in summaries]
            if missing:
                if args.no_generate:
                    print(
                        "Multiple documents are available, but some sources are missing summaries. "
                        "Re-run without --no-generate or specify --source-id."
                    )
                    for source in missing:
                        print(f"- {source}")
                    return

                if generator is None:
                    generator = MlxGenerator(model_id)

                for source in missing:
                    parent_texts = storage.get_parent_texts_by_source(source_id=source)
                    context_text = "\n\n".join(parent_texts)
                    if len(context_text) > 12000:
                        context_text = context_text[:12000]
                    summary_messages = build_messages(
                        context=context_text,
                        question="Summarize this document.",
                        intent=Intent.SUMMARIZE,
                        mode=config.mode,
                    )
                    summary_text = generator.generate_chat(summary_messages)
                    storage.upsert_source_summary(
                        source_id=source,
                        summary=summary_text,
                    )
                summaries = storage.get_source_summaries()
            summary_blocks = [
                f"Source: {source}\nSummary: {summaries[source]}"
                for source in sources
                if source in summaries
            ]
            context = "\n\n".join(summary_blocks)
            results = []
            source_ids = sources
            parent_texts = []
            extra_instructions = (
                "Provide a single consolidated answer addressing the user's question. "
                "Do not output per-source summaries or repeat points."
            )
            # Summary-based context lacks [CHUNK START] markers, so
            # citations must be disabled to prevent hallucinated references.
            if citations_enabled:
                logger.info("Auto-disabling citations: context is built from document summaries (no chunk markers)")
                citations_enabled = False
        else:
            with profiler.span("Retrieval (hybrid search + rerank)"):
                results = retrieval.search(search_query, source_id=args.source_id)
            source_ids = sorted({r.metadata.get("source_id") for r in results if r.metadata.get("source_id")})
            parent_texts = [r.parent_text for r in results if r.parent_text]
    else:
        with profiler.span("Retrieval (hybrid search + rerank)"):
            results = retrieval.search(search_query, source_id=args.source_id)
        source_ids = sorted({r.metadata.get("source_id") for r in results if r.metadata.get("source_id")})
        parent_texts = [r.parent_text for r in results if r.parent_text]

    retrieval_metrics: Optional[RetrievalMetrics] = results[0].metrics if results and results[0].metrics else None

    # Release retrieval-only models to free memory before LLM generation.
    # On 32GB Apple Silicon, the reranker (~1.2GB) and embedding model (~2GB)
    # competing with the MLX LLM for unified memory can cause swap thrashing.
    with profiler.span("Memory cleanup (gc)"):
        del reranker, embedding_model, retrieval
        gc.collect()
        # Force MLX to release cached Metal buffers back to the system
        try:
            import mlx.core as mx
            mx.clear_cache()
        except Exception:
            pass
    logger.debug("Released reranker and embedding model to free memory for LLM generation")

    budget_metrics: Optional[BudgetMetrics] = None
    source_legend: Optional[str] = None
    # context may already be set by COLLECTION or multi-doc SUMMARIZE paths above
    if 'context' not in locals():
        context: str = ""
    has_parent_texts = 'parent_texts' in locals() and parent_texts
    result_metadatas: list[dict] = [r.metadata for r in results if r.parent_text] if has_parent_texts and results else []
    
    if not args.no_generate and has_parent_texts:
        if generator is None:
            generator = MlxGenerator(model_id)

        import time
        with profiler.span("Budget packing"):
            pack_start = time.perf_counter()
            pack_result = enforce_token_budget(
                docs=parent_texts,
                max_tokens=config.retrieval_budget,
                tokenizer=generator.tokenizer,
                consecutive_fail_threshold=3,
                allow_truncation=True,
                log=logger,
            )
            pack_time_ms = (time.perf_counter() - pack_start) * 1000

        packed_metadatas = [result_metadatas[i] for i in pack_result.packed_indices if i < len(result_metadatas)]
        
        if citations_enabled and packed_metadatas:
            context, source_mapping = format_context_with_citations(
                texts=pack_result.packed_docs,
                metadatas=packed_metadatas,
            )
            source_legend = build_source_legend(source_mapping)
            logger.info(f"Citations enabled: formatted {len(pack_result.packed_docs)} chunks with source markers")
        elif citations_enabled:
            logger.info("Auto-disabling citations: packed context is missing metadata for chunk markers")
            citations_enabled = False
        else:
            context = "\n\n".join(pack_result.packed_docs)

        budget_metrics = BudgetMetrics(
            budget_tokens=config.retrieval_budget,
            used_tokens=pack_result.used_tokens,
            utilization_pct=100 * pack_result.used_tokens / config.retrieval_budget if config.retrieval_budget > 0 else 0,
            avg_doc_tokens=pack_result.used_tokens / len(pack_result.packed_docs) if pack_result.packed_docs else 0,
            docs_packed=len(pack_result.packed_docs),
            docs_skipped=pack_result.skipped_count,
            docs_truncated=pack_result.truncated_count,
        )

        if retrieval_metrics:
            updated_timing = retrieval_metrics.timing
            updated_timing.budget_packing_ms = pack_time_ms
            updated_timing.total_ms += pack_time_ms
            
            retrieval_metrics = RetrievalMetrics(
                budget=budget_metrics,
                timing=updated_timing,
                reranker=retrieval_metrics.reranker,
                deduplication=retrieval_metrics.deduplication,
                threshold=retrieval_metrics.threshold,
                query=retrieval_metrics.query,
                mode=retrieval_metrics.mode,
            )

        logger.info(f"Budget packing: {pack_result.used_tokens:,}/{config.retrieval_budget:,} tokens ({budget_metrics.utilization_pct:.1f}%), {len(pack_result.packed_docs)} docs packed")
    elif has_parent_texts:
        if citations_enabled and result_metadatas:
            context, source_mapping = format_context_with_citations(
                texts=parent_texts,
                metadatas=result_metadatas,
            )
            source_legend = build_source_legend(source_mapping)
        elif citations_enabled:
            logger.info("Auto-disabling citations: retrieved context is missing metadata for chunk markers")
            citations_enabled = False
        else:
            context = _dedupe_context(parent_texts)

    if retrieval_metrics:
        if budget_metrics and retrieval_metrics.budget.budget_tokens == 0:
            retrieval_metrics = RetrievalMetrics(
                budget=budget_metrics,
                timing=retrieval_metrics.timing,
                reranker=retrieval_metrics.reranker,
                deduplication=retrieval_metrics.deduplication,
                threshold=retrieval_metrics.threshold,
                query=retrieval_metrics.query,
                mode=retrieval_metrics.mode,
            )
        if verbose:
            log_metrics(retrieval_metrics, config.mode, logger)
        print(f"[Retrieval: {format_metrics_summary(retrieval_metrics)}]")

    if args.no_generate:
        print("Top retrieved context:\n")
        print(f"[Intent: {intent_result.intent.value} (confidence: {intent_result.confidence:.2f})]\n")
        if source_ids:
            print(f"[Sources: {', '.join(source_ids)}]\n")
        if results:
            for idx, result in enumerate(results, start=1):
                header_path = result.metadata.get("header_path", "")
                snippet = (result.parent_text or result.text or "").strip()
                snippet = textwrap.shorten(snippet, width=600, placeholder="...")
                print(f"[{idx}] score={result.score:.4f} header={header_path}")
                print(snippet)
                print("-" * 80)
        elif context:
            print(context)
        else:
            print("No context retrieved.")
        _print_latency_report(retrieval_metrics)
        return

    with profiler.span("Build prompt / messages"):
        messages = build_messages(
            context,
            args.query,
            intent=intent_result.intent,
            extra_instructions=extra_instructions,
            citations_enabled=citations_enabled,
            source_legend=source_legend,
            mode=config.mode,
        )

    if generator is None:
        # Set conservative cache limit before loading the large LLM to ensure
        # headroom for KV cache growth during generation (~2-3GB for 16K context)
        try:
            import mlx.core as mx
            mx.set_cache_limit(0)  # Disable caching to maximize available memory
        except Exception:
            pass
        with profiler.span("Load LLM model"):
            generator = MlxGenerator(model_id)

    # Academic mode (citations): 600 tokens for concise, cited answers
    # Casual mode: 1200 tokens for comprehensive, long-form responses
    gen_config = GenerationConfig(
        max_tokens=600 if citations_enabled else 1200,
        context_window=config.context_window,
    )
    with profiler.span("LLM generation"):
        raw_answer = generator.generate_chat(messages, config=gen_config)
    with profiler.span("Output sanitisation"):
        answer = _sanitize_output(raw_answer)
    
    # Print intent info and answer
    cite_mode = "Academic" if citations_enabled else "Casual"
    print(f"\n[Intent: {intent_result.intent.value} | Confidence: {intent_result.confidence:.2f} | Method: {intent_result.method} | Citations: {cite_mode}]")
    if source_ids:
        print(f"[Sources: {', '.join(source_ids)}]\n")
    else:
        print()
    print(answer)

    # ---- latency report ----
    _print_latency_report(retrieval_metrics)


if __name__ == "__main__":
    run()
