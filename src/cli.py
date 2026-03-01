"""CLI entry point for the offline RAG system.

Thin wrapper around :class:`~src.rag_engine.RagEngine` — parses arguments and
prints results.  All pipeline logic lives in ``rag_engine.py``.
"""

from __future__ import annotations

import argparse
import logging
import os
from typing import Optional

from .metrics import format_metrics_summary, log_metrics
from .rag_engine import RagEngine, RagEngineConfig

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger(__name__)

_VALID_FTS_POLICIES = ("immediate", "deferred", "batch")


def _get_fts_policy_default() -> str:
    raw = os.getenv("RAG_FTS_REBUILD_POLICY", "deferred").strip().lower()
    if raw in _VALID_FTS_POLICIES:
        return raw
    if raw:
        logger.warning(
            "Invalid RAG_FTS_REBUILD_POLICY='%s'; falling back to 'deferred'", raw
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
            "Invalid RAG_FTS_REBUILD_BATCH_SIZE='%s'; falling back to 0", raw
        )
        return 0


def run() -> None:
    logging.getLogger("transformers.tokenization_utils_base").setLevel(logging.ERROR)

    parser = argparse.ArgumentParser(description="Offline RAG CLI")
    fts_policy_default = _get_fts_policy_default()
    fts_batch_size_default = _get_fts_batch_size_default()
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose logging",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # ---- ingest subcommand -----------------------------------------------
    ingest_parser = subparsers.add_parser("ingest", help="Ingest a document")
    ingest_parser.add_argument("file", help="File path (PDF or Markdown)")
    ingest_parser.add_argument("--source-id", required=True, help="Source identifier")
    ingest_parser.add_argument(
        "--page-number", type=int, default=None, help="Page number"
    )
    ingest_parser.add_argument(
        "--lance", default="data/lance", help="LanceDB directory"
    )
    ingest_parser.add_argument(
        "--collection", default="child_chunks", help="LanceDB table name"
    )
    ingest_parser.add_argument(
        "--fts-rebuild-policy",
        choices=list(_VALID_FTS_POLICIES),
        default=fts_policy_default,
        help="FTS index rebuild policy (default: %(default)s)",
    )
    ingest_parser.add_argument(
        "--fts-rebuild-batch-size",
        type=int,
        default=fts_batch_size_default,
        help="Row threshold for batch FTS rebuild (default: %(default)s)",
    )
    ingest_parser.add_argument(
        "--mode",
        choices=["regular", "deep-research"],
        default=None,
        help="Operating mode",
    )
    ingest_parser.add_argument(
        "--verbose", "-v", action="store_true", help="Verbose logging"
    )
    ingest_parser.add_argument(
        "--model", default=None, help="LLM model for summaries"
    )
    _summarize_group = ingest_parser.add_mutually_exclusive_group()
    _summarize_group.add_argument(
        "--summarize",
        action="store_true",
        dest="summarize",
        default=True,
        help="Generate per-source summary during ingest (default: enabled)",
    )
    _summarize_group.add_argument(
        "--no-summarize",
        action="store_false",
        dest="summarize",
        help="Disable summary generation",
    )

    # ---- query subcommand ------------------------------------------------
    query_parser = subparsers.add_parser("query", help="Query the RAG system")
    query_parser.add_argument("query", help="User query")
    query_parser.add_argument(
        "--lance", default="data/lance", help="LanceDB directory"
    )
    query_parser.add_argument(
        "--collection", default="child_chunks", help="LanceDB table name"
    )
    query_parser.add_argument(
        "--fts-rebuild-policy",
        choices=list(_VALID_FTS_POLICIES),
        default=fts_policy_default,
        help="FTS index rebuild policy (default: %(default)s)",
    )
    query_parser.add_argument(
        "--fts-rebuild-batch-size",
        type=int,
        default=fts_batch_size_default,
        help="Row threshold for batch FTS rebuild (default: %(default)s)",
    )
    query_parser.add_argument(
        "--mode",
        choices=["regular", "deep-research"],
        default=None,
        help="Operating mode",
    )
    query_parser.add_argument(
        "--verbose", "-v", action="store_true", help="Verbose logging"
    )
    query_parser.add_argument(
        "--source-id", default=None, help="Restrict to a specific source"
    )
    query_parser.add_argument(
        "--list-sources", action="store_true", help="List sources and exit"
    )
    query_parser.add_argument("--model", default=None, help="LLM model path/ID")
    query_parser.add_argument(
        "--no-generate",
        action="store_true",
        help="Show retrieved context without LLM generation",
    )
    query_parser.add_argument(
        "--intent",
        choices=[
            "overview",
            "summarize",
            "explain",
            "analyze",
            "compare",
            "critique",
            "factual",
            "collection",
            "extract",
            "timeline",
            "how_to",
            "quote_evidence",
        ],
        default=None,
        help="Override automatic intent classification",
    )
    query_parser.add_argument(
        "--intent-confidence-threshold",
        type=float,
        default=0.6,
        help="Minimum intent classification confidence (default: 0.6)",
    )
    query_parser.add_argument(
        "--no-llm-fallback",
        action="store_true",
        help="Disable LLM intent fallback (heuristic only).",
    )
    query_parser.add_argument(
        "--llm-fallback-threshold",
        type=float,
        default=0.70,
        help="Heuristic confidence below this triggers LLM fallback (default: 0.70)",
    )
    query_parser.add_argument(
        "--intent-model",
        default="mlx-community/LFM2-8B-A1B-4bit",
        help="Model for LLM intent fallback",
    )

    query_parser.add_argument(
        "--cite",
        action="store_true",
        default=None,
        help="Enable inline citations (Academic Mode)",
    )
    query_parser.add_argument(
        "--no-cite",
        action="store_true",
        default=None,
        help="Disable inline citations",
    )
    query_parser.add_argument(
        "--latency",
        action="store_true",
        help="Enable detailed latency profiling",
    )
    query_parser.add_argument(
        "--dump-prompt",
        action="store_true",
        default=False,
        help=(
            "Print the exact prompt (system + user messages) that would be sent to the LLM, "
            "then exit without running generation. Useful for benchmarking other models "
            "against the same retrieval context."
        ),
    )

    args = parser.parse_args()

    if getattr(args, "cite", None) and getattr(args, "no_cite", None):
        parser.error("Conflicting flags: use only one of --cite or --no-cite.")
    if getattr(args, "fts_rebuild_batch_size", 0) < 0:
        parser.error("--fts-rebuild-batch-size must be >= 0.")
    if getattr(args, "dump_prompt", False) and getattr(args, "no_generate", False):
        parser.error("Conflicting flags: --dump-prompt and --no-generate are mutually exclusive.")

    verbose = getattr(args, "verbose", False)

    # -- resolve citations -------------------------------------------------
    citations: Optional[bool] = None
    if getattr(args, "cite", None) is True:
        citations = True
    elif getattr(args, "no_cite", None) is True:
        citations = False

    # -- build engine config -----------------------------------------------
    engine_cfg = RagEngineConfig(
        lance_dir=args.lance,
        collection=args.collection,
        mode=getattr(args, "mode", None),
        model=getattr(args, "model", None),
        fts_rebuild_policy=args.fts_rebuild_policy,
        fts_rebuild_batch_size=args.fts_rebuild_batch_size,
        citations_enabled=citations,

        intent_confidence_threshold=getattr(
            args, "intent_confidence_threshold", 0.6
        ),
        llm_fallback=not getattr(args, "no_llm_fallback", False),
        llm_fallback_threshold=getattr(args, "llm_fallback_threshold", 0.70),
        intent_model=getattr(args, "intent_model", "mlx-community/LFM2-8B-A1B-4bit"),
        verbose=verbose,
        latency=getattr(args, "latency", False) and args.command == "query",
    )

    engine = RagEngine(engine_cfg)
    config = engine.model_config

    mode_source = (
        "CLI"
        if getattr(args, "mode", None)
        else "env"
        if os.getenv("RAG_MODE")
        else "auto"
    )
    print(f"\n[Hardware: {config.system_ram_gb:.0f}GB | Mode: {config.mode} ({mode_source})]")
    print(f"[LLM: {config.llm_model} | Quant: {config.quantization}]")
    print(f"[Context: {config.context_window:,} | Budget: {config.retrieval_budget:,}]\n")

    # ---- ingest ----------------------------------------------------------
    if args.command == "ingest":
        result = engine.ingest(
            args.file,
            source_id=args.source_id,
            page_number=args.page_number,
            summarize=args.summarize,
        )
        print(f"Ingested {result.parents_count} parents and {result.children_count} children.")
        if result.summarized:
            print(f"Stored summary for source: {result.source_id}")
        return

    # ---- list sources ----------------------------------------------------
    if args.command == "query" and args.list_sources:
        sources = engine.list_sources()
        if not sources:
            print("No sources found in the database.")
        else:
            print("Available sources:")
            for source in sources:
                print(f"- {source}")
        return

    # ---- query -----------------------------------------------------------
    # Pre-load retrieval models in parallel for query performance
    engine.load_retrieval_models()

    result = engine.query(
        args.query,
        source_id=args.source_id,
        intent_override=args.intent,
        citations_enabled=citations,
        no_generate=args.no_generate,
        dump_prompt=args.dump_prompt,
    )

    # -- display retrieval metrics -----------------------------------------
    if result.retrieval_metrics:
        if verbose:
            log_metrics(result.retrieval_metrics, config.mode, logger)
        print(f"[Retrieval: {format_metrics_summary(result.retrieval_metrics)}]")
    if getattr(args, "latency", False) and result.latency_report:
        print(result.latency_report)

    # -- dump-prompt output ------------------------------------------------
    if args.dump_prompt:
        if result.prompt_messages is None:
            print("[ERROR] No prompt was generated (pipeline may have exited early).")
            return
        cite_mode = "Academic" if result.citations_enabled else "Casual"
        print(
            f"\n[Intent: {result.intent.intent.value} | "
            f"Confidence: {result.intent.confidence:.2f} | "
            f"Method: {result.intent.method} | "
            f"Citations: {cite_mode}]"
        )
        if result.source_ids:
            print(f"[Sources: {', '.join(result.source_ids)}]\n")
        sep = "=" * 72
        for i, msg in enumerate(result.prompt_messages):
            role = msg["role"].upper()
            content = msg["content"]
            print(f"{sep}")
            print(f"[MESSAGE {i + 1} — {role}]")
            print(sep)
            print(content)
        print(sep)
        return

    # -- no-generate output ------------------------------------------------
    if args.no_generate:
        print("Top retrieved context:\n")
        print(
            f"[Intent: {result.intent.intent.value} "
            f"(confidence: {result.intent.confidence:.2f})]\n"
        )
        if result.source_ids:
            print(f"[Sources: {', '.join(result.source_ids)}]\n")
        if result.context:
            print(result.context)
        else:
            print("No context retrieved.")
        return

    # -- full answer output ------------------------------------------------
    cite_mode = "Academic" if result.citations_enabled else "Casual"
    print(
        f"\n[Intent: {result.intent.intent.value} | "
        f"Confidence: {result.intent.confidence:.2f} | "
        f"Method: {result.intent.method} | "
        f"Citations: {cite_mode}]"
    )
    if result.source_ids:
        print(f"[Sources: {', '.join(result.source_ids)}]\n")
    else:
        print()
    print(result.answer)


if __name__ == "__main__":
    run()
