from __future__ import annotations

import argparse
import json
import logging
import re
import warnings
import textwrap
from pathlib import Path
from typing import Iterable, Optional

warnings.filterwarnings(
    "ignore",
    message=r"urllib3 v2 only supports OpenSSL.*",
)

from .config import select_model_config
from .generation import build_prompt
from .generator import MlxGenerator
from .ingest import ingest_file_to_storage
from .intent import Intent, IntentClassifier, IntentResult
from .retrieval import RetrievalEngine
from .storage import StorageConfig, StorageEngine

# Configure logging for intent classification
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def _dedupe_context(texts: Iterable[str]) -> str:
    seen = set()
    ordered = []
    for text in texts:
        cleaned = text.strip()
        if not cleaned or cleaned in seen:
            continue
        seen.add(cleaned)
        ordered.append(cleaned)
    return "\n\n".join(ordered)


# ----- Output Sanitization -----
# Patterns that indicate instruction leakage or repetition artifacts
_INSTRUCTION_PATTERNS = [
    re.compile(r"\s*Important:.*$", re.IGNORECASE | re.DOTALL),
    re.compile(r"\s*Grounding rule:.*$", re.IGNORECASE | re.DOTALL),
    re.compile(r"\s*Base your answer.*$", re.IGNORECASE | re.DOTALL),
    re.compile(r"\s*Do not substitute.*$", re.IGNORECASE | re.DOTALL),
    re.compile(r"\s*Task:.*$", re.IGNORECASE | re.DOTALL),
    re.compile(r"\s*Format:.*$", re.IGNORECASE | re.DOTALL),
    re.compile(r"\s*Tone:.*$", re.IGNORECASE | re.DOTALL),
]

# Pattern to detect repeating headers/page numbers (e.g., "4. A Review of..." repeated)
_REPETITION_PATTERN = re.compile(
    r"(\d+\.\s+[A-Z][^.]{10,50}\.?)(\s*\1)+",
    re.IGNORECASE,
)

# Common "chatter" phrases that slip past stop tokens (case-insensitive)
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
]


def _strip_chatter(text: str) -> str:
    """
    Remove common trailing chatter phrases that may slip past stop tokens.
    Case-insensitive matching.
    """
    if not text:
        return text
    
    result = text
    text_lower = result.lower()
    
    for phrase in _CHATTER_PHRASES:
        idx = text_lower.rfind(phrase)
        if idx != -1:
            # Only strip if it's near the end (last 20% of text)
            if idx > len(result) * 0.8:
                result = result[:idx].rstrip()
                text_lower = result.lower()
    
    return result


def _sanitize_output(text: str) -> str:
    """
    Post-process LLM output to remove noise and artifacts.
    
    - Strips trailing instruction-like phrases
    - Removes meta-commentary and chatter
    - Truncates repeating headers/page citations
    - Cleans up incomplete sentences
    """
    if not text:
        return text
    
    result = text
    
    # Remove instruction leakage from the end
    for pattern in _INSTRUCTION_PATTERNS:
        result = pattern.sub("", result)
    
    # Remove chatter phrases
    result = _strip_chatter(result)
    
    # Remove repeating headers/page numbers
    result = _REPETITION_PATTERN.sub(r"\1", result)
    
    # Remove any trailing incomplete sentences after stop (ends mid-word or with odd punctuation)
    # Find last complete sentence
    lines = result.rstrip().split("\n")
    if lines:
        last_line = lines[-1]
        # If last line looks incomplete (no ending punctuation, or ends with "the", "a", "to", etc.)
        incomplete_endings = re.compile(r"\b(the|a|an|to|of|in|for|and|or|but|is|are|was|were|that|this|with)\s*$", re.IGNORECASE)
        if incomplete_endings.search(last_line) or (last_line and last_line[-1] not in ".!?:"):
            # Try to find a good truncation point
            for end_char in [". ", ".\n", "! ", "!\n", "? ", "?\n"]:
                last_good = result.rfind(end_char)
                if last_good > len(result) * 0.5:  # Only truncate if we keep most of the content
                    result = result[:last_good + 1]
                    break
    
    return result.strip()


# ----- Query Expansion -----
# Query expansion terms by intent (used when --enable-query-expansion is set)
# NOTE: expansion_enabled is hardcoded to False by default - only active with explicit flag
_EXPANSION_TERMS: dict[Intent, list[str]] = {
    Intent.OVERVIEW: [],  # No expansion for overview - keep high-level
    Intent.SUMMARIZE: ["main argument", "thesis", "conclusion", "key points"],
    Intent.EXPLAIN: [],  # No expansion for explain - keep broad
    Intent.ANALYZE: ["criticism", "critique", "debate", "objection", "response", "controversy"],
}


def _expand_query(query: str, intent: Intent) -> str:
    """
    Expand query with intent-specific terms.
    
    This is behind a feature flag (--enable-query-expansion) because it
    modifies retrieval input and could degrade results for some queries.
    """
    terms = _EXPANSION_TERMS.get(intent, [])
    if not terms:
        return query
    
    # Append expansion terms to query
    expansion = " ".join(terms)
    return f"{query} {expansion}"


def _log_query(
    original_query: str,
    expanded_query: Optional[str],
    intent: IntentResult,
    expansion_enabled: bool,
) -> None:
    """
    Log query details for analysis and A/B comparison.
    
    Logs: {original_query, expanded_query, intent, expansion_enabled}
    """
    log_entry = {
        "original_query": original_query,
        "expanded_query": expanded_query,
        "intent": intent.intent.value,
        "intent_confidence": intent.confidence,
        "intent_method": intent.method,
        "expansion_enabled": expansion_enabled,
    }
    logger.info(f"Query log: {json.dumps(log_entry)}")


def run() -> None:
    logging.getLogger("transformers.tokenization_utils_base").setLevel(logging.ERROR)

    parser = argparse.ArgumentParser(description="Offline RAG CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    ingest_parser = subparsers.add_parser("ingest", help="Ingest a Markdown file")
    ingest_parser.add_argument("file", help="Markdown file path")
    ingest_parser.add_argument("--source-id", required=True, help="Source identifier")
    ingest_parser.add_argument("--page-number", type=int, default=None, help="Page number")
    ingest_parser.add_argument("--sqlite", default="data/context.sqlite", help="SQLite DB path")
    ingest_parser.add_argument("--chroma", default="data/chroma", help="Chroma persistence dir")
    ingest_parser.add_argument("--bm25", default="data/bm25.json", help="BM25 JSON path")
    ingest_parser.add_argument("--collection", default="child_chunks", help="Chroma collection name")
    ingest_parser.add_argument("--tier", default=None, help="Override hardware tier")

    query_parser = subparsers.add_parser("query", help="Query the RAG system")
    query_parser.add_argument("query", help="User query")
    query_parser.add_argument("--sqlite", default="data/context.sqlite", help="SQLite DB path")
    query_parser.add_argument("--chroma", default="data/chroma", help="Chroma persistence dir")
    query_parser.add_argument("--bm25", default="data/bm25.json", help="BM25 JSON path")
    query_parser.add_argument("--collection", default="child_chunks", help="Chroma collection name")
    query_parser.add_argument("--tier", default=None, help="Override hardware tier")
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
    # Intent classification options
    query_parser.add_argument(
        "--intent",
        choices=["overview", "summarize", "explain", "analyze"],
        default=None,
        help="Override automatic intent classification",
    )
    query_parser.add_argument(
        "--intent-confidence-threshold",
        type=float,
        default=0.6,
        help="Minimum confidence for intent classification (default: 0.6)",
    )
    query_parser.add_argument(
        "--no-llm-intent",
        action="store_true",
        help="Use heuristic-only intent classification (faster, no extra LLM call)",
    )
    # Query expansion options (feature flag, default OFF)
    query_parser.add_argument(
        "--enable-query-expansion",
        action="store_true",
        help="Enable intent-based query expansion for retrieval (experimental)",
    )

    args = parser.parse_args()

    config = select_model_config(manual_tier=args.tier)

    try:
        from sentence_transformers import SentenceTransformer
    except Exception as exc:  # pragma: no cover - dependency runtime
        raise RuntimeError("sentence-transformers is required for embeddings.") from exc

    try:
        from FlagEmbedding import FlagReranker
    except Exception as exc:  # pragma: no cover - dependency runtime
        raise RuntimeError("FlagEmbedding is required for reranking.") from exc

    embedding_model = SentenceTransformer(config.embedding_model, device=config.embedding_device)

    storage = StorageEngine(
        StorageConfig(
            sqlite_path=Path(args.sqlite),
            chroma_dir=Path(args.chroma),
            chroma_collection=args.collection,
        )
    )

    bm25_path = Path(args.bm25)

    if args.command == "ingest":
        parents_count, children_count = ingest_file_to_storage(
            args.file,
            source_id=args.source_id,
            page_number=args.page_number,
            storage=storage,
            embedding_model=embedding_model,
            bm25_path=bm25_path,
        )
        print(f"Ingested {parents_count} parents and {children_count} children.")
        return

    if not bm25_path.exists():
        raise FileNotFoundError(
            "BM25 index missing. Run 'ingest' to build indexes before querying."
        )

    storage.load_bm25(bm25_path)
    reranker = FlagReranker(config.reranker_model, use_fp16=True)

    retrieval = RetrievalEngine(storage=storage, embedding_model=embedding_model, reranker=reranker)

    # --- Intent Classification ---
    model_id = args.model or config.llm_model
    generator: Optional[MlxGenerator] = None
    
    # Determine intent (manual override or automatic classification)
    intent_result: Optional[IntentResult] = None
    
    if args.intent:
        # Manual override - use specified intent with full confidence
        intent_map = {
            "overview": Intent.OVERVIEW,
            "summarize": Intent.SUMMARIZE,
            "explain": Intent.EXPLAIN,
            "analyze": Intent.ANALYZE,
        }
        intent = intent_map[args.intent]
        intent_result = IntentResult(intent=intent, confidence=1.0, method="manual")
        logger.info(f"Using manual intent override: {intent.value}")
    else:
        # Automatic classification
        # Only load generator for LLM-based classification if needed
        use_llm_intent = not args.no_llm_intent and not args.no_generate
        
        if use_llm_intent:
            generator = MlxGenerator(model_id)
        
        classifier = IntentClassifier(
            generator=generator,
            confidence_threshold=args.intent_confidence_threshold,
            use_llm=use_llm_intent,
        )
        intent_result = classifier.classify(args.query)
        logger.info(
            f"Classified intent: {intent_result.intent.value} "
            f"(confidence={intent_result.confidence:.2f}, method={intent_result.method})"
        )
    
    # --- Query Expansion (behind feature flag) ---
    search_query = args.query
    if args.enable_query_expansion:
        search_query = _expand_query(args.query, intent_result.intent)
        logger.info(
            f"Query expansion enabled: original='{args.query}' -> expanded='{search_query}'"
        )
    
    # Log query details for analysis
    _log_query(
        original_query=args.query,
        expanded_query=search_query if args.enable_query_expansion else None,
        intent=intent_result,
        expansion_enabled=args.enable_query_expansion,
    )

    # --- Retrieval ---
    results = retrieval.search(search_query)
    context = _dedupe_context(
        [result.parent_text for result in results if result.parent_text]
    )

    if args.no_generate:
        print("Top retrieved context:\n")
        print(f"[Intent: {intent_result.intent.value} (confidence: {intent_result.confidence:.2f})]\n")
        for idx, result in enumerate(results, start=1):
            header_path = result.metadata.get("header_path", "")
            snippet = (result.parent_text or result.text or "").strip()
            snippet = textwrap.shorten(snippet, width=600, placeholder="...")
            print(f"[{idx}] score={result.score:.4f} header={header_path}")
            print(snippet)
            print("-" * 80)
        return

    # --- Generation ---
    prompt = build_prompt(context, args.query, intent=intent_result.intent)

    if generator is None:
        generator = MlxGenerator(model_id)
    
    answer = generator.generate(prompt)
    
    # --- Output Sanitization ---
    answer = _sanitize_output(answer)
    
    # Print intent info and answer
    print(f"\n[Intent: {intent_result.intent.value} | Confidence: {intent_result.confidence:.2f} | Method: {intent_result.method}]\n")
    print(answer)


if __name__ == "__main__":
    run()
