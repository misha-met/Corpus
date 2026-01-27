from __future__ import annotations

import argparse
import textwrap
from pathlib import Path
from typing import Iterable

from .config import select_model_config
from .generator import MlxGenerator
from .ingest import ingest_file_to_storage
from .retrieval import RetrievalEngine
from .storage import StorageConfig, StorageEngine


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


def run() -> None:
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
    query_parser.add_argument("--model", default="models/llm", help="Path to mlx-lm model")
    query_parser.add_argument(
        "--no-generate",
        action="store_true",
        help="Only show retrieved context without LLM generation",
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

    results = retrieval.search(args.query)
    context = _dedupe_context(
        [result.parent_text for result in results if result.parent_text]
    )

    if args.no_generate:
        print("Top retrieved context:\n")
        for idx, result in enumerate(results, start=1):
            header_path = result.metadata.get("header_path", "")
            snippet = (result.parent_text or result.text or "").strip()
            snippet = textwrap.shorten(snippet, width=600, placeholder="...")
            print(f"[{idx}] score={result.score:.4f} header={header_path}")
            print(snippet)
            print("-" * 80)
        return

    prompt = f"Context:\n{context}\n\nQuestion: {args.query}\nAnswer:"

    generator = MlxGenerator(args.model)
    answer = generator.generate(prompt)
    print(answer)


if __name__ == "__main__":
    run()
