"""Shared fixtures, mock factories, and test corpus for the RAG test suite.

This module provides deterministic test data and lightweight mocks for 
heavy ML dependencies (mlx-lm, sentence-transformers, Jina reranker) so
that correctness tests can run without GPU or large model downloads.
"""
from __future__ import annotations

import json
import logging
import os
import sys
import tempfile
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import pytest

# ---------------------------------------------------------------------------
# Ensure project root is on sys.path
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ---------------------------------------------------------------------------
# Structured logging for test runs
# ---------------------------------------------------------------------------

LOG_DIR = Path(__file__).parent / "logs"
LOG_DIR.mkdir(exist_ok=True)

_RUN_ID = time.strftime("%Y%m%d_%H%M%S")


def get_test_logger(name: str) -> logging.Logger:
    """Create a logger that writes structured JSON lines to tests/logs/."""
    logger = logging.getLogger(f"test.{name}")
    logger.setLevel(logging.DEBUG)
    if not logger.handlers:
        fh = logging.FileHandler(LOG_DIR / f"{_RUN_ID}_{name}.jsonl", mode="a")
        fh.setLevel(logging.DEBUG)

        class _JsonFormatter(logging.Formatter):
            def format(self, record: logging.LogRecord) -> str:
                entry = {
                    "ts": self.formatTime(record),
                    "level": record.levelname,
                    "logger": record.name,
                    "msg": record.getMessage(),
                }
                if hasattr(record, "data"):
                    entry["data"] = record.data  # type: ignore[attr-defined]
                return json.dumps(entry)

        fh.setFormatter(_JsonFormatter())
        logger.addHandler(fh)

        # Also log to console at INFO level for live feedback
        sh = logging.StreamHandler()
        sh.setLevel(logging.INFO)
        sh.setFormatter(logging.Formatter("[%(name)s] %(message)s"))
        logger.addHandler(sh)
    return logger


# ---------------------------------------------------------------------------
# Timing context manager
# ---------------------------------------------------------------------------

@dataclass
class TimingResult:
    """Result from a timed block."""
    label: str
    elapsed_ms: float
    context: dict[str, Any]


class Timer:
    """Context manager for timing blocks with structured metadata."""

    def __init__(self, label: str, **context: Any) -> None:
        self.label = label
        self.context = context
        self._start: float = 0.0
        self.result: Optional[TimingResult] = None

    def __enter__(self) -> "Timer":
        self._start = time.perf_counter()
        return self

    def __exit__(self, *args: Any) -> None:
        elapsed = (time.perf_counter() - self._start) * 1000
        self.result = TimingResult(self.label, elapsed, self.context)


# ---------------------------------------------------------------------------
# Mock embedding model
# ---------------------------------------------------------------------------

class MockEmbeddingModel:
    """Deterministic embedding model for testing.
    
    Returns fixed-dimension vectors based on hash of input text,
    so identical inputs always produce identical embeddings.
    """

    def __init__(self, dim: int = 384) -> None:
        self.dim = dim
        self.call_count = 0

    def encode(
        self,
        texts: list[str],
        normalize_embeddings: bool = False,
        **kwargs: Any,
    ) -> list[list[float]]:
        self.call_count += 1
        import hashlib
        embeddings = []
        for text in texts:
            h = hashlib.sha256(text.encode()).digest()
            # Stretch hash to fill dimension
            raw = []
            for i in range(self.dim):
                byte_idx = i % len(h)
                raw.append((h[byte_idx] + i) / 255.0 - 0.5)
            if normalize_embeddings:
                mag = max(sum(x * x for x in raw) ** 0.5, 1e-9)
                raw = [x / mag for x in raw]
            embeddings.append(raw)
        return embeddings


# ---------------------------------------------------------------------------
# Mock reranker
# ---------------------------------------------------------------------------

class MockReranker:
    """Deterministic reranker for testing.
    
    Scores based on word overlap between query and document.
    This is intentionally simple to make test outcomes predictable.
    """

    def __init__(self) -> None:
        self.call_count = 0
        self.last_batch_size = 0

    def compute_score(self, pairs: list[tuple[str, str]]) -> list[float]:
        self.call_count += 1
        self.last_batch_size = len(pairs)
        scores = []
        for query, doc in pairs:
            q_words = set(query.lower().split())
            d_words = set(doc.lower().split())
            if not q_words or not d_words:
                scores.append(0.0)
            else:
                overlap = len(q_words & d_words)
                # Return score in range roughly 0..1 like Jina cosine sim
                score = overlap / max(len(q_words), 1)
                scores.append(min(score, 1.0))
        return scores


# ---------------------------------------------------------------------------
# Mock tokenizer
# ---------------------------------------------------------------------------

class MockTokenizer:
    """Simple whitespace tokenizer for testing token budget logic."""

    def encode(self, text: str) -> list[int]:
        if not text:
            return []
        return list(range(len(text.split())))

    def decode(self, token_ids: list[int]) -> str:
        return " ".join(f"tok{i}" for i in token_ids)

    def tokenize(self, text: str) -> list[str]:
        return text.split()


# ---------------------------------------------------------------------------
# Test corpus generator
# ---------------------------------------------------------------------------

FIXED_CORPUS = [
    {
        "source_id": "test_doc_linguistics",
        "header_path": "Chapter 1 > Introduction",
        "page_number": 1,
        "page_label": "1",
        "text": (
            "Chomsky's theory of generative grammar revolutionized the study of linguistics "
            "in the mid-twentieth century. His review of Skinner's Verbal Behavior argued that "
            "language acquisition cannot be explained by behaviorist principles alone. The "
            "poverty of the stimulus argument suggests that children acquire linguistic "
            "competence despite limited and impoverished input from their environment. "
            "This nativist position contrasts sharply with empiricist accounts that emphasize "
            "the role of environmental factors. Universal Grammar is posited as an innate "
            "biological endowment shared by all humans, constituting the initial state of "
            "the language faculty. The principles and parameters framework attempts to "
            "characterize the set of possible human languages by identifying invariant "
            "principles and a finite set of binary parameters that differentiate languages."
        ),
    },
    {
        "source_id": "test_doc_linguistics",
        "header_path": "Chapter 2 > Methodology",
        "page_number": 2,
        "page_label": "2",
        "text": (
            "The methodology of generative linguistics relies heavily on native speaker "
            "intuitions about grammaticality. Introspective judgments serve as the primary "
            "source of data for constructing syntactic analyses. This approach has been "
            "criticized by corpus linguists and psycholinguists who argue for more empirical "
            "methods of data collection. The distinction between competence and performance "
            "is central to the Chomskyan paradigm, where competence refers to the idealized "
            "knowledge of language and performance to the actual use of language in concrete "
            "situations. This dichotomy has been challenged by usage-based approaches that "
            "view language structure as emergent from patterns of use."
        ),
    },
    {
        "source_id": "test_doc_philosophy",
        "header_path": "Section A > Epistemology",
        "page_number": 1,
        "page_label": "1",
        "text": (
            "Epistemology, the branch of philosophy concerned with knowledge, examines "
            "questions about the nature, sources, and limits of human understanding. "
            "Rationalists such as Descartes argue that certain knowledge is innate and "
            "accessible through pure reason, while empiricists like Hume maintain that all "
            "knowledge derives from sensory experience. Kant's critical philosophy attempts "
            "to synthesize these positions by proposing that knowledge requires both sensory "
            "input and the a priori structures of the mind. Contemporary epistemology "
            "addresses issues such as the Gettier problem, reliabilism, and the role of "
            "testimony in knowledge acquisition."
        ),
    },
    {
        "source_id": "test_doc_philosophy",
        "header_path": "Section B > Ethics",
        "page_number": 2,
        "page_label": "2",
        "text": (
            "Normative ethics concerns the systematization of moral principles. Consequentialist "
            "theories evaluate actions based on their outcomes, with utilitarianism being the "
            "most prominent example. Deontological ethics, associated with Kant, holds that "
            "certain actions are inherently right or wrong regardless of consequences. Virtue "
            "ethics, rooted in Aristotle, focuses on character traits rather than rules or "
            "outcomes. Contemporary debates in applied ethics address topics such as medical "
            "ethics, environmental ethics, and the ethics of artificial intelligence."
        ),
    },
    {
        "source_id": "test_doc_empty_edge",
        "header_path": "Document",
        "page_number": None,
        "page_label": None,
        "text": "Short document with minimal content for edge case testing."
    },
]

FIXED_QUERIES = [
    "What is Chomsky's theory of language?",
    "Explain the poverty of the stimulus argument",
    "Compare rationalism and empiricism in epistemology",
    "What are the main approaches to normative ethics?",
    "Summarize the methodology of generative linguistics",
    "",  # empty query edge case
]


def generate_parent_child_corpus(
    corpus: Optional[list[dict[str, Any]]] = None,
) -> tuple[list[Any], list[Any]]:
    """Generate ParentChunk / ChildChunk pairs from test corpus."""
    from src.models import ChildChunk, Metadata, ParentChunk

    corpus = corpus or FIXED_CORPUS
    parents = []
    children = []

    for item in corpus:
        parent_id = str(uuid.uuid4())
        meta = Metadata(
            source_id=item["source_id"],
            page_number=item.get("page_number"),
            page_label=item.get("page_label"),
            display_page=item.get("page_label"),
            header_path=item["header_path"],
            parent_id=None,
        )
        parent = ParentChunk(id=parent_id, text=item["text"], metadata=meta)
        parents.append(parent)

        # Create 2 child chunks per parent (simulate split)
        words = item["text"].split()
        mid = len(words) // 2
        for j, chunk_words in enumerate([words[:mid], words[mid:]]):
            chunk_text = " ".join(chunk_words)
            if len(chunk_text.strip()) < 10:
                continue
            child_meta = Metadata(
                source_id=item["source_id"],
                page_number=item.get("page_number"),
                page_label=item.get("page_label"),
                display_page=item.get("page_label"),
                header_path=item["header_path"],
                parent_id=parent_id,
            )
            child = ChildChunk(text=chunk_text, metadata=child_meta)
            children.append(child)

    return parents, children


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_embedder() -> MockEmbeddingModel:
    return MockEmbeddingModel(dim=384)


@pytest.fixture
def mock_reranker() -> MockReranker:
    return MockReranker()


@pytest.fixture
def mock_tokenizer() -> MockTokenizer:
    return MockTokenizer()


@pytest.fixture
def tmp_storage(tmp_path: Path, mock_embedder: MockEmbeddingModel):
    """Create a StorageEngine with test corpus loaded."""
    from src.storage import StorageConfig, StorageEngine

    config = StorageConfig(
        lance_dir=tmp_path / "lance",
        lance_table="test_chunks",
    )
    engine = StorageEngine(config)

    parents, children = generate_parent_child_corpus()
    engine.add_parents(parents)

    texts = [c.text for c in children]
    embeddings = mock_embedder.encode(texts, normalize_embeddings=True)
    engine.add_children(children, embeddings=embeddings)

    yield engine
    engine.close()


@pytest.fixture
def test_corpus():
    """Return the fixed test corpus."""
    return FIXED_CORPUS


@pytest.fixture
def test_queries():
    """Return the fixed test queries."""
    return FIXED_QUERIES


@pytest.fixture
def parent_child_corpus():
    """Generate parent and child chunks from fixed corpus."""
    return generate_parent_child_corpus()


@pytest.fixture
def test_logger():
    """Provide a structured test logger."""
    return get_test_logger("test_run")
