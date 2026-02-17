"""Jina Reranker v3 – MLX-native implementation.

Loads the Qwen3-0.6B backbone via ``mlx_lm`` and applies the Jina-trained MLP
projector head for listwise reranking on Apple Silicon (Metal).

Architecture (from safetensors inspection):
    backbone  : Qwen3-0.6B  (28 layers, hidden_size=1024)
    projector : Linear(1024→512) → ReLU → Linear(512→512)
    scoring   : cosine similarity between projected query & doc embeddings

Special tokens:
    <|embed_token|>   (id 151670) – marks end of each document passage
    <|rerank_token|>  (id 151671) – marks end of the query
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any, Optional

import mlx.core as mx
import mlx.nn as nn
import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_JINA_MLX_REPO = "jinaai/jina-reranker-v3-mlx"
_PROJECTOR_FILENAME = "projector.safetensors"

# Fallback IDs – used only when the tokenizer lacks the special tokens.
_FALLBACK_DOC_EMBED_TOKEN_ID = 151670   # <|embed_token|>
_FALLBACK_QUERY_EMBED_TOKEN_ID = 151671  # <|rerank_token|>

_SPECIAL_TOKENS: dict[str, str] = {
    "query_embed_token": "<|rerank_token|>",
    "doc_embed_token": "<|embed_token|>",
}

_MAX_DOC_TOKENS = 384
_MAX_QUERY_TOKENS = 512

# Context-window safety: leave headroom below the 131 K backbone limit.
_MAX_RERANK_PROMPT_TOKENS = 120_000

# Rough token-overhead estimates for fast budget check (avoids re-tokenising
# the entire prompt when the total is clearly within budget).
_PROMPT_OVERHEAD_TOKENS = 300   # system text + query section + markers
_PER_DOC_OVERHEAD_TOKENS = 25   # <passage id="X"> … </passage> per doc

# Numerical stability for cosine similarity.
_EPS = 1e-8


# ---------------------------------------------------------------------------
# Prompt formatting  (mirrors the official Jina implementation)
# ---------------------------------------------------------------------------

def _sanitize(text: str) -> str:
    """Strip special marker tokens from user-supplied text."""
    for tok in _SPECIAL_TOKENS.values():
        text = text.replace(tok, "")
    return text


def _build_prompt(query: str, docs: list[str]) -> str:
    """Build the listwise prompt consumed by jina-reranker-v3.

    Format:
        <|im_start|>system … <|im_end|>
        <|im_start|>user
        … passages with <|embed_token|> …
        … query  with <|rerank_token|> …
        <|im_end|>
        <|im_start|>assistant
        <think>…</think>
    """
    query = _sanitize(query)
    docs = [_sanitize(d) for d in docs]

    doc_tok = _SPECIAL_TOKENS["doc_embed_token"]
    qry_tok = _SPECIAL_TOKENS["query_embed_token"]

    prefix = (
        "<|im_start|>system\n"
        "You are a search relevance expert who can determine a ranking of the "
        "passages based on how relevant they are to the query. "
        "If the query is a question, how relevant a passage is depends on how "
        "well it answers the question. "
        "If not, try to analyze the intent of the query and assess how well "
        "each passage satisfies the intent. "
        "If an instruction is provided, you should follow the instruction when "
        "determining the ranking."
        "<|im_end|>\n<|im_start|>user\n"
    )
    suffix = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"

    body = (
        f"I will provide you with {len(docs)} passages, each indicated by a "
        f"numerical identifier. "
        f"Rank the passages based on their relevance to query: {query}\n"
    )
    for idx, doc in enumerate(docs):
        body += f'<passage id="{idx}">\n{doc}{doc_tok}\n</passage>\n'
    body += f"<query>\n{query}{qry_tok}\n</query>"

    return prefix + body + suffix


# ---------------------------------------------------------------------------
# MLP Projector
# ---------------------------------------------------------------------------

class _MLPProjector(nn.Module):
    """1024 → 512 → 512 projection head with ReLU activation."""

    def __init__(self) -> None:
        super().__init__()
        self.linear1 = nn.Linear(1024, 512, bias=False)
        self.linear2 = nn.Linear(512, 512, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        x = self.linear1(x)
        x = nn.relu(x)
        x = self.linear2(x)
        return x


def _load_projector(projector_path: str) -> _MLPProjector:
    """Load projector weights from a safetensors file."""
    from safetensors import safe_open

    proj = _MLPProjector()
    with safe_open(projector_path, framework="numpy") as f:
        proj.linear1.weight = mx.array(f.get_tensor("linear1.weight"))
        proj.linear2.weight = mx.array(f.get_tensor("linear2.weight"))

    logger.info(
        "Projector loaded  linear1=%s  linear2=%s",
        proj.linear1.weight.shape,
        proj.linear2.weight.shape,
    )
    return proj


# ---------------------------------------------------------------------------
# JinaRerankerMLX  – drop-in replacement for FlagReranker
# ---------------------------------------------------------------------------

class JinaRerankerMLX:
    """MLX-native Jina Reranker v3 with ``compute_score`` interface.

    This class is a drop-in replacement for ``FlagReranker`` in the existing
    retrieval pipeline: :pymethod:`RetrievalEngine._rerank` calls
    ``reranker.compute_score(pairs)`` and expects a list of floats.

    Internally the model processes all documents *listwise* in a single
    forward pass (within 131K context), which is how Jina v3 is designed.
    """

    def __init__(self, model_id: str = _JINA_MLX_REPO) -> None:
        t0 = time.perf_counter()
        logger.info("Loading Jina Reranker v3 MLX backbone from %s …", model_id)

        from mlx_lm import load as mlx_load
        from huggingface_hub import hf_hub_download

        # Load backbone (Qwen3-0.6B converted to MLX)
        self._model, self._tokenizer = mlx_load(model_id)
        self._model.eval()

        # Load projector (separate safetensors)
        proj_path = hf_hub_download(model_id, _PROJECTOR_FILENAME)
        self._projector = _load_projector(proj_path)

        # Resolve special-token IDs from the tokenizer at runtime so that
        # model swaps don't silently break position detection.
        self._doc_embed_token_id = self._resolve_token_id(
            _SPECIAL_TOKENS["doc_embed_token"], _FALLBACK_DOC_EMBED_TOKEN_ID,
        )
        self._query_embed_token_id = self._resolve_token_id(
            _SPECIAL_TOKENS["query_embed_token"], _FALLBACK_QUERY_EMBED_TOKEN_ID,
        )

        elapsed = time.perf_counter() - t0
        logger.info(
            "Jina Reranker v3 MLX ready in %.1fs  (backbone + projector)", elapsed
        )

    # -- Helpers ------------------------------------------------------------

    def _resolve_token_id(self, token_text: str, fallback_id: int) -> int:
        """Resolve a special token to its ID via the tokenizer, with fallback."""
        try:
            vocab = self._tokenizer.get_vocab() if hasattr(self._tokenizer, "get_vocab") else {}
            if token_text in vocab:
                resolved = vocab[token_text]
                logger.debug("Resolved %s -> %d (vocab lookup)", token_text, resolved)
                return resolved

            ids = self._tokenizer.encode(token_text, add_special_tokens=False)
            if ids:
                resolved = ids[-1]
                logger.debug("Resolved %s -> %d (encode)", token_text, resolved)
                return resolved
        except Exception as exc:
            logger.warning(
                "Failed to resolve %s dynamically (%s); using fallback %d",
                token_text, exc, fallback_id,
            )

        logger.warning(
            "Could not resolve %s from tokenizer; using hard-coded fallback %d. "
            "This may break on a different model checkpoint.",
            token_text, fallback_id,
        )
        return fallback_id

    # -- Public interface (compatible with FlagReranker) --------------------

    def compute_score(self, pairs: list[tuple[str, str]]) -> list[float]:
        """Score ``(query, document)`` pairs.

        All pairs are assumed to share the **same query** (which is the case
        in the retrieval pipeline).  The documents are scored listwise in a
        single forward pass.

        Returns a list of float relevance scores in the **same order** as
        the input pairs.
        """
        if not pairs:
            return []

        query = pairs[0][0]
        docs = [p[1] for p in pairs]

        # Truncate documents that are excessively long.
        # Returns per-doc token counts so we can skip re-tokenisation later.
        docs, doc_token_counts = self._truncate_docs(docs)

        # Context-window safety: trim the document list so the
        # listwise prompt stays within the backbone's context limit.
        docs = self._enforce_context_budget(query, docs, doc_token_counts)

        scores = self._score_listwise(query, docs)
        return scores

    # -- Internals ----------------------------------------------------------

    def _truncate_docs(self, docs: list[str]) -> tuple[list[str], list[int]]:
        """Truncate documents to ``_MAX_DOC_TOKENS`` tokens each.

        Returns ``(truncated_docs, per_doc_token_counts)`` so that
        downstream code can estimate the total prompt length without
        re-tokenising every document.
        """
        truncated: list[str] = []
        token_counts: list[int] = []
        total_original_tokens = 0
        total_truncated_tokens = 0
        truncation_count = 0
        for doc in docs:
            tok_ids = self._tokenizer.encode(doc)
            total_original_tokens += len(tok_ids)
            if len(tok_ids) > _MAX_DOC_TOKENS:
                tok_ids = tok_ids[:_MAX_DOC_TOKENS]
                doc = self._tokenizer.decode(tok_ids)
                truncation_count += 1
            total_truncated_tokens += len(tok_ids)
            truncated.append(doc)
            token_counts.append(len(tok_ids))
        if truncation_count > 0:
            logger.info(
                "Reranker doc truncation: %d/%d docs truncated, "
                "tokens %d → %d (saved %d tokens, %.0f%% reduction)",
                truncation_count, len(docs),
                total_original_tokens, total_truncated_tokens,
                total_original_tokens - total_truncated_tokens,
                100 * (1 - total_truncated_tokens / max(total_original_tokens, 1)),
            )
        return truncated, token_counts

    def _estimate_prompt_tokens(
        self, query: str, doc_token_counts: list[int],
    ) -> int:
        """Fast estimate of total prompt tokens from pre-computed doc counts."""
        query_tokens = len(self._tokenizer.encode(query))
        return (
            _PROMPT_OVERHEAD_TOKENS
            + _PER_DOC_OVERHEAD_TOKENS * len(doc_token_counts)
            + sum(doc_token_counts)
            + query_tokens
        )

    def _enforce_context_budget(
        self, query: str, docs: list[str],
        doc_token_counts: Optional[list[int]] = None,
    ) -> list[str]:
        """Trim the document list so the listwise prompt fits the context window."""
        if not docs:
            return docs

        # ---- fast path: skip full tokenisation when clearly under budget ----
        if doc_token_counts is not None:
            estimate = self._estimate_prompt_tokens(query, doc_token_counts)
            if estimate < _MAX_RERANK_PROMPT_TOKENS * 0.85:
                logger.debug(
                    "Context budget fast-pass: estimated %d tokens (limit %d)",
                    estimate, _MAX_RERANK_PROMPT_TOKENS,
                )
                return docs

        # ---- slow path: full tokenisation (only near the limit) ----
        prompt = _build_prompt(query, docs)
        token_count = len(self._tokenizer.encode(prompt))

        if token_count <= _MAX_RERANK_PROMPT_TOKENS:
            return docs

        logger.warning(
            "Reranker prompt (%d tokens) exceeds context budget (%d). "
            "Truncating document list from %d docs.",
            token_count, _MAX_RERANK_PROMPT_TOKENS, len(docs),
        )

        # Binary search for the largest prefix of docs that fits.
        lo, hi = 1, len(docs) - 1
        while lo < hi:
            mid = (lo + hi + 1) // 2
            trial = _build_prompt(query, docs[:mid])
            if len(self._tokenizer.encode(trial)) <= _MAX_RERANK_PROMPT_TOKENS:
                lo = mid
            else:
                hi = mid - 1

        logger.warning("Truncated to %d documents to fit context window.", lo)
        return docs[:lo]

    def _score_listwise(self, query: str, docs: list[str]) -> list[float]:
        """Run a single-pass listwise forward and return per-doc scores."""
        doc_token_counts = [len(self._tokenizer.encode(doc)) for doc in docs]
        if doc_token_counts:
            logger.info(
                "Reranker doc tokens (per-doc): %s",
                doc_token_counts,
            )
            logger.info(
                "Reranker doc token stats: min=%d max=%d mean=%.1f n=%d",
                min(doc_token_counts),
                max(doc_token_counts),
                sum(doc_token_counts) / len(doc_token_counts),
                len(doc_token_counts),
            )

        prompt = _build_prompt(query, docs)
        input_ids = self._tokenizer.encode(prompt)

        token_count = len(input_ids)
        logger.info(
            "Reranker forward pass: %d tokens total, %d documents "
            "(avg %.0f tok/doc)",
            token_count, len(docs),
            token_count / max(len(docs), 1),
        )

        # Forward through backbone – get last-layer hidden states
        # mlx_lm models expose ``model.model(...)`` for the transformer body.
        input_mx = mx.array(input_ids)[None, :]  # [1, seq_len]
        hidden = self._model.model(input_mx)       # [1, seq_len, hidden_size]
        hidden = hidden[0]                          # [seq_len, hidden_size]

        # Locate special-token positions
        ids_np = np.array(input_ids)
        doc_positions = np.where(ids_np == self._doc_embed_token_id)[0]
        query_positions = np.where(ids_np == self._query_embed_token_id)[0]

        if len(doc_positions) == 0:
            raise RuntimeError(
                "No <|embed_token|> found in reranker input – "
                "tokenizer may be missing the special token."
            )
        if len(query_positions) == 0:
            raise RuntimeError(
                "No <|rerank_token|> found in reranker input – "
                "tokenizer may be missing the special token."
            )

        # Extract & project embeddings
        doc_hidden = mx.stack(
            [hidden[int(p)] for p in doc_positions]
        )  # [num_docs, 1024]
        query_hidden = hidden[int(query_positions[0])][None, :]  # [1, 1024]

        doc_embeds = self._projector(doc_hidden)    # [num_docs, 512]
        query_embeds = self._projector(query_hidden) # [1, 512]

        # Cosine similarity
        query_expanded = mx.broadcast_to(query_embeds, doc_embeds.shape)
        cos_sim = mx.sum(doc_embeds * query_expanded, axis=-1) / (
            mx.sqrt(mx.sum(doc_embeds * doc_embeds, axis=-1))
            * mx.sqrt(mx.sum(query_expanded * query_expanded, axis=-1))
            + _EPS
        )  # [num_docs]

        # Force evaluation & convert to Python floats
        mx.eval(cos_sim)
        scores = cos_sim.tolist()

        # Pad / align if number of docs doesn't match positions (safety)
        while len(scores) < len(docs):
            scores.append(float("-inf"))

        logger.debug(
            "Reranker scores: min=%.4f  max=%.4f  n=%d",
            min(scores[: len(docs)]),
            max(scores[: len(docs)]),
            len(docs),
        )
        return scores[: len(docs)]

