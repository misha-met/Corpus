from __future__ import annotations

from dataclasses import dataclass
import logging
import re
from typing import Any, Optional

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class GenerationConfig:
    max_tokens: Optional[int] = None
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    repetition_penalty: Optional[float] = None
    stop_tokens: Optional[list[str]] = None


# Default stop tokens to prevent runaway generation and meta-commentary
DEFAULT_STOP_TOKENS = [
    # Model-specific end tokens
    "<|endoftext|>",
    "<|im_end|>",
    "<|eot_id|>",
    # Conversation markers
    "Human:",
    "Assistant:",
    # Prompt continuation markers
    "\n\nQuestion:",
    "\n\nContext:",
    # Meta-commentary phrases (model explaining its own response)
    "Answer ends here",
    "This answer acknowledges",
    "This answer was generated",
    "This response reflects",
    "This response was",
]


class MlxGenerator:
    def __init__(self, model_path: str) -> None:
        self._model_id = model_path
        try:
            from mlx_lm import load
        except Exception as exc:  # pragma: no cover - dependency runtime
            raise RuntimeError("mlx-lm is not available. Install mlx-lm to continue.") from exc

        try:
            self._model, self._tokenizer = load(model_path)
        except Exception as exc:  # pragma: no cover - dependency runtime
            raise RuntimeError(f"Failed to load mlx-lm model at {model_path}.") from exc

    @property
    def tokenizer(self) -> Any:
        """Expose the MLX tokenizer for external token counting.
        
        Returns:
            The tokenizer instance loaded with the model.
        """
        return self._tokenizer

    @property
    def model_id(self) -> str:
        """Get the model identifier.
        
        Returns:
            The model path/ID used to load this generator.
        """
        return self._model_id

    @staticmethod
    def _infer_model_size_b(model_id: str) -> Optional[float]:
        match = re.search(r"(\d+(?:\.\d+)?)\s*[bB]", model_id)
        if not match:
            return None
        try:
            return float(match.group(1))
        except ValueError:
            return None

    @staticmethod
    def _build_repetition_penalty_processor(penalty: float):
        if penalty <= 1.0:
            return None

        def _processor(tokens, logits):
            try:
                import mlx.core as mx
            except Exception:
                return logits

            token_ids = tokens.tolist()
            if token_ids and isinstance(token_ids[0], list):
                token_ids = token_ids[0]
            if not token_ids:
                return logits

            logits_np = mx.array(logits)
            token_ids = sorted(set(int(t) for t in token_ids))
            cols = mx.take(logits_np, mx.array(token_ids), axis=-1)
            adjusted = mx.where(
                cols > 0,
                cols / penalty,
                cols * penalty,
            )
            try:
                import numpy as np

                logits_np_np = logits_np.astype("float32").tolist()
                adjusted_np = adjusted.astype("float32").tolist()
                for col_idx, token_id in enumerate(token_ids):
                    logits_np_np[0][token_id] = adjusted_np[0][col_idx]
                return mx.array(logits_np_np)
            except Exception:
                return logits_np

        return _processor

    def generate(self, prompt: str, *, config: Optional[GenerationConfig] = None) -> str:
        if not prompt.strip():
            raise ValueError("prompt must be a non-empty string.")

        cfg = config or GenerationConfig()
        model_size = self._infer_model_size_b(self._model_id)
        max_tokens = cfg.max_tokens
        repetition_penalty = cfg.repetition_penalty
        temperature = cfg.temperature
        top_p = cfg.top_p

        if model_size is not None:
            if model_size < 30:
                max_tokens = max_tokens or 140
                repetition_penalty = repetition_penalty or 1.25
                temperature = temperature or 0.05
                top_p = top_p or 0.7
            elif model_size >= 70:
                repetition_penalty = repetition_penalty or 1.05
                temperature = temperature or 0.2
                top_p = top_p or 0.9
            else:
                max_tokens = max_tokens or 400
                repetition_penalty = repetition_penalty or 1.15
                temperature = temperature or 0.15
                top_p = top_p or 0.9
        else:
            temperature = temperature or 0.2
            top_p = top_p or 0.9
        try:
            from mlx_lm import generate
            from mlx_lm.generate import make_sampler
        except Exception as exc:  # pragma: no cover - dependency runtime
            raise RuntimeError("mlx-lm generate is not available.") from exc

        # Collect stop tokens
        stop_tokens = cfg.stop_tokens if cfg.stop_tokens is not None else DEFAULT_STOP_TOKENS

        try:
            sampler = make_sampler(temp=temperature, top_p=top_p)
            logits_processors = []
            if repetition_penalty is not None:
                processor = self._build_repetition_penalty_processor(repetition_penalty)
                if processor is not None:
                    logits_processors.append(processor)

            # Use explicit max_tokens if provided, otherwise default cap at 300
            final_max_tokens = max_tokens if max_tokens is not None else 300
            output = generate(
                self._model,
                self._tokenizer,
                prompt,
                max_tokens=final_max_tokens,
                sampler=sampler,
                logits_processors=logits_processors or None,
            )
            
            # Apply stop token truncation (mlx-lm may not support all stop tokens natively)
            output = self._apply_stop_tokens(output, stop_tokens)
            return output
        except Exception as exc:  # pragma: no cover - dependency runtime
            raise RuntimeError("mlx-lm generation failed.") from exc

    def generate_chat(
        self,
        messages: list[dict[str, str]],
        *,
        config: Optional[GenerationConfig] = None,
    ) -> str:
        if not messages:
            raise ValueError("messages must be a non-empty list of role/content dicts.")

        prompt = self._apply_chat_template(messages)
        output = self.generate(prompt, config=config)
        return self._strip_thinking_blocks(output)

    def _apply_chat_template(self, messages: list[dict[str, str]]) -> str:
        if hasattr(self._tokenizer, "apply_chat_template"):
            try:
                return self._tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=False,
                )
            except TypeError:
                return self._tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
            except Exception:
                pass

        parts = []
        for message in messages:
            role = message.get("role", "user").strip().lower()
            content = message.get("content", "").strip()
            if not content:
                continue
            if role == "system":
                parts.append(f"System: {content}")
            elif role == "assistant":
                parts.append(f"Assistant: {content}")
            else:
                parts.append(f"User: {content}")
        parts.append("Assistant:")
        return "\n\n".join(parts)

    @staticmethod
    def _strip_thinking_blocks(text: str) -> str:
        if not text:
            return text
        return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()

    @staticmethod
    def _apply_stop_tokens(text: str, stop_tokens: list[str]) -> str:
        """Truncate output at the first occurrence of any stop token."""
        if not stop_tokens:
            return text
        
        earliest_pos = len(text)
        for token in stop_tokens:
            pos = text.find(token)
            if pos != -1 and pos < earliest_pos:
                earliest_pos = pos
        
        return text[:earliest_pos].rstrip()


# =============================================================================
# Token Counting and Budget Packing
# =============================================================================

def count_tokens(text: str, tokenizer: Any) -> int:
    """Count tokens in text using the MLX tokenizer.
    
    Args:
        text: Text to tokenize
        tokenizer: MLX tokenizer instance (from MlxGenerator.tokenizer)
        
    Returns:
        Number of tokens in text
    """
    if not text:
        return 0
    
    try:
        # MLX tokenizers typically have an encode method
        if hasattr(tokenizer, 'encode'):
            tokens = tokenizer.encode(text)
            return len(tokens)
        # Fallback: try tokenize method
        if hasattr(tokenizer, 'tokenize'):
            tokens = tokenizer.tokenize(text)
            return len(tokens)
        # Last resort: approximate by characters (rough estimate)
        logger.warning("Tokenizer lacks encode/tokenize method, using character approximation")
        return len(text) // 4
    except Exception as exc:
        logger.warning(f"Token counting failed: {exc}, using character approximation")
        return len(text) // 4


@dataclass
class BudgetPackResult:
    """Result of greedy budget packing.
    
    Attributes:
        packed_docs: List of documents that fit within budget
        packed_indices: Original indices of the packed documents (preserves metadata alignment)
        used_tokens: Total tokens used
        skipped_count: Number of documents skipped
        truncated_count: Number of documents truncated
        consecutive_fails: Number of consecutive documents that didn't fit
    """
    packed_docs: list[str]
    packed_indices: list[int]
    used_tokens: int
    skipped_count: int
    truncated_count: int
    consecutive_fails: int


def enforce_token_budget(
    docs: list[str],
    max_tokens: int,
    tokenizer: Any,
    *,
    consecutive_fail_threshold: int = 3,
    allow_truncation: bool = True,
    min_doc_tokens: int = 50,
    log: Optional[logging.Logger] = None,
) -> BudgetPackResult:
    """Greedy token budget packing with consecutive fail guard.
    
    Packs documents into the token budget using a greedy algorithm.
    Stops early if consecutive documents fail to fit (guard against
    spending time on documents that are all too large).
    
    Args:
        docs: List of document texts to pack (in priority order)
        max_tokens: Maximum token budget
        tokenizer: MLX tokenizer for accurate counting
        consecutive_fail_threshold: Stop after this many consecutive skips
        allow_truncation: Whether to truncate large docs as fallback
        min_doc_tokens: Minimum tokens for a truncated doc to be useful
        log: Logger for decision logging
        
    Returns:
        BudgetPackResult with packed documents and statistics
    """
    if log is None:
        log = logger
    
    packed: list[str] = []
    packed_indices: list[int] = []  # Track original indices for metadata alignment
    used_tokens = 0
    skipped = 0
    truncated = 0
    consecutive_fails = 0
    
    for i, doc in enumerate(docs):
        if not doc or not doc.strip():
            continue
        
        doc_tokens = count_tokens(doc, tokenizer)
        remaining = max_tokens - used_tokens
        
        # Document fits entirely
        if doc_tokens <= remaining:
            packed.append(doc)
            packed_indices.append(i)
            used_tokens += doc_tokens
            consecutive_fails = 0
            log.debug(f"Doc {i}: packed ({doc_tokens} tokens, total: {used_tokens})")
            continue
        
        # Document doesn't fit - try truncation
        if allow_truncation and remaining >= min_doc_tokens:
            # Estimate truncation point (rough: target 80% of remaining)
            target_tokens = int(remaining * 0.8)
            
            # Binary search for truncation point
            truncated_doc = _truncate_to_tokens(doc, target_tokens, tokenizer)
            if truncated_doc:
                truncated_tokens = count_tokens(truncated_doc, tokenizer)
                if truncated_tokens <= remaining:
                    packed.append(truncated_doc)
                    packed_indices.append(i)
                    used_tokens += truncated_tokens
                    truncated += 1
                    consecutive_fails = 0
                    log.debug(
                        f"Doc {i}: truncated ({doc_tokens} -> {truncated_tokens} tokens, "
                        f"total: {used_tokens})"
                    )
                    continue
        
        # Skip document
        skipped += 1
        consecutive_fails += 1
        log.debug(f"Doc {i}: skipped ({doc_tokens} tokens, remaining: {remaining})")
        
        # Guard: stop if too many consecutive fails
        if consecutive_fails >= consecutive_fail_threshold:
            log.info(
                f"Budget packing stopped: {consecutive_fails} consecutive docs "
                f"exceeded remaining budget ({remaining} tokens)"
            )
            break
    
    if max_tokens:
        percent_used_str = f"{100 * used_tokens / max_tokens:.1f}%"
    else:
        percent_used_str = "n/a"

    log.info(
        f"Budget packing complete: {len(packed)} docs, {used_tokens}/{max_tokens} tokens "
        f"({percent_used_str}), {skipped} skipped, {truncated} truncated"
    )
    
    return BudgetPackResult(
        packed_docs=packed,
        packed_indices=packed_indices,
        used_tokens=used_tokens,
        skipped_count=skipped,
        truncated_count=truncated,
        consecutive_fails=consecutive_fails,
    )


def _truncate_to_tokens(
    text: str,
    target_tokens: int,
    tokenizer: Any,
    *,
    tolerance: float = 0.1,
) -> Optional[str]:
    """Truncate text to approximately target token count.
    
    Uses binary search to find a good truncation point that
    preserves complete sentences where possible.
    
    Args:
        text: Text to truncate
        target_tokens: Target token count
        tokenizer: MLX tokenizer
        tolerance: Acceptable deviation from target (0.1 = 10%)
        
    Returns:
        Truncated text, or None if truncation not feasible
    """
    if target_tokens <= 0:
        return None
    
    # Start with character-based estimate
    char_estimate = target_tokens * 4  # ~4 chars per token
    
    if char_estimate >= len(text):
        return text
    
    # Try to find a sentence boundary near the estimate
    search_start = max(0, char_estimate - 200)
    search_end = min(len(text), char_estimate + 200)
    search_region = text[search_start:search_end]
    
    # Look for sentence endings and choose the closest boundary *before* char_estimate
    best_pos: Optional[int] = None
    for ending in ['. ', '.\n', '! ', '!\n', '? ', '?\n']:
        # Only search up to the estimated position to avoid picking later boundaries
        relative_limit = max(0, min(char_estimate - search_start, len(search_region)))
        if relative_limit == 0:
            continue
        pos = search_region.rfind(ending, 0, relative_limit)
        if pos != -1:
            candidate = search_start + pos + len(ending)
            if candidate <= char_estimate and (best_pos is None or candidate > best_pos):
                best_pos = candidate
    
    if best_pos is None:
        best_pos = char_estimate
    
    # Truncate and verify
    truncated = text[:best_pos].rstrip()
    actual_tokens = count_tokens(truncated, tokenizer)
    
    # Check if within tolerance
    if actual_tokens <= target_tokens * (1 + tolerance):
        return truncated
    
    # If still too long, do hard truncation
    if actual_tokens > target_tokens:
        # Reduce proportionally
        ratio = target_tokens / actual_tokens
        new_len = int(len(truncated) * ratio * 0.9)  # 10% safety margin
        return text[:new_len].rstrip()
    
    return truncated
