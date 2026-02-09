from __future__ import annotations

import logging
import time
import re
from dataclasses import dataclass
from typing import Any, Optional

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class GenerationConfig:
    max_tokens: Optional[int] = None
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    repetition_penalty: Optional[float] = None
    stop_tokens: Optional[list[str]] = None


DEFAULT_STOP_TOKENS = [
    "<|endoftext|>", "<|im_end|>", "<|eot_id|>",
    "Human:", "Assistant:",
    "\n\nQuestion:", "\n\nContext:",
    "Answer ends here", "This answer acknowledges", "This answer was generated",
    "This response reflects", "This response was",
]


class MlxGenerator:
    """MLX-LM based text generator with configurable sampling."""

    def __init__(self, model_path: str) -> None:
        self._model_id = model_path
        try:
            from mlx_lm import load
            self._model, self._tokenizer = load(model_path)
        except Exception as exc:
            raise RuntimeError(f"Failed to load mlx-lm model at {model_path}") from exc

    @property
    def tokenizer(self) -> Any:
        return self._tokenizer

    @property
    def model_id(self) -> str:
        return self._model_id

    @staticmethod
    def _infer_model_size_b(model_id: str) -> Optional[float]:
        match = re.search(r"(\d+(?:\.\d+)?)\s*[bB]", model_id)
        return float(match.group(1)) if match else None

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

            # IMPORTANT: Always pass explicit max_tokens to override mlx-lm's hidden 256 default
            # Use 1200 tokens (~900 words) for long-form academic answers when not specified
            final_max_tokens = max_tokens if max_tokens is not None else 1200
            prompt_tokens = count_tokens(prompt, self._tokenizer)
            start_time = time.perf_counter()
            output = generate(
                self._model,
                self._tokenizer,
                prompt,
                max_tokens=final_max_tokens,
                sampler=sampler,
                logits_processors=logits_processors or None,
            )
            elapsed_s = time.perf_counter() - start_time
            
            # Apply stop token truncation (mlx-lm may not support all stop tokens natively)
            output = self._apply_stop_tokens(output, stop_tokens)
            output_tokens = count_tokens(output, self._tokenizer)
            total_tokens = prompt_tokens + output_tokens
            elapsed_safe = max(elapsed_s, 1e-6)
            tokens_per_sec = output_tokens / elapsed_safe
            logger.info(
                "LLM generation | model=%s prompt_tokens=%d output_tokens=%d total_tokens=%d "
                "time_s=%.2f tokens_per_s=%.2f",
                self._model_id,
                prompt_tokens,
                output_tokens,
                total_tokens,
                elapsed_s,
                tokens_per_sec,
            )
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
        """Truncate output at first stop token occurrence."""
        if not stop_tokens:
            return text
        earliest_pos = len(text)
        for token in stop_tokens:
            pos = text.find(token)
            if pos != -1 and pos < earliest_pos:
                earliest_pos = pos
        return text[:earliest_pos].rstrip()


def count_tokens(text: str, tokenizer: Any) -> int:
    """Count tokens using MLX tokenizer, with character-based fallback."""
    if not text:
        return 0
    try:
        if hasattr(tokenizer, 'encode'):
            return len(tokenizer.encode(text))
        if hasattr(tokenizer, 'tokenize'):
            return len(tokenizer.tokenize(text))
    except Exception:
        pass
    return len(text) // 4  # ~4 chars per token fallback


@dataclass
class BudgetPackResult:
    """Result of greedy budget packing."""
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
    """Greedy token budget packing with early termination on consecutive failures."""
    log = log or logger
    packed: list[str] = []
    packed_indices: list[int] = []
    used_tokens = 0
    skipped = 0
    truncated = 0
    consecutive_fails = 0

    for i, doc in enumerate(docs):
        if not doc or not doc.strip():
            continue

        doc_tokens = count_tokens(doc, tokenizer)
        remaining = max_tokens - used_tokens

        if doc_tokens <= remaining:
            packed.append(doc)
            packed_indices.append(i)
            used_tokens += doc_tokens
            consecutive_fails = 0
            continue

        if allow_truncation and remaining >= min_doc_tokens:
            truncated_doc = _truncate_to_tokens(doc, int(remaining * 0.8), tokenizer)
            if truncated_doc:
                truncated_tokens = count_tokens(truncated_doc, tokenizer)
                if truncated_tokens <= remaining:
                    packed.append(truncated_doc)
                    packed_indices.append(i)
                    used_tokens += truncated_tokens
                    truncated += 1
                    consecutive_fails = 0
                    continue

        skipped += 1
        consecutive_fails += 1
        if consecutive_fails >= consecutive_fail_threshold:
            log.info(f"Budget packing stopped: {consecutive_fails} consecutive docs exceeded remaining budget")
            break

    pct = f"{100 * used_tokens / max_tokens:.1f}%" if max_tokens else "n/a"
    log.info(f"Budget packing: {len(packed)} docs, {used_tokens}/{max_tokens} tokens ({pct}), {skipped} skipped, {truncated} truncated")

    return BudgetPackResult(
        packed_docs=packed, packed_indices=packed_indices, used_tokens=used_tokens,
        skipped_count=skipped, truncated_count=truncated, consecutive_fails=consecutive_fails,
    )


def _truncate_to_tokens(text: str, target_tokens: int, tokenizer: Any, *, tolerance: float = 0.1) -> Optional[str]:
    """Truncate text to approximately target token count, preferring sentence boundaries."""
    if target_tokens <= 0:
        return None

    char_estimate = target_tokens * 4
    if char_estimate >= len(text):
        return text

    # Find sentence boundary near estimate
    search_start = max(0, char_estimate - 200)
    search_region = text[search_start:min(len(text), char_estimate + 200)]
    best_pos: Optional[int] = None

    for ending in ['. ', '.\n', '! ', '!\n', '? ', '?\n']:
        relative_limit = max(0, min(char_estimate - search_start, len(search_region)))
        if relative_limit == 0:
            continue
        pos = search_region.rfind(ending, 0, relative_limit)
        if pos != -1:
            candidate = search_start + pos + len(ending)
            if candidate <= char_estimate and (best_pos is None or candidate > best_pos):
                best_pos = candidate

    truncated = text[:best_pos or char_estimate].rstrip()
    actual_tokens = count_tokens(truncated, tokenizer)

    if actual_tokens <= target_tokens * (1 + tolerance):
        return truncated
    if actual_tokens > target_tokens:
        ratio = target_tokens / actual_tokens
        return text[:int(len(truncated) * ratio * 0.9)].rstrip()
    return truncated
