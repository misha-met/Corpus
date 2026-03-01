from __future__ import annotations

import logging
import time
import re
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class GenerationConfig:
    max_tokens: Optional[int] = None
    max_internal_tokens: Optional[int] = None  # total cap (thinking + answer)
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    top_k: Optional[int] = None
    min_p: Optional[float] = None
    repetition_penalty: Optional[float] = None
    presence_penalty: Optional[float] = None
    stop_tokens: Optional[list[str]] = None
    context_window: Optional[int] = None


DEFAULT_STOP_TOKENS = [
    "<|endoftext|>", "<|im_end|>", "<|eot_id|>",
    "Human:", "Assistant:",
    "\n\nQuestion:", "\n\nContext:",
    "Answer ends here", "This answer acknowledges", "This answer was generated",
    "This response reflects", "This response was",
]

STREAM_BUFFER_LIMIT_CHARS = 500
STREAM_TAIL_GUARD_CHARS = 128
SENTENCE_BOUNDARY_REGEX = re.compile(r"[.!?][\s\n]+")


class MlxGenerator:
    """MLX-LM based text generator with KVCache for context warming."""

    def __init__(self, model_path: str) -> None:
        self._model_id = model_path
        self._make_prompt_cache = None
        self._thinking_open_tag = "<think>"
        self._thinking_close_tag = "</think>"
        self._thinking_block_pattern = re.compile(r"<think>.*?</think>", flags=re.DOTALL)
        self._stop_token_pattern_cache: dict[tuple[str, ...], re.Pattern[str]] = {}
        self._default_stop_pattern = self._compile_stop_tokens_pattern(DEFAULT_STOP_TOKENS)
        try:
            from mlx_lm import load
            try:
                self._model, self._tokenizer = load(model_path)
            except Exception as load_exc:
                message = str(load_exc)
                if "Tokenizer class" in message or "TokenizersBackend" in message:
                    logger.info(
                        "Retrying tokenizer load with trust_remote_code=True for %s",
                        model_path,
                    )
                    try:
                        self._model, self._tokenizer = load(
                            model_path,
                            tokenizer_config={"trust_remote_code": True},
                        )
                    except Exception:
                        patched_snapshot = self._patch_tokenizer_backend_config(model_path)
                        if patched_snapshot is None:
                            raise
                        logger.info(
                            "Retrying tokenizer load with patched config at %s",
                            patched_snapshot,
                        )
                        self._model, self._tokenizer = load(
                            str(patched_snapshot),
                            tokenizer_config={"fix_mistral_regex": True},
                        )
                else:
                    raise
            # Attempt to verify KVCache support for this model
            try:
                from mlx_lm.utils import make_prompt_cache
                self._make_prompt_cache = make_prompt_cache
                # Verify cache creation works before committing to the path
                _test_cache = make_prompt_cache(self._model)
                del _test_cache
                logger.info("KVCache support verified for model %s", model_path)
            except (ImportError, AttributeError, Exception) as cache_exc:
                logger.debug("KVCache not available (mlx-lm version may not support it): %s", cache_exc)
                self._make_prompt_cache = None
        except Exception as exc:
            raise RuntimeError(f"Failed to load mlx-lm model at {model_path}") from exc

    @staticmethod
    def _patch_tokenizer_backend_config(model_path: str) -> Optional[Path]:
        """Patch cached tokenizer_config for TokenizersBackend-only models.

        Some MLX-exported Ministral repos ship ``tokenizer_class=TokenizersBackend``
        which current ``transformers`` cannot instantiate directly in this runtime.
        We normalize to ``PreTrainedTokenizerFast`` and drop incompatible
        ``extra_special_tokens`` list payload before reloading.
        """
        try:
            from huggingface_hub import snapshot_download

            snapshot_dir = Path(
                snapshot_download(
                    repo_id=model_path,
                    local_files_only=True,
                )
            )
        except Exception:
            return None

        cfg_path = snapshot_dir / "tokenizer_config.json"
        if not cfg_path.exists():
            return None

        try:
            with cfg_path.open("r", encoding="utf-8") as handle:
                cfg = json.load(handle)
        except Exception:
            return None

        if cfg.get("tokenizer_class") != "TokenizersBackend":
            return snapshot_dir

        cfg["tokenizer_class"] = "PreTrainedTokenizerFast"
        if isinstance(cfg.get("extra_special_tokens"), list):
            cfg.pop("extra_special_tokens", None)

        try:
            with cfg_path.open("w", encoding="utf-8") as handle:
                json.dump(cfg, handle)
        except Exception:
            return None

        return snapshot_dir

    @staticmethod
    def _compile_stop_tokens_pattern(stop_tokens: list[str]) -> Optional[re.Pattern[str]]:
        if not stop_tokens:
            return None
        escaped = [re.escape(token) for token in stop_tokens if token]
        if not escaped:
            return None
        return re.compile("|".join(escaped))

    def _get_stop_tokens_pattern(self, stop_tokens: list[str]) -> Optional[re.Pattern[str]]:
        stop_tuple = tuple(stop_tokens)
        if stop_tuple == tuple(DEFAULT_STOP_TOKENS):
            return self._default_stop_pattern
        pattern = self._stop_token_pattern_cache.get(stop_tuple)
        if pattern is None:
            pattern = self._compile_stop_tokens_pattern(stop_tokens)
            if pattern is not None:
                self._stop_token_pattern_cache[stop_tuple] = pattern
        return pattern

    def _resolve_generation_inputs(
        self,
        prompt: str,
        config: Optional[GenerationConfig],
    ) -> tuple[int, float, float, int, float, Optional[float], Optional[float], list[str], int]:
        """Resolve generation parameters from config and model-size defaults.

        Returns
        -------
        tuple of (max_tokens, temperature, top_p, top_k, min_p,
                  repetition_penalty, presence_penalty, stop_tokens, prompt_tokens)
        """
        cfg = config or GenerationConfig()
        model_size = self._infer_model_size_b(self._model_id)
        max_tokens = cfg.max_tokens
        repetition_penalty = cfg.repetition_penalty
        presence_penalty = cfg.presence_penalty
        temperature = cfg.temperature
        top_p = cfg.top_p
        top_k = cfg.top_k if cfg.top_k is not None else 0
        min_p = cfg.min_p if cfg.min_p is not None else 0.0

        if model_size is not None:
            if model_size < 30:
                repetition_penalty = repetition_penalty or 1.25
                temperature = temperature or 0.05
                top_p = top_p or 0.7
            elif model_size >= 70:
                repetition_penalty = repetition_penalty or 1.15
                temperature = temperature or 0.2
                top_p = top_p or 0.9
            else:
                repetition_penalty = repetition_penalty or 1.15
                temperature = temperature or 0.15
                top_p = top_p or 0.9
        else:
            temperature = temperature or 0.2
            top_p = top_p or 0.9

        stop_tokens = cfg.stop_tokens if cfg.stop_tokens is not None else DEFAULT_STOP_TOKENS
        final_max_tokens = max_tokens if max_tokens is not None else 1200
        prompt_tokens = count_tokens(prompt, self._tokenizer)

        ctx_limit = cfg.context_window
        if ctx_limit and prompt_tokens > int(ctx_limit * 0.8):
            logger.warning(
                "High prompt token count: %d tokens (%.0f%% of %dk context window). "
                "Consider reducing context.",
                prompt_tokens,
                100 * prompt_tokens / ctx_limit,
                ctx_limit // 1000,
            )

        return final_max_tokens, temperature, top_p, top_k, min_p, repetition_penalty, presence_penalty, stop_tokens, prompt_tokens

    def _build_generation_kwargs(
        self,
        *,
        prompt: str,
        max_tokens: int,
        temperature: float,
        top_p: float,
        top_k: int = 0,
        min_p: float = 0.0,
        repetition_penalty: Optional[float],
        presence_penalty: Optional[float] = None,
    ) -> dict[str, Any]:
        from mlx_lm.generate import make_sampler

        sampler = make_sampler(temp=temperature, top_p=top_p, top_k=top_k, min_p=min_p)
        logits_processors = []
        if repetition_penalty is not None:
            processor = self._build_repetition_penalty_processor(repetition_penalty)
            if processor is not None:
                logits_processors.append(processor)
        if presence_penalty is not None and presence_penalty > 0:
            pp_processor = self._build_presence_penalty_processor(presence_penalty)
            if pp_processor is not None:
                logits_processors.append(pp_processor)

        generate_kwargs: dict[str, Any] = dict(
            model=self._model,
            tokenizer=self._tokenizer,
            prompt=prompt,
            max_tokens=max_tokens,
            sampler=sampler,
            logits_processors=logits_processors or None,
        )
        if self._make_prompt_cache is not None:
            try:
                cache = self._make_prompt_cache(self._model)
                generate_kwargs["prompt_cache"] = cache
            except (ImportError, AttributeError, TypeError):
                pass
        return generate_kwargs

    def _generate_full_text(
        self,
        prompt: str,
        max_tokens: int,
        temperature: float,
        top_p: float,
        top_k: int,
        min_p: float,
        repetition_penalty: Optional[float],
        presence_penalty: Optional[float],
        stop_tokens: list[str],
        prompt_tokens: int,
    ) -> str:
        generate_kwargs = self._build_generation_kwargs(
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            min_p=min_p,
            repetition_penalty=repetition_penalty,
            presence_penalty=presence_penalty,
        )

        try:
            from mlx_lm.generate import stream_generate

            start_time = time.perf_counter()
            text_parts: list[str] = []
            response = None

            for response in stream_generate(**generate_kwargs):
                text_parts.append(response.text)

            elapsed_s = time.perf_counter() - start_time
            output = "".join(text_parts)
            output = self._apply_stop_tokens(output, stop_tokens)
            output_tokens = count_tokens(output, self._tokenizer)
            total_tokens = prompt_tokens + output_tokens

            if response is not None:
                logger.info(
                    "LLM generation | model=%s prompt_tokens=%d output_tokens=%d total_tokens=%d "
                    "time_s=%.2f prefill_tps=%.1f decode_tps=%.1f peak_memory_gb=%.2f",
                    self._model_id,
                    response.prompt_tokens,
                    output_tokens,
                    total_tokens,
                    elapsed_s,
                    response.prompt_tps,
                    response.generation_tps,
                    response.peak_memory,
                )
            else:
                logger.info(
                    "LLM generation | model=%s prompt_tokens=%d output_tokens=%d "
                    "time_s=%.2f (no stream response)",
                    self._model_id, prompt_tokens, output_tokens, elapsed_s,
                )
            return output
        except ImportError:
            pass

        try:
            from mlx_lm import generate
        except Exception as exc:
            raise RuntimeError("mlx-lm generate is not available.") from exc

        try:
            start_time = time.perf_counter()
            output = generate(**generate_kwargs)
            elapsed_s = time.perf_counter() - start_time

            output = self._apply_stop_tokens(output, stop_tokens)
            output_tokens = count_tokens(output, self._tokenizer)
            total_tokens = prompt_tokens + output_tokens
            elapsed_safe = max(elapsed_s, 1e-6)
            tokens_per_sec = output_tokens / elapsed_safe
            logger.info(
                "LLM generation | model=%s prompt_tokens=%d output_tokens=%d total_tokens=%d "
                "time_s=%.2f tokens_per_s=%.2f",
                self._model_id, prompt_tokens, output_tokens, total_tokens,
                elapsed_s, tokens_per_sec,
            )
            return output
        except Exception as exc:
            raise RuntimeError("mlx-lm generation failed.") from exc

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
    def _build_repetition_penalty_processor(penalty: float, context_size: int = 20):
        """Build a repetition penalty logits processor.

        Uses mlx-lm's native implementation which operates entirely in the MLX
        compute graph (pure mx.array ops, no GPU→CPU syncs).  Falls back to a
        minimal pure-MLX implementation if the native helper is unavailable.
        """
        if penalty <= 1.0:
            return None

        # Prefer mlx-lm's native, GPU-resident implementation
        try:
            from mlx_lm.sample_utils import make_logits_processors
            processors = make_logits_processors(
                repetition_penalty=penalty,
                repetition_context_size=context_size,
            )
            if processors:
                logger.debug(
                    "Using native mlx-lm repetition_penalty=%.2f context_size=%d",
                    penalty, context_size,
                )
                return processors[0]
        except (ImportError, AttributeError):
            pass

        # Fallback: pure MLX ops — no .tolist() or Python loops
        try:
            import mlx.core as mx
        except ImportError:
            return None

        def _processor(tokens, logits):
            if len(tokens) == 0:
                return logits
            recent = tokens[-context_size:]
            selected = logits[:, recent]
            selected = mx.where(
                selected < 0,
                selected * penalty,
                selected / penalty,
            )
            # Avoid fancy-index assignment (logits[:, recent] = ...), which is
            # not supported on some MLX versions in fallback environments.
            recent_idx = mx.array(recent, dtype=mx.int32)[None, :]
            recent_idx = mx.broadcast_to(recent_idx, selected.shape)
            return mx.put_along_axis(logits, recent_idx, selected, axis=-1)

        logger.debug(
            "Using fallback pure-MLX repetition_penalty=%.2f context_size=%d",
            penalty, context_size,
        )
        return _processor

    @staticmethod
    def _build_presence_penalty_processor(penalty: float):
        """Build a presence penalty logits processor.

        Subtracts a fixed `penalty` value from the logit of any token that has
        already appeared in the generated sequence (binary — appeared or didn't,
        no frequency scaling).  This differs from repetition penalty which
        multiplies by count.
        """
        if penalty <= 0:
            return None

        try:
            import mlx.core as mx
        except ImportError:
            return None

        def _processor(tokens, logits):
            if len(tokens) == 0:
                return logits
            unique_ids = mx.array(list(set(tokens)), dtype=mx.int32)
            logits[:, unique_ids] -= penalty
            return logits

        logger.debug("Using presence_penalty=%.2f", penalty)
        return _processor

    def generate(self, prompt: str, *, config: Optional[GenerationConfig] = None) -> str:
        if not prompt.strip():
            raise ValueError("prompt must be a non-empty string.")

        final_max_tokens, temperature, top_p, top_k, min_p, repetition_penalty, presence_penalty, stop_tokens, prompt_tokens = (
            self._resolve_generation_inputs(prompt, config)
        )
        return self._generate_full_text(
            prompt,
            final_max_tokens,
            temperature,
            top_p,
            top_k,
            min_p,
            repetition_penalty,
            presence_penalty,
            stop_tokens,
            prompt_tokens,
        )

    def generate_chat(
        self,
        messages: list[dict[str, str]],
        *,
        config: Optional[GenerationConfig] = None,
    ) -> str:
        """Non-streaming chat generation. All sampling params come from config."""
        if not messages:
            raise ValueError("messages must be a non-empty list of role/content dicts.")

        prompt = self._apply_chat_template(messages, enable_thinking=False)
        final_max_tokens, temperature, top_p, top_k, min_p, repetition_penalty, presence_penalty, stop_tokens, prompt_tokens = (
            self._resolve_generation_inputs(prompt, config)
        )
        output = self._generate_full_text(
            prompt,
            final_max_tokens,
            temperature,
            top_p,
            top_k,
            min_p,
            repetition_penalty,
            presence_penalty,
            stop_tokens,
            prompt_tokens,
        )
        return self._strip_thinking_blocks(output)

    def generate_chat_stream(
        self,
        messages: list[dict[str, str]],
        *,
        config: Optional[GenerationConfig] = None,
        should_stop: Optional[callable] = None,
    ):
        """Streaming version of generate_chat that yields individual tokens.

        All sampling parameters (temperature, top_p, top_k, min_p,
        presence_penalty, repetition_penalty) are resolved from
        ``GenerationConfig``.  Callers should set them there.

        Yields
        ------
        str
            Individual text tokens as they are generated.
        """
        if not messages:
            raise ValueError("messages must be a non-empty list of role/content dicts.")

        prompt = self._apply_chat_template(messages, enable_thinking=False)
        final_max_tokens, temperature, top_p, top_k, min_p, repetition_penalty, presence_penalty, stop_tokens, prompt_tokens = (
            self._resolve_generation_inputs(prompt, config)
        )

        try:
            from mlx_lm.generate import stream_generate

            yield from self._stream_tokens(
                prompt, final_max_tokens, temperature, top_p,
                top_k, min_p,
                repetition_penalty, presence_penalty, stop_tokens, prompt_tokens,
                should_stop=should_stop,
            )
        except ImportError:
            # Fallback: generate full output, yield as single token
            output = self._generate_full_text(
                prompt, final_max_tokens, temperature, top_p,
                top_k, min_p,
                repetition_penalty, presence_penalty, stop_tokens, prompt_tokens,
            )
            output = self._strip_thinking_blocks(output)
            if output:
                yield output

    def stream_chat_with_thinking(
        self,
        messages: list[dict[str, str]],
        *,
        config: Optional[GenerationConfig] = None,
        should_stop: Optional[callable] = None,
    ):
        """Stream chat with the model's reasoning chain exposed.

        All sampling parameters are resolved from ``GenerationConfig``.

        Yields dicts::
            {"type": "thinking", "text": str}  — inside <think> block
            {"type": "answer",   "text": str}  — visible response tokens

        The Qwen3 chat template appends ``<think>`` to the prompt tail when
        ``enable_thinking=True``, so the very first generated token is already
        inside the think block.  Generation parameters are floored to values
        recommended for thinking mode (temp >= 0.6, max_tokens >= 8192).
        """
        if not messages:
            raise ValueError("messages must be a non-empty list of role/content dicts.")

        prompt = self._apply_chat_template(messages, enable_thinking=True)
        final_max_tokens, resolved_temperature, resolved_top_p, top_k, min_p, repetition_penalty, presence_penalty, stop_tokens, prompt_tokens = (
            self._resolve_generation_inputs(prompt, config)
        )
        cfg = config or GenerationConfig()
        # Thinking mode: use max_internal_tokens as the total cap (thinking + answer)
        # Fall back to RAM-tier defaults if not set
        if cfg.max_internal_tokens is not None:
            effective_max_tokens = cfg.max_internal_tokens
        else:
            effective_max_tokens = max(final_max_tokens, 16384)
        # Visible cap: max_tokens from config, or sensible default
        visible_cap = cfg.max_tokens if cfg.max_tokens is not None else 2048

        # Floor temperature and min budget to sensible minimums for thinking mode
        effective_temp = max(resolved_temperature, 0.6)
        effective_top_p = max(resolved_top_p, 0.9)

        try:
            from mlx_lm.generate import stream_generate  # noqa: F401 – confirm available
        except ImportError:
            raise RuntimeError("mlx-lm stream_generate is not available for thinking mode.")

        yield from self._stream_tokens_thinking(
            prompt, effective_max_tokens, effective_temp, effective_top_p,
            top_k, min_p,
            repetition_penalty, presence_penalty, stop_tokens, prompt_tokens,
            should_stop=should_stop,
            visible_cap=visible_cap,
        )

    def _stream_tokens_thinking(
        self, prompt: str, max_tokens: int, temperature: float, top_p: float,
        top_k: int, min_p: float,
        repetition_penalty: Optional[float], presence_penalty: Optional[float],
        stop_tokens: list[str],
        prompt_tokens: int,
        *,
        should_stop: Optional[callable] = None,
        visible_cap: Optional[int] = None,
    ):
        """Yield thinking/answer classified dicts from mlx-lm stream_generate.

        The chat template pre-injects ``<think>`` at the prompt tail, so the
        stream starts already inside the think block (``in_think`` starts True).
        """
        from mlx_lm.generate import stream_generate

        generate_kwargs = self._build_generation_kwargs(
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            min_p=min_p,
            repetition_penalty=repetition_penalty,
            presence_penalty=presence_penalty,
        )

        stop_pattern = self._get_stop_tokens_pattern(stop_tokens)
        max_stop_len = max((len(t) for t in stop_tokens), default=0)
        answer_boundary_guard = max(max_stop_len - 1, 0)
        think_boundary_guard = len(self._thinking_close_tag) - 1

        start_time = time.perf_counter()
        accumulated = ""
        in_think = True  # chat template pre-injects <think> into prompt tail
        token_count = 0
        visible_count = 0

        for response in stream_generate(**generate_kwargs):
            if should_stop is not None and should_stop():
                logger.info("Thinking stream stopped by callback")
                break

            accumulated += response.text
            token_count += 1

            while accumulated:
                if in_think:
                    close_idx = accumulated.find(self._thinking_close_tag)
                    if close_idx == -1:
                        # Still inside think; emit all but the closing-tag boundary guard
                        safe = len(accumulated) - think_boundary_guard
                        if safe > 0:
                            yield {"type": "thinking", "text": accumulated[:safe]}
                            accumulated = accumulated[safe:]
                        break
                    else:
                        if close_idx > 0:
                            yield {"type": "thinking", "text": accumulated[:close_idx]}
                        accumulated = accumulated[close_idx + len(self._thinking_close_tag):]
                        in_think = False
                        # continue scanning for answer text
                else:
                    stop_match = stop_pattern.search(accumulated) if stop_pattern else None
                    if stop_match:
                        content = accumulated[:stop_match.start()]
                        if content:
                            yield {"type": "answer", "text": content}
                            visible_count += len(content)
                        logger.info(
                            "LLM thinking stream | model=%s prompt_tokens=%d output_tokens=%d "
                            "visible_tokens=%d time_s=%.2f (stop token)",
                            self._model_id, prompt_tokens, token_count, visible_count,
                            time.perf_counter() - start_time,
                        )
                        return
                    safe = len(accumulated) - answer_boundary_guard
                    if safe > 0:
                        emit_text = accumulated[:safe]
                        yield {"type": "answer", "text": emit_text}
                        visible_count += len(emit_text)
                        accumulated = accumulated[safe:]
                    elif len(accumulated) > STREAM_BUFFER_LIMIT_CHARS:
                        yield {"type": "answer", "text": accumulated}
                        visible_count += len(accumulated)
                        accumulated = ""

                    # Enforce visible token cap (approximate — character-based)
                    if visible_cap is not None and visible_count >= visible_cap * 4:
                        logger.info(
                            "Visible token cap reached (%d chars ≈ %d tokens)",
                            visible_count, visible_cap,
                        )
                        # Flush remainder as-is and stop
                        return

                    break

        # Flush remainder
        if accumulated:
            stop_match = stop_pattern.search(accumulated) if stop_pattern else None
            if stop_match:
                accumulated = accumulated[:stop_match.start()]
            if accumulated:
                chunk_type = "answer" if not in_think else "thinking"
                yield {"type": chunk_type, "text": accumulated}
                if chunk_type == "answer":
                    visible_count += len(accumulated)

        # Handle zero-visible-token edge case
        if visible_count == 0 and token_count > 0:
            logger.warning(
                "Thinking budget exhausted: %d tokens generated but no visible answer produced",
                token_count,
            )
            yield {"type": "error", "text": "THINKING_BUDGET_EXHAUSTED"}

        logger.info(
            "LLM thinking stream | model=%s prompt_tokens=%d output_tokens=%d visible_chars=%d time_s=%.2f",
            self._model_id, prompt_tokens, token_count, visible_count, time.perf_counter() - start_time,
        )

    def _stream_tokens(
        self, prompt: str, max_tokens: int, temperature: float, top_p: float,
        top_k: int, min_p: float,
        repetition_penalty: Optional[float], presence_penalty: Optional[float],
        stop_tokens: list[str],
        prompt_tokens: int,
        *,
        should_stop: Optional[callable] = None,
    ):
        """Yield individual tokens from mlx-lm stream_generate.

        Handles stop-token detection and thinking-block removal on the fly.
        """
        from mlx_lm.generate import stream_generate

        generate_kwargs = self._build_generation_kwargs(
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            min_p=min_p,
            repetition_penalty=repetition_penalty,
            presence_penalty=presence_penalty,
        )

        stop_pattern = self._get_stop_tokens_pattern(stop_tokens)
        max_stop_len = max((len(token) for token in stop_tokens), default=0)
        boundary_guard = max(
            max_stop_len - 1,
            len(self._thinking_open_tag) - 1,
            len(self._thinking_close_tag) - 1,
            0,
        )
        hidden_tail_guard = max(STREAM_TAIL_GUARD_CHARS, boundary_guard + 2)

        start_time = time.perf_counter()
        accumulated = ""
        in_thinking_block = False
        token_count = 0
        stopped_on_stop_token = False

        for response in stream_generate(**generate_kwargs):
            # Check cooperative cancellation
            if should_stop is not None and should_stop():
                logger.info("Generation stopped by should_stop callback")
                break

            token = response.text
            accumulated += token
            token_count += 1

            if in_thinking_block:
                close_pos = accumulated.find(self._thinking_close_tag)
                if close_pos == -1:
                    if len(accumulated) > STREAM_BUFFER_LIMIT_CHARS:
                        accumulated = accumulated[-hidden_tail_guard:]
                    continue

                accumulated = accumulated[close_pos + len(self._thinking_close_tag):]
                in_thinking_block = False

            open_pos = accumulated.find(self._thinking_open_tag)
            if open_pos != -1:
                visible_prefix = accumulated[:open_pos]
                if visible_prefix:
                    yield visible_prefix

                accumulated = accumulated[open_pos + len(self._thinking_open_tag):]
                in_thinking_block = True

                close_pos = accumulated.find(self._thinking_close_tag)
                if close_pos != -1:
                    accumulated = accumulated[close_pos + len(self._thinking_close_tag):]
                    in_thinking_block = False
                else:
                    continue

            stop_match = stop_pattern.search(accumulated) if stop_pattern is not None else None
            if stop_match is not None:
                full_content = accumulated[:stop_match.start()]
                if full_content:
                    yield full_content
                stopped_on_stop_token = True
                break

            if not in_thinking_block:
                safe_emit_len = len(accumulated) - boundary_guard
                if safe_emit_len > 0:
                    emit_text = accumulated[:safe_emit_len]
                    if emit_text:
                        yield emit_text
                    accumulated = accumulated[safe_emit_len:]
                elif len(accumulated) > STREAM_BUFFER_LIMIT_CHARS:
                    emit_text = accumulated[:-boundary_guard] if boundary_guard > 0 else accumulated
                    if emit_text:
                        yield emit_text
                    accumulated = accumulated[-boundary_guard:] if boundary_guard > 0 else ""

        if not in_thinking_block and accumulated:
            final_content = accumulated
            stop_match = stop_pattern.search(final_content) if stop_pattern is not None else None
            if stop_match is not None:
                final_content = final_content[:stop_match.start()]
            if final_content:
                yield final_content

        elapsed_s = time.perf_counter() - start_time
        logger.info(
            "LLM streaming generation | model=%s prompt_tokens=%d output_tokens=%d "
            "time_s=%.2f stopped_on_stop_token=%s",
            self._model_id, prompt_tokens, token_count, elapsed_s, stopped_on_stop_token,
        )

    def _apply_chat_template(self, messages: list[dict[str, str]], *, enable_thinking: bool = False) -> str:
        if hasattr(self._tokenizer, "apply_chat_template"):
            try:
                return self._tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=enable_thinking,
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

    def _strip_thinking_blocks(self, text: str) -> str:
        if not text:
            return text
        return self._thinking_block_pattern.sub("", text).strip()

    def _apply_stop_tokens(self, text: str, stop_tokens: list[str]) -> str:
        """Truncate output at first stop token occurrence."""
        stop_pattern = self._get_stop_tokens_pattern(stop_tokens)
        if stop_pattern is None:
            return text
        match = stop_pattern.search(text)
        if match is None:
            return text
        return text[:match.start()].rstrip()


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
    consecutive_fail_threshold: int = 3,  # deprecated — no longer drives loop termination
    allow_truncation: bool = True,
    min_doc_tokens: int = 50,
    log: Optional[logging.Logger] = None,
) -> BudgetPackResult:
    """Greedy token budget packing with early termination when budget is exhausted."""
    log = log or logger
    packed: list[str] = []
    packed_indices: list[int] = []
    used_tokens = 0
    skipped = 0
    truncated = 0
    consecutive_fails = 0
    effective_floor = max(min_doc_tokens, 20)

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
        if remaining < effective_floor:
            log.info(f"Budget packing stopped: remaining budget {remaining} < floor {effective_floor}")
            break
        if skipped > len(docs) // 2:
            log.info(f"Budget packing stopped: {skipped} skipped docs exceed half of {len(docs)} total")
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

    relative_limit = max(0, min(char_estimate - search_start, len(search_region)))
    if relative_limit > 0:
        for match in SENTENCE_BOUNDARY_REGEX.finditer(search_region):
            boundary_end = match.end()
            if boundary_end <= relative_limit:
                candidate = search_start + boundary_end
                if best_pos is None or candidate > best_pos:
                    best_pos = candidate

    truncated = text[:best_pos or char_estimate].rstrip()
    actual_tokens = count_tokens(truncated, tokenizer)

    if actual_tokens <= target_tokens * (1 + tolerance):
        return truncated
    if actual_tokens > target_tokens:
        ratio = target_tokens / actual_tokens
        return text[:int(len(truncated) * ratio * 0.9)].rstrip()
    return truncated
