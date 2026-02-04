from __future__ import annotations

import logging
import os
import platform
from dataclasses import dataclass, replace
from typing import Optional

logger = logging.getLogger(__name__)

# Global cache for detected RAM (avoid repeated detection)
_detected_ram_gb: Optional[float] = None


@dataclass(frozen=True)
class ModelConfig:
    """Configuration for RAG pipeline models and retrieval parameters.
    
    Attributes:
        mode: Operating mode (regular, power-fast, power-deep-research)
        llm_model: MLX model identifier for generation
        embedding_model: Sentence transformer model for embeddings
        reranker_model: FlagEmbedding reranker model
        embedding_device: Device for embedding model (cpu to preserve VRAM)
        quantization: MLX quantization level (4-bit, 8-bit)
        context_window: Maximum context window size in tokens
        retrieval_budget: Token budget for retrieved context
        top_k_dense: Number of dense search results
        top_k_sparse: Number of sparse (BM25) search results
        top_k_fused: Number of results after RRF fusion
        top_k_rerank: Number of results to pass to reranker
        top_k_final: Number of final results to return
        reranker_threshold: Minimum reranker score to keep document
        reranker_min_docs: Safety net minimum documents (ignores threshold)
        system_ram_gb: Detected system RAM (for logging)
    """
    mode: str
    llm_model: str
    embedding_model: str
    reranker_model: str
    embedding_device: str = "cpu"
    quantization: str = "4-bit"
    context_window: int = 128_000
    retrieval_budget: int = 100_000
    top_k_dense: int = 100
    top_k_sparse: int = 100
    top_k_fused: int = 50
    top_k_rerank: int = 20
    top_k_final: int = 5
    reranker_threshold: float = -6.0
    reranker_min_docs: int = 3
    system_ram_gb: float = 0.0


# =============================================================================
# Hardware-Aware Mode Configurations
# =============================================================================
# Token budgets are dynamically adjusted based on system RAM to ensure
# the model + KV cache fits within hardware limits.
#
# 32GB (M1/M2/M3 Pro): Only regular mode, constrained budgets
# 64GB+ (M-Max): power-fast and power-deep-research available

def _get_mode_config(mode: str, ram_gb: float) -> ModelConfig:
    """Get mode configuration with RAM-aware token budget adjustments.
    
    Args:
        mode: Operating mode name
        ram_gb: Detected system RAM in GB
        
    Returns:
        ModelConfig with appropriate budgets for the hardware
    """
    # Base configurations (will be adjusted based on RAM)
    if mode == "regular":
        # Regular: 30B 4-bit model (~18GB) - works on 32GB+
        if ram_gb < 48:
            # 32GB system: Constrained budgets to fit model + KV cache
            return ModelConfig(
                mode="regular",
                llm_model="mlx-community/Qwen3-30B-A3B-Instruct-4bit",
                embedding_model="BAAI/bge-m3",
                reranker_model="BAAI/bge-reranker-v2-m3",
                embedding_device="cpu",
                quantization="4-bit",
                context_window=64_000,
                retrieval_budget=32_000,
                top_k_dense=100,
                top_k_sparse=100,
                top_k_fused=50,
                top_k_rerank=20,
                top_k_final=5,
                reranker_threshold=-6.0,  # Aggressive filtering for RAM protection
                reranker_min_docs=3,
                system_ram_gb=ram_gb,
            )
        else:
            # 64GB+ system: Full budgets
            return ModelConfig(
                mode="regular",
                llm_model="mlx-community/Qwen3-30B-A3B-Instruct-4bit",
                embedding_model="BAAI/bge-m3",
                reranker_model="BAAI/bge-reranker-v2-m3",
                embedding_device="cpu",
                quantization="4-bit",
                context_window=64_000,
                retrieval_budget=32_000,
                top_k_dense=100,
                top_k_sparse=100,
                top_k_fused=50,
                top_k_rerank=20,
                top_k_final=5,
                reranker_threshold=-6.0,  # Aggressive filtering for RAM protection
                reranker_min_docs=3,
                system_ram_gb=ram_gb,
            )
    
    elif mode == "power-fast":
        # Power-fast: 30B 8-bit model (~36GB) - requires 64GB+
        if ram_gb < 64:
            logger.warning(
                f"power-fast mode requires 64GB+ RAM for optimal performance. "
                f"Detected {ram_gb:.0f}GB. Consider using 'regular' mode."
            )
        return ModelConfig(
            mode="power-fast",
            llm_model="mlx-community/Qwen3-30B-A3B-Instruct-2507-8bit",
            embedding_model="BAAI/bge-m3",
            reranker_model="BAAI/bge-reranker-v2-m3",
            embedding_device="cpu",
            quantization="8-bit",
            context_window=96_000,
            retrieval_budget=50_000,
            top_k_dense=300,
            top_k_sparse=300,
            top_k_fused=150,
            top_k_rerank=80,
            top_k_final=15,
            reranker_threshold=-8.0,  # Balanced filtering for speed optimization
            reranker_min_docs=5,
            system_ram_gb=ram_gb,
        )
    
    elif mode == "power-deep-research":
        # Power-deep-research: 80B 4-bit model (~48GB) - requires 64GB+
        if ram_gb < 64:
            logger.warning(
                f"power-deep-research mode requires 64GB+ RAM. "
                f"Detected {ram_gb:.0f}GB. Performance will be severely degraded."
            )
        return ModelConfig(
            mode="power-deep-research",
            llm_model="mlx-community/Qwen3-Next-80B-A3B-Instruct-4bit",
            embedding_model="BAAI/bge-m3",
            reranker_model="BAAI/bge-reranker-v2-m3",
            embedding_device="cpu",
            quantization="4-bit",
            context_window=32_000,
            retrieval_budget=20_000,
            top_k_dense=300,
            top_k_sparse=300,
            top_k_fused=150,
            top_k_rerank=80,
            top_k_final=15,
            reranker_threshold=-10.0,  # Permissive filtering for max recall
            reranker_min_docs=10,
            system_ram_gb=ram_gb,
        )
    
    raise ValueError(f"Unknown mode: {mode}")


# Valid mode names
VALID_MODES = {"regular", "power-fast", "power-deep-research"}

# RAM requirements per mode (in GB) - for validation/warnings
MODE_RAM_REQUIREMENTS: dict[str, float] = {
    "regular": 32.0,
    "power-fast": 64.0,
    "power-deep-research": 64.0,
}


# =============================================================================
# System Detection
# =============================================================================

def _detect_ram_gb() -> float:
    """Detect system RAM in gigabytes.
    
    Uses psutil if available for cross-platform accuracy,
    falls back to platform-specific methods.
    Caches result for subsequent calls.
    
    Returns:
        System RAM in GB, or 0.0 if detection fails.
    """
    global _detected_ram_gb
    if _detected_ram_gb is not None:
        return _detected_ram_gb
    
    ram_gb = 0.0
    
    # Try psutil first (most reliable)
    try:
        import psutil
        ram_gb = psutil.virtual_memory().total / (1024**3)
        _detected_ram_gb = ram_gb
        return ram_gb
    except ImportError:
        pass
    except Exception as exc:
        logger.debug(f"psutil RAM detection failed: {exc}")

    # Fallback: macOS-specific
    system = platform.system().lower()
    if system == "darwin":
        try:
            import subprocess
            output = subprocess.check_output(["sysctl", "-n", "hw.memsize"], text=True)
            ram_gb = int(output.strip()) / (1024**3)
            _detected_ram_gb = ram_gb
            return ram_gb
        except Exception as exc:
            logger.debug(f"macOS RAM detection failed: {exc}")

    # Fallback: Linux sysconf
    if hasattr(os, "sysconf"):
        try:
            pages = os.sysconf("SC_PHYS_PAGES")
            page_size = os.sysconf("SC_PAGE_SIZE")
            ram_gb = (pages * page_size) / (1024**3)
            _detected_ram_gb = ram_gb
            return ram_gb
        except (ValueError, OSError, AttributeError) as exc:
            logger.debug(f"sysconf RAM detection failed: {exc}")

    _detected_ram_gb = ram_gb
    return ram_gb


def get_system_ram_gb() -> float:
    """Public accessor for detected system RAM.
    
    Returns:
        System RAM in GB (cached after first detection).
    """
    return _detect_ram_gb()


def _auto_select_mode(ram_gb: float) -> str:
    """Auto-select the best mode based on available RAM.
    
    Args:
        ram_gb: Detected system RAM in GB
        
    Returns:
        Recommended mode name
    """
    if ram_gb >= 64:
        return "power-fast"  # 64GB+ can handle 8-bit model comfortably
    else:
        return "regular"  # 32GB systems use 4-bit model


# =============================================================================
# Mode Selection
# =============================================================================

def select_mode_config(
    *,
    manual_mode: Optional[str] = None,
    validate_ram: bool = True,
) -> ModelConfig:
    """Select configuration based on mode with CLI/env var precedence.
    
    Resolution order:
    1. manual_mode argument (from CLI --mode flag)
    2. RAG_MODE environment variable
    3. Auto-detection based on system RAM
    
    Token budgets are automatically adjusted based on detected RAM
    to ensure model + KV cache fits within hardware limits.
    
    Args:
        manual_mode: Mode override from CLI (highest priority)
        validate_ram: Whether to check RAM meets mode requirements
        
    Returns:
        ModelConfig for the selected mode with RAM-appropriate budgets
        
    Raises:
        ValueError: If specified mode is unknown
    """
    # Detect system RAM
    ram_gb = _detect_ram_gb()
    
    # Determine mode with precedence: CLI > env var > auto
    mode = (manual_mode or os.getenv("RAG_MODE", "")).strip().lower()
    source = "cli" if manual_mode else "env" if os.getenv("RAG_MODE") else "auto"
    
    if not mode:
        mode = _auto_select_mode(ram_gb)
        source = "auto"
        logger.info(
            f"Auto-selected mode '{mode}' based on {ram_gb:.0f}GB detected RAM"
        )
    
    # Handle legacy tier names for backward compatibility
    legacy_mapping = {
        "high": "power-fast",
        "high-performance": "power-fast",
        "tier1": "power-fast",
        "efficiency": "regular",
        "tier2": "regular",
    }
    if mode in legacy_mapping:
        old_mode = mode
        mode = legacy_mapping[mode]
        logger.warning(
            f"Legacy tier '{old_mode}' mapped to mode '{mode}'. "
            f"Please update to use --mode={mode}"
        )
    
    if mode not in VALID_MODES:
        valid_modes = ", ".join(sorted(VALID_MODES))
        raise ValueError(
            f"Unknown mode '{mode}'. Valid modes: {valid_modes}. "
            f"Set RAG_MODE env var or use --mode flag."
        )
    
    # Get RAM-aware configuration
    config = _get_mode_config(mode, ram_gb)
    
    logger.info(
        f"[Hardware: {ram_gb:.0f}GB] Selected mode: {mode} (source: {source}) | "
        f"LLM: {config.llm_model} | "
        f"Quantization: {config.quantization} | "
        f"Context: {config.context_window:,} | "
        f"Budget: {config.retrieval_budget:,}"
    )
    
    return config


# =============================================================================
# Legacy API (Deprecated)
# =============================================================================

def select_model_config(*, manual_tier: Optional[str] = None) -> ModelConfig:
    """Legacy function for backward compatibility.
    
    Deprecated: Use select_mode_config() instead.
    
    Args:
        manual_tier: Legacy tier name
        
    Returns:
        ModelConfig from the mapped mode
    """
    logger.warning(
        "select_model_config() is deprecated. Use select_mode_config() instead."
    )
    return select_mode_config(manual_mode=manual_tier)
