"""Compatibility patch for FlagEmbedding with transformers 5.x"""
from __future__ import annotations

import sys


def patch_transformers_compatibility() -> None:
    """Patch missing imports in transformers 5.x for backward compatibility with FlagEmbedding."""
    try:
        import transformers.utils.import_utils as import_utils
        
        # Add missing is_torch_fx_available for FlagEmbedding compatibility
        if not hasattr(import_utils, 'is_torch_fx_available'):
            def is_torch_fx_available() -> bool:
                """Stub for removed is_torch_fx_available function."""
                return False
            
            import_utils.is_torch_fx_available = is_torch_fx_available
            
    except ImportError:
        pass
    
    # Patch tokenizer prepare_for_model method for FlagEmbedding reranker
    try:
        from transformers import PreTrainedTokenizerBase
        
        if not hasattr(PreTrainedTokenizerBase, 'prepare_for_model'):
            original_call = PreTrainedTokenizerBase.__call__
            
            def prepare_for_model(self, ids, pair_ids=None, **kwargs):
                """Compatibility wrapper for prepare_for_model in transformers 5.x."""
                text = self.decode(ids, skip_special_tokens=False)
                pair = self.decode(pair_ids, skip_special_tokens=False) if pair_ids else None
                return self(text, text_pair=pair, **kwargs)
            
            PreTrainedTokenizerBase.prepare_for_model = prepare_for_model
            
    except (ImportError, AttributeError):
        pass
