"""
SGLang Qwen3 Next Plugin

This plugin provides a customized implementation of Qwen3 Next model for SGLang.
The plugin can be loaded via:
1. Entry points mechanism (automatic, recommended)
2. Environment variable SGLANG_EXTERNAL_MODEL_PACKAGE
3. Manual registration

Note: The EntryClass is defined in qwen3_next.py (package root), which will be automatically
discovered by SGLang's ModelRegistry when SGLANG_EXTERNAL_MODEL_PACKAGE is set.
SGLang's import_model_classes only scans direct submodules, not nested packages.
"""

# Import for backward compatibility and manual registration
from sglang_qwen3_next_plugin.qwen3_next import Qwen3NextForCausalLM

# Auto-register plugin model to override SGLang's default implementation
# This ensures the plugin model takes precedence when both are available
def register():
    """Register the plugin model with SGLang's ModelRegistry.
    
    This function is called via entry points when SGLang starts.
    It can also be called manually for testing or alternative registration methods.
    """
    try:
        from sglang.srt.models.registry import ModelRegistry
        ModelRegistry.models["Qwen3NextForCausalLM"] = Qwen3NextForCausalLM
        print('\33[92m[SGLang Qwen3 Next Plugin] Model registered via entry point\33[0m')
        return True
    except (ImportError, AttributeError) as e:
        # SGLang not available or ModelRegistry not initialized yet
        print(f'\33[93m[SGLang Qwen3 Next Plugin] Registration deferred: {e}\33[0m')
        return False

# Auto-register on import (fallback mechanism)
# This works even if entry points are not used
# Use a delayed registration approach to ensure it happens after SGLang initializes
def _auto_register():
    """Auto-register the plugin model."""
    try:
        from sglang.srt.models.registry import ModelRegistry
        # Force registration to override default model
        ModelRegistry.models["Qwen3NextForCausalLM"] = Qwen3NextForCausalLM
        return True
    except (ImportError, AttributeError):
        return False

# Try immediate registration
_auto_register()

# Also set up a hook for delayed registration
# This will be called when SGLang's registry is fully initialized
import atexit
atexit.register(_auto_register)

# MODIFIED: Patch SGLang's profile_max_num_token to handle empty full_attention_layer_ids
# This fixes the cell_size=0 issue when all layers are linear_attention
def _patch_profile_max_num_token():
    """Patch ModelRunner.profile_max_num_token to handle empty full_attention_layer_ids."""
    try:
        from sglang.srt.model_executor.model_runner import ModelRunner
        original_method = ModelRunner.profile_max_num_token
        
        def patched_profile_max_num_token(self, total_gpu_memory: int):
            """Patched version that handles empty full_attention_layer_ids."""
            from sglang.srt.utils import get_available_gpu_memory
            from sglang.srt.distributed import get_world_group
            from sglang.srt.layers.dp_attention import get_attention_tp_size
            import torch
            
            available_gpu_memory = get_available_gpu_memory(
                self.device,
                self.gpu_id,
                distributed=get_world_group().world_size > 1,
                cpu_group=get_world_group().cpu_group,
            )
            if self.is_draft_worker:
                num_layers = getattr(
                    self.model_config.hf_config,
                    "num_nextn_predict_layers",
                    self.num_effective_layers,
                )
            elif self.is_hybrid_gdn:
                # MODIFIED: Handle empty full_attention_layer_ids
                # If empty, use num_effective_layers to avoid cell_size=0
                full_attention_layer_ids = self.model_config.hf_config.full_attention_layer_ids
                num_layers = len(full_attention_layer_ids) if full_attention_layer_ids else self.num_effective_layers
            else:
                num_layers = self.num_effective_layers
            if self.use_mla_backend:
                cell_size = (
                    (self.model_config.kv_lora_rank + self.model_config.qk_rope_head_dim)
                    * num_layers
                    * torch._utils._element_size(self.kv_cache_dtype)
                )
            else:
                cell_size = (
                    self.model_config.get_num_kv_heads(get_attention_tp_size())
                    * self.model_config.head_dim
                    * num_layers
                    * 2
                    * torch._utils._element_size(self.kv_cache_dtype)
                )
            rest_memory = available_gpu_memory - total_gpu_memory * (
                1 - self.mem_fraction_static
            )
            if self.is_hybrid_gdn:
                rest_memory -= (
                    self.server_args.max_mamba_cache_size
                    * self.model_config.hf_config.mamba_cache_per_req
                    / (1 << 30)
                )
            # MODIFIED: Ensure cell_size is not zero
            if cell_size == 0:
                # Fallback: use a minimal cell_size based on num_effective_layers
                cell_size = (
                    self.model_config.get_num_kv_heads(get_attention_tp_size())
                    * self.model_config.head_dim
                    * self.num_effective_layers
                    * 2
                    * torch._utils._element_size(self.kv_cache_dtype)
                )
            max_num_token = int(rest_memory * (1 << 30) // cell_size)
            return max_num_token
        
        # Apply the patch
        ModelRunner.profile_max_num_token = patched_profile_max_num_token
        return True
    except (ImportError, AttributeError):
        return False

# Apply the patch when module is imported
_patch_profile_max_num_token()

__all__ = ["Qwen3NextForCausalLM", "register"]

print(f'\33[92m[SGLang Qwen3 Next Plugin] Package loaded\33[0m')
