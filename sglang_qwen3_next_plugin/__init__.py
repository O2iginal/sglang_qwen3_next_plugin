"""
SGLang Qwen3 Next Plugin

This plugin provides a customized implementation of Qwen3 Next model for SGLang.
The plugin can be loaded via environment variable SGLANG_EXTERNAL_MODEL_PACKAGE.
"""

from sglang_qwen3_next_plugin._0_5_2.qwen3_next import Qwen3NextForCausalLM

# Export the model class for ModelRegistry
EntryClass = [Qwen3NextForCausalLM]

__all__ = ["Qwen3NextForCausalLM", "EntryClass"]
