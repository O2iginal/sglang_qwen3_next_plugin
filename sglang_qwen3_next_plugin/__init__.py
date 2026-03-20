"""
SGLang Qwen3 Next 外部插件入口。

该模块只负责暴露 SGLang 原生插件发现所需的模型类与 EntryClass。
"""

from sglang_qwen3_next_plugin.qwen3_next import Qwen3NextForCausalLM

EntryClass = Qwen3NextForCausalLM

__all__ = ["Qwen3NextForCausalLM", "EntryClass"]
