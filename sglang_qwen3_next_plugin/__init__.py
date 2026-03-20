"""
SGLang Qwen3 Next 外部插件入口。

包根保持轻量，只在真正请求模型类时才导入重型实现。
"""

__all__ = ["Qwen3NextForCausalLM", "EntryClass"]


def __getattr__(name: str):
    if name in {"Qwen3NextForCausalLM", "EntryClass"}:
        from sglang_qwen3_next_plugin.qwen3_next import Qwen3NextForCausalLM

        if name == "EntryClass":
            return Qwen3NextForCausalLM
        return Qwen3NextForCausalLM
    raise AttributeError(name)
