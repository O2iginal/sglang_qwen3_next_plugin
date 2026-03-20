#!/usr/bin/env python3
"""
最小插件入口检查。

只验证两件事：
1. 包可以在目标 venv 中被导入
2. 包根导出了 SGLang 原生插件发现所需的 EntryClass
"""

import importlib
import os
import sys


def prepare_env() -> None:
    os.environ.setdefault("HOME", "/tmp")
    os.environ.setdefault("XDG_CACHE_HOME", "/tmp")
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def test_entryclass_contract() -> None:
    prepare_env()
    module = importlib.import_module("sglang_qwen3_next_plugin")

    assert hasattr(module, "EntryClass"), "包根未导出 EntryClass"

    entry_class = module.EntryClass
    assert entry_class.__name__ == "Qwen3NextForCausalLM", (
        f"EntryClass 名称异常: {entry_class.__name__}"
    )
    print("✓ EntryClass 已导出")
    print(f"✓ EntryClass: {entry_class}")


def test_upstream_baseline_contract() -> None:
    prepare_env()
    module = importlib.import_module("sglang_qwen3_next_plugin.qwen3_next_upstream")
    assert hasattr(module, "Qwen3NextForCausalLM"), "缺少上游基线模型类"
    print("✓ 上游基线模块已存在")
    print(f"✓ Upstream class: {module.Qwen3NextForCausalLM}")


def test_layer_types_override_contract() -> None:
    prepare_env()
    importlib.import_module("sglang_qwen3_next_plugin")
    from sglang.srt.configs.qwen3_next import Qwen3NextConfig

    cfg = Qwen3NextConfig(
        num_hidden_layers=4,
        full_attention_interval=0,
        layer_types=[
            "linear_attention",
            "full_attention",
            "linear_attention",
            "full_attention",
        ],
    )
    assert cfg.layers_block_type == [
        "linear_attention",
        "attention",
        "linear_attention",
        "attention",
    ], f"layer_types 未被正确采用: {cfg.layers_block_type}"
    print("✓ 显式 layer_types 会覆盖 full_attention_interval 推导")


def test_conv_backend_override_contract() -> None:
    prepare_env()
    importlib.import_module("sglang_qwen3_next_plugin")
    import sglang.srt.layers.attention.hybrid_linear_attn_backend as backend
    from sglang.srt.layers.attention.mamba.causal_conv1d import causal_conv1d_fn

    assert (
        backend.causal_conv1d_fn is causal_conv1d_fn
    ), "hybrid linear attention backend 未切换到稳定 causal_conv1d_fn"
    print("✓ hybrid linear attention backend 已切换到稳定 causal_conv1d_fn")


def test_hybrid_gdn_conv_dtype_contract() -> None:
    prepare_env()
    importlib.import_module("sglang_qwen3_next_plugin")
    from sglang.srt.configs.qwen3_next import Qwen3NextConfig
    import sglang_qwen3_next_plugin.qwen3_next as plugin_qwen3_next
    import torch

    original_get_attention_tp_size = plugin_qwen3_next.get_attention_tp_size
    plugin_qwen3_next.get_attention_tp_size = lambda: 1
    cfg = Qwen3NextConfig(
        num_hidden_layers=2,
        full_attention_interval=1,
        layer_types=["linear_attention", "linear_attention"],
        torch_dtype=torch.float16,
    )
    try:
        conv_dtype = cfg.hybrid_gdn_params[2]
        assert conv_dtype == torch.float16, f"conv_state dtype 异常: {conv_dtype}"
        print("✓ hybrid_gdn_params 会跟随模型 dtype 选择 conv_state dtype")
    finally:
        plugin_qwen3_next.get_attention_tp_size = original_get_attention_tp_size


def test_model_runner_dtype_patch_contract() -> None:
    prepare_env()
    importlib.import_module("sglang_qwen3_next_plugin")
    import sglang.srt.model_executor.model_runner as model_runner_module

    assert getattr(
        model_runner_module, "_sglang_qwen3_next_plugin_dtype_patch", False
    ), "ModelRunner.init_memory_pool 未打上 dtype 同步补丁"
    print("✓ ModelRunner.init_memory_pool 已添加 hybrid_gdn dtype 同步补丁")


if __name__ == "__main__":
    print("Testing SGLang Qwen3 Next Plugin...")
    print("=" * 60)
    try:
        test_entryclass_contract()
        test_upstream_baseline_contract()
        test_layer_types_override_contract()
        test_conv_backend_override_contract()
        test_hybrid_gdn_conv_dtype_contract()
        test_model_runner_dtype_patch_contract()
    except AssertionError as exc:
        print(f"✗ Contract failed: {exc}")
        sys.exit(1)
    except Exception as exc:
        print(f"✗ Unexpected error: {exc}")
        sys.exit(1)
    print("=" * 60)
    print("Plugin entry contract is correct!")
