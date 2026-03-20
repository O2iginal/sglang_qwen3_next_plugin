"""SGLang 0.5.2 专用 compat 补丁。"""

from __future__ import annotations

import os

import torch

from sglang.srt.configs.qwen3_next import HybridLayerType, Qwen3NextConfig
from sglang.srt.distributed import divide
from sglang.srt.layers.attention.mamba.causal_conv1d import (
    causal_conv1d_fn as stable_causal_conv1d_fn,
)
from sglang.srt.layers.dp_attention import get_attention_tp_size
import sglang.srt.layers.attention.hybrid_linear_attn_backend as hybrid_linear_attn_backend
import sglang.srt.model_executor.model_runner as model_runner_module


def _plugin_layers_block_type(config: Qwen3NextConfig):
    explicit_layer_types = getattr(config, "layer_types", None)
    if explicit_layer_types is not None:
        if len(explicit_layer_types) != config.num_hidden_layers:
            raise ValueError(
                "config.layer_types length does not match num_hidden_layers: "
                f"{len(explicit_layer_types)} != {config.num_hidden_layers}"
            )
        mapping = {
            "full_attention": HybridLayerType.full_attention.value,
            HybridLayerType.full_attention.value: HybridLayerType.full_attention.value,
            "linear_attention": HybridLayerType.linear_attention.value,
            HybridLayerType.linear_attention.value: HybridLayerType.linear_attention.value,
            "swa_attention": HybridLayerType.swa_attention.value,
            HybridLayerType.swa_attention.value: HybridLayerType.swa_attention.value,
            "mamba": HybridLayerType.mamba2.value,
            HybridLayerType.mamba2.value: HybridLayerType.mamba2.value,
        }
        try:
            return [mapping[layer_type] for layer_type in explicit_layer_types]
        except KeyError as exc:
            raise ValueError(f"unsupported layer_types entry: {exc.args[0]}") from exc

    full_attention_interval = getattr(config, "full_attention_interval", None)
    if not full_attention_interval:
        raise ValueError(
            "full_attention_interval must be non-zero when layer_types is absent"
        )

    layer_type_list = []
    for layer_idx in range(config.num_hidden_layers):
        if (layer_idx + 1) % full_attention_interval == 0:
            layer_type_list.append(HybridLayerType.full_attention.value)
        else:
            layer_type_list.append(HybridLayerType.linear_attention.value)
    return layer_type_list


def _plugin_hybrid_gdn_params(config: Qwen3NextConfig):
    world_size = get_attention_tp_size()
    conv_dim = (
        config.linear_key_head_dim * config.linear_num_key_heads * 2
        + config.linear_value_head_dim * config.linear_num_value_heads
    )
    conv_state_shape = (
        divide(conv_dim, world_size),
        config.linear_conv_kernel_dim - 1,
    )

    temporal_state_shape = (
        divide(config.linear_num_value_heads, world_size),
        config.linear_key_head_dim,
        config.linear_value_head_dim,
    )

    dtype_value = getattr(config, "torch_dtype", None)
    if dtype_value is None:
        dtype_value = getattr(config, "dtype", None)
    dtype_map = {
        "float16": torch.float16,
        "torch.float16": torch.float16,
        torch.float16: torch.float16,
        "bfloat16": torch.bfloat16,
        "torch.bfloat16": torch.bfloat16,
        torch.bfloat16: torch.bfloat16,
        "float32": torch.float32,
        "torch.float32": torch.float32,
        torch.float32: torch.float32,
    }
    conv_dtype = dtype_map.get(dtype_value, torch.bfloat16)

    ssm_dtype_map = {
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
    }
    ssm_dtype = ssm_dtype_map[os.environ.get("SGLANG_MAMBA_SSM_DTYPE", "float32")]
    mamba_layers = [
        idx
        for idx, type_value in enumerate(config.layers_block_type)
        if type_value == HybridLayerType.linear_attention.value
    ]
    return (
        conv_state_shape,
        temporal_state_shape,
        conv_dtype,
        ssm_dtype,
        mamba_layers,
    )


def apply_runtime_compat() -> list[str]:
    applied: list[str] = []

    if not getattr(Qwen3NextConfig, "_sglang_qwen3_next_plugin_layer_types_patch", False):
        original_init = Qwen3NextConfig.__init__

        def _patched_qwen3_next_config_init(self, *args, layer_types=None, **kwargs):
            original_init(self, *args, layer_types=layer_types, **kwargs)
            self.layer_types = layer_types

        Qwen3NextConfig.__init__ = _patched_qwen3_next_config_init
        Qwen3NextConfig.layers_block_type = property(_plugin_layers_block_type)
        Qwen3NextConfig.hybrid_gdn_params = property(_plugin_hybrid_gdn_params)
        Qwen3NextConfig._sglang_qwen3_next_plugin_layer_types_patch = True
    applied.append("layer_types")

    if not getattr(
        hybrid_linear_attn_backend, "_sglang_qwen3_next_plugin_conv_patch", False
    ):
        hybrid_linear_attn_backend.causal_conv1d_fn = stable_causal_conv1d_fn
        hybrid_linear_attn_backend._sglang_qwen3_next_plugin_conv_patch = True
    applied.append("hybrid_conv_backend")

    if not getattr(model_runner_module, "_sglang_qwen3_next_plugin_dtype_patch", False):
        original_init_memory_pool = model_runner_module.ModelRunner.init_memory_pool

        def _patched_init_memory_pool(self, *args, **kwargs):
            if getattr(self, "is_hybrid_gdn", False):
                hf_config = getattr(getattr(self, "model_config", None), "hf_config", None)
                model_dtype = getattr(getattr(self, "model_config", None), "dtype", None)
                if hf_config is not None and model_dtype is not None:
                    hf_config.torch_dtype = model_dtype
                    hf_config.dtype = model_dtype
            return original_init_memory_pool(self, *args, **kwargs)

        model_runner_module.ModelRunner.init_memory_pool = _patched_init_memory_pool
        model_runner_module._sglang_qwen3_next_plugin_dtype_patch = True
    applied.append("model_runner_dtype")

    return applied
