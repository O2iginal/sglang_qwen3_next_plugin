# DeepWiki Q&A with Code Context for Repository: sgl-project/sglang
## Q1
如果我想要支持一个新的模型，应该如何做？
### Answer
要在 SGLang 中支持新模型，您需要根据模型类型（语言模型或多模态模型）执行不同的步骤。

## 概述

对于语言模型，您只需在 `python/sglang/srt/models/` 目录下添加一个文件 sglang:8-12 。对于多模态模型，需要额外的组件注册和处理。

## 支持新语言模型

### 1. 创建模型文件
在 `python/sglang/srt/models/` 目录下创建新文件，可以参考现有实现如 `qwen2.py` sglang:1-100  或 `llama.py` sglang:393-449 。

### 2. 关键实现要点
- 继承适当的基类并实现 `forward()` 和 `forward_batch()` 方法
- 使用 SGLang 的组件替换 vLLM 组件：
  - `RadixAttention` 替换 vLLM 的 `Attention` sglang:97-98 
  - `LogitsProcessor` 替换 vLLM 的 `LogitsProcessor` sglang:98-99 
  - 其他层如 `RMSNorm`、`SiluAndMul` 等 sglang:100-101 
- 添加 `EntryClass` 在文件末尾 sglang:103-104 

### 3. 模型注册
模型通过 `ModelRegistry` 自动注册，`get_model_architecture()` 函数会根据配置解析模型类 sglang:84-106 。

## 支持新多模态模型

### 1. 注册为多模态模型
在 `model_config.py` 中扩展 `is_multimodal_model` 函数 sglang:19-22 。

### 2. 必要组件
- **聊天模板**：在 `conversation.py` 中注册新模板 sglang:24-25 
- **多模态处理器**：继承 `BaseMultimodalProcessor` sglang:27-31 
- **输入处理**：实现 `pad_input_ids()` 函数 sglang:33-36 
- **图像特征提取**：实现 `get_image_feature()` 函数 sglang:38-40 
- **视觉注意力**：使用 `VisionAttention` 适配 ViT sglang:41-43 

参考 `qwen2_5_vl.py` 等实现 sglang:1-80 。

## 测试和调试

### 1. 交互式调试
对比 HuggingFace 和 SGLang 输出：
```bash
# HF 参考输出
python3 scripts/playground/reference_hf.py --model-path [new model] --model-type {text,mllm}
# SGLang 输出
python3 -m sglang.bench_one_batch --correct --model [new model]
``` sglang:53-63 

### 2. 添加到测试套件
将模型添加到 `test_generation_models.py` 的 `ALL_OTHER_MODELS` 列表中 sglang:67-71 。

### 3. 基准测试
运行 MMMU 基准测试确保精度 sglang:80-83 。

## 外部模型注册

您也可以在不修改源码的情况下注册模型：

```python
from sglang.srt.models.registry import ModelRegistry
ModelRegistry.models[model_name] = model_class
``` sglang:110-136 

## 注意事项

- 确保新实现只使用 SGLang 组件，不依赖 vLLM 组件 sglang:104-105 
- 将新模型添加到支持模型文档中 sglang:106-107 
- 对于特殊架构（如 MLA），需要在 `model_config.py` 中添加特殊处理 sglang:373-451 

## Notes

- SGLang 支持多种硬件后端，包括 NPU（Ascend） sglang:31-97 
- 对于 MoE 模型，需要考虑专家并行配置 sglang:443-467 
- KV 缓存管理根据模型架构（如 MLA、NSA）有不同的实现 sglang:47-110 

Wiki pages you might want to explore:
- [Expert Parallelism for MoE Models (sgl-project/sglang)](/wiki/sgl-project/sglang#5.2)
### Citations
**File:** docs/supported_models/support_new_models.md (L8-12)
```markdown
To support a new model in SGLang, you only need to add a single file under
the [SGLang Models Directory](https://github.com/sgl-project/sglang/tree/main/python/sglang/srt/models). You can learn
from existing model implementations and create a new file for your model. For most models, you should be able to find a
similar model to start with (e.g., starting from Llama). Also refer how
to [port a Model from vLLM to SGLang](#port-a-model-from-vllm-to-sglang)
```
**File:** docs/supported_models/support_new_models.md (L19-22)
```markdown
1. **Register your new model as multimodal**:
   Extend `is_multimodal_model`
   in [model_config.py](https://github.com/sgl-project/sglang/blob/0ab3f437aba729b348a683ab32b35b214456efc7/python/sglang/srt/configs/model_config.py#L561)
   to return `True` for your model.
```
**File:** docs/supported_models/support_new_models.md (L24-25)
```markdown
2. **Register a new chat-template**:
   Only when your default chat-template is unable to accept images as input: Register a new chat template in [conversation.py](https://github.com/sgl-project/sglang/tree/main/python/sglang/srt/conversation.py) and the corresponding matching function.
```
**File:** docs/supported_models/support_new_models.md (L27-31)
```markdown
3. **Multimodal Data Processor**:
   Define a new `Processor` class that inherits from `BaseMultimodalProcessor` and register this processor as your
   model’s dedicated processor.
   See [multimodal_processor.py](https://github.com/sgl-project/sglang/tree/main/python/sglang/srt/multimodal/processors)
   for more details.
```
**File:** docs/supported_models/support_new_models.md (L33-36)
```markdown
4. **Handle Multimodal Tokens**:
   Implement a `pad_input_ids` function for your new model. In this function, multimodal tokens in the prompt should be
   expanded (if necessary) and padded with multimodal-data-hashes so that SGLang can recognize different multimodal data
   with `RadixAttention`.
```
**File:** docs/supported_models/support_new_models.md (L38-40)
```markdown
5. **Handle Image Feature Extraction**:
   Implement a `get_image_feature` function for your new model, which extracts image features from raw image data and converts them into the embeddings used by the language model.

```
**File:** docs/supported_models/support_new_models.md (L41-43)
```markdown
6. **Adapt to Vision Attention**:
   Adapt the multi-headed `Attention` of ViT with SGLang’s `VisionAttention`.

```
**File:** docs/supported_models/support_new_models.md (L53-63)
```markdown
For interactive debugging, compare the outputs of Hugging Face/Transformers and SGLang. The following two commands
should give the same text output and very similar prefill logits:

- Get the reference output:
  ```bash
  python3 scripts/playground/reference_hf.py --model-path [new model] --model-type {text,mllm}
  ```
- Get the SGLang output:
  ```bash
  python3 -m sglang.bench_one_batch --correct --model [new model]
  ```
```
**File:** docs/supported_models/support_new_models.md (L67-71)
```markdown
To ensure the new model is well maintained, add it to the test suite by including it in the `ALL_OTHER_MODELS` list in
the [test_generation_models.py](https://github.com/sgl-project/sglang/blob/main/test/srt/models/test_generation_models.py)
file, test the new model on your local machine and report the results on demonstrative benchmarks (GSM8K, MMLU, MMMU,
MMMU-Pro, etc.) in your PR. \\
For VLMs, also include a test in `test_vision_openai_server_{x}.py` (e.g. [test_vision_openai_server_a.py](https://github.com/sgl-project/sglang/blob/main/test/srt/test_vision_openai_server_a.py), [test_vision_openai_server_b.py](https://github.com/sgl-project/sglang/blob/main/test/srt/test_vision_openai_server_b.py)).
```
**File:** docs/supported_models/support_new_models.md (L80-83)
```markdown
### Benchmark

- **(Required) MMMU**: follow MMMU benchmark [README.md](https://github.com/sgl-project/sglang/blob/main/benchmark/mmmu/README.md) to get SGLang vs. HF Transformer accuracy comparison. The accuracy score from SGLang run should not be much lower than that from HF Transformer run. Similarly, follow https://docs.sglang.io/developer_guide/benchmark_and_profiling.html to get performance comparison: TTFT and throughput must meet or exceed baselines (e.g., HF Transformer).
- **(Optional) Other evals**: If you ran other evals, please note the results in PR description.
```
**File:** docs/supported_models/support_new_models.md (L97-99)
```markdown
    - **Replace vLLM’s `Attention` with `RadixAttention`** (ensure you pass `layer_id` to `RadixAttention`).
    - **Replace vLLM’s `LogitsProcessor` with SGLang’s `LogitsProcessor`.**
    - **Replace the multi-headed `Attention` of ViT with SGLang’s `VisionAttention`.**
```
**File:** docs/supported_models/support_new_models.md (L100-101)
```markdown
    - **Replace other vLLM layers** (such as `RMSNorm`, `SiluAndMul`) with SGLang layers.
    - **Remove `Sample`.**
```
**File:** docs/supported_models/support_new_models.md (L103-105)
```markdown
    - **Add `EntryClass`** at the end.
    - **Ensure that the new implementation uses only SGLang components** and does not rely on any vLLM components.

```
**File:** docs/supported_models/support_new_models.md (L106-107)
```markdown
Note: make sure you add your new model to the supported models list in the supported models documentation.

```
**File:** docs/supported_models/support_new_models.md (L110-136)
```markdown
In addition to the methods above, you can register your new model with the `ModelRegistry` before launching the server.
This allows you to integrate your model without modifying the source code.

For example:

```python
from sglang.srt.models.registry import ModelRegistry
from sglang.srt.entrypoints.http_server import launch_server

# For a single model, add it to the registry:
ModelRegistry.models[model_name] = model_class

# For multiple models, you can imitate the import_model_classes() function:
from functools import lru_cache

@lru_cache()
def import_new_model_classes():
    model_arch_name_to_cls = {}
    # Populate model_arch_name_to_cls with your new model classes.
    ...
    return model_arch_name_to_cls

ModelRegistry.models.update(import_new_model_classes())

# Launch the server with your server arguments:
launch_server(server_args)
```
```
**File:** python/sglang/srt/models/qwen2.py (L1-100)
```python
# Copyright 2023-2024 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

# Adapted from llama2.py
# Modify details for the adaptation of Qwen2 model.
"""Inference-only Qwen2 model compatible with HuggingFace weights."""
import logging
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

import torch
from torch import nn

from sglang.srt.distributed import (
    get_pp_group,
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
)
from sglang.srt.layers.activation import SiluAndMul
from sglang.srt.layers.dp_attention import is_dp_attention_enabled
from sglang.srt.layers.layernorm import RMSNorm
from sglang.srt.layers.linear import (
    MergedColumnParallelLinear,
    QKVParallelLinear,
    RowParallelLinear,
)
from sglang.srt.layers.logits_processor import LogitsProcessor
from sglang.srt.layers.pooler import Pooler, PoolingType
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.layers.radix_attention import RadixAttention
from sglang.srt.layers.rotary_embedding import get_rope
from sglang.srt.layers.utils import PPMissingLayer, get_layer_id
from sglang.srt.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, PPProxyTensors
from sglang.srt.model_loader.weight_utils import (
    default_weight_loader,
    kv_cache_scales_loader,
)
from sglang.srt.server_args import get_global_server_args
from sglang.srt.utils import add_prefix, make_layers

Qwen2Config = None


logger = logging.getLogger(__name__)


class Qwen2MLP(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: str,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.gate_up_proj = MergedColumnParallelLinear(
            hidden_size,
            [intermediate_size] * 2,
            bias=False,
            quant_config=quant_config,
            prefix=add_prefix("gate_up_proj", prefix),
        )
        self.down_proj = RowParallelLinear(
            intermediate_size,
            hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=add_prefix("down_proj", prefix),
        )
        if hidden_act != "silu":
            raise ValueError(
                f"Unsupported activation: {hidden_act}. "
                "Only silu is supported for now."
            )
        self.act_fn = SiluAndMul()

    def forward(self, x):
        if get_global_server_args().rl_on_policy_target is not None:
            x = x.bfloat16()

        gate_up, _ = self.gate_up_proj(x)
        x = self.act_fn(gate_up)
        x, _ = self.down_proj(x)
        return x

```
**File:** python/sglang/srt/models/llama.py (L393-449)
```python
class LlamaForCausalLM(nn.Module):
    # BitandBytes specific attributes
    default_bitsandbytes_target_modules = [
        ".gate_proj.",
        ".down_proj.",
        ".up_proj.",
        ".q_proj.",
        ".k_proj.",
        ".v_proj.",
        ".o_proj.",
    ]
    # in TP, these weights are partitioned along the column dimension (dim=-1)
    column_parallel_weights_modules = [".down_proj.", ".o_proj."]
    bitsandbytes_stacked_params_mapping = {
        # shard_name, weight_name, index
        ".q_proj": (".qkv_proj", 0),
        ".k_proj": (".qkv_proj", 1),
        ".v_proj": (".qkv_proj", 2),
        ".gate_proj": (".gate_up_proj", 0),
        ".up_proj": (".gate_up_proj", 1),
    }

    def __init__(
        self,
        config: LlamaConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.pp_group = get_pp_group()
        self.config = config
        self.quant_config = quant_config
        self.model = self._init_model(config, quant_config, add_prefix("model", prefix))
        # Llama 3.2 1B Instruct set tie_word_embeddings to True
        # Llama 3.1 8B Instruct set tie_word_embeddings to False
        if self.config.tie_word_embeddings:
            self.lm_head = self.model.embed_tokens
        else:
            self.lm_head = ParallelLMHead(
                config.vocab_size,
                config.hidden_size,
                quant_config=quant_config,
                prefix=add_prefix("lm_head", prefix),
                use_attn_tp_group=get_global_server_args().enable_dp_lm_head,
            )
        self.logits_processor = LogitsProcessor(config)
        self.pooler = Pooler(pooling_type=PoolingType.LAST, normalize=True)
        self.stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            (".qkv_proj", ".q_proj", "q"),
            (".qkv_proj", ".k_proj", "k"),
            (".qkv_proj", ".v_proj", "v"),
            (".gate_up_proj", ".gate_proj", 0),
            (".gate_up_proj", ".up_proj", 1),
        ]

        self.capture_aux_hidden_states = False
```
**File:** python/sglang/srt/model_loader/utils.py (L84-106)
```python
def get_model_architecture(model_config: ModelConfig) -> Tuple[Type[nn.Module], str]:
    from sglang.srt.models.registry import ModelRegistry

    architectures = getattr(model_config.hf_config, "architectures", [])
    # Special handling for quantized Mixtral.
    # FIXME(woosuk): This is a temporary hack.
    mixtral_supported = ["fp8", "compressed-tensors", "gptq_marlin", "awq_marlin"]

    if (
        model_config.quantization is not None
        and model_config.quantization not in mixtral_supported
        and "MixtralForCausalLM" in architectures
    ):
        architectures = ["QuantMixtralForCausalLM"]

    supported_archs = ModelRegistry.get_supported_archs()
    is_native_supported = any(arch in supported_archs for arch in architectures)

    if model_config.model_impl == ModelImpl.MINDSPORE:
        architectures = ["MindSporeForCausalLM"]
    elif not is_native_supported or model_config.model_impl == ModelImpl.TRANSFORMERS:
        architectures = resolve_transformers_arch(model_config, architectures)
    return ModelRegistry.resolve_model_cls(architectures)
```
**File:** python/sglang/srt/models/qwen2_5_vl.py (L1-80)
```python
# coding=utf-8
# Adapted from
# https://github.com/huggingface/transformers/blob/19e6e80e10118f855137b90740936c0b11ac397f/src/transformers/models/qwen2_vl/modeling_qwen2_vl.py
# Copyright 2024 The Qwen team.
# Copyright 2023 The vLLM team.
# Copyright 2022 EleutherAI and the HuggingFace Inc. team. All rights reserved.
#
# This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
# and OPT implementations in this library. It has been modified from its
# original forms to accommodate minor architectural differences compared
# to GPT-NeoX and OPT used by the Meta AI team that trained the model.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Inference-only Qwen2-VL model compatible with HuggingFace weights."""
import logging
import re
from functools import partial
from typing import Iterable, List, Optional, Tuple, Type

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from transformers.activations import ACT2FN
from transformers.models.qwen2_5_vl.configuration_qwen2_5_vl import (
    Qwen2_5_VLConfig,
    Qwen2_5_VLVisionConfig,
)
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import (
    Qwen2_5_VisionPatchEmbed,
    Qwen2_5_VisionRotaryEmbedding,
)

from sglang.srt.distributed import (
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
)
from sglang.srt.distributed.parallel_state import get_pp_group
from sglang.srt.environ import envs
from sglang.srt.layers.activation import SiluAndMul
from sglang.srt.layers.attention.vision import VisionAttention
from sglang.srt.layers.layernorm import RMSNorm
from sglang.srt.layers.linear import (
    ColumnParallelLinear,
    MergedColumnParallelLinear,
    RowParallelLinear,
)
from sglang.srt.layers.logits_processor import LogitsProcessor
from sglang.srt.layers.pooler import Pooler, PoolingType
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.layers.utils import PPMissingLayer, get_layer_id
from sglang.srt.layers.vocab_parallel_embedding import ParallelLMHead
from sglang.srt.managers.mm_utils import (
    MultiModalityDataPaddingPatternMultimodalTokens,
    general_mm_embed_routine,
)
from sglang.srt.managers.schedule_batch import (
    Modality,
    MultimodalDataItem,
    MultimodalInputs,
)
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, PPProxyTensors
from sglang.srt.model_loader.weight_utils import default_weight_loader
from sglang.srt.models.qwen2 import Qwen2Model
from sglang.srt.models.utils import RotaryPosMixin, WeightsMapper, permute_inv
from sglang.srt.multimodal.mm_utils import run_dp_sharded_mrope_vision_model
from sglang.srt.multimodal.vit_cuda_graph_runner import ViTCudaGraphRunner
from sglang.srt.server_args import get_global_server_args
from sglang.srt.utils import add_prefix

```
**File:** python/sglang/srt/configs/model_config.py (L373-451)
```python
        # FIXME: temporary special judge for MLA architecture
        if (
            "DeepseekV2ForCausalLM" in self.hf_config.architectures
            or "DeepseekV32ForCausalLM" in self.hf_config.architectures
            or "DeepseekV3ForCausalLM" in self.hf_config.architectures
            or "DeepseekV3ForCausalLMNextN" in self.hf_config.architectures
            or "LongcatFlashForCausalLM" in self.hf_config.architectures
            or "LongcatFlashForCausalLMNextN" in self.hf_config.architectures
            or "DotsVLMForCausalLM" in self.hf_config.architectures
            or "MistralLarge3ForCausalLM" in self.hf_config.architectures
            or "PixtralForConditionalGeneration" in self.hf_config.architectures
            or "MistralLarge3ForCausalLMEagle" in self.hf_config.architectures
        ):
            self.head_dim = 256
            self.attention_arch = AttentionArch.MLA
            self.kv_lora_rank = self.hf_config.kv_lora_rank
            self.qk_nope_head_dim = self.hf_config.qk_nope_head_dim
            self.qk_rope_head_dim = self.hf_config.qk_rope_head_dim
            self.v_head_dim = self.hf_config.v_head_dim
            self.index_head_dim = (
                get_nsa_index_head_dim(self.hf_config)
                if is_deepseek_nsa(self.hf_config)
                else None
            )

            # Handle rope scaling with yarn
            self.scaling = 1 / math.sqrt(self.qk_nope_head_dim + self.qk_rope_head_dim)
            if self.hf_config.rope_scaling:
                mscale_all_dim = self.hf_config.rope_scaling.get(
                    "mscale_all_dim", False
                )
                scaling_factor = self.hf_config.rope_scaling["factor"]
                mscale = yarn_get_mscale(scaling_factor, float(mscale_all_dim))
                self.scaling = self.scaling * mscale * mscale

        elif "MiniCPM3ForCausalLM" in self.hf_config.architectures:
            self.head_dim = 128
            self.attention_arch = AttentionArch.MLA
            self.kv_lora_rank = self.hf_config.kv_lora_rank
            self.qk_rope_head_dim = self.hf_config.qk_rope_head_dim
        elif "DeepseekVL2ForCausalLM" in self.hf_config.architectures and getattr(
            self.hf_text_config, "use_mla", True
        ):
            self.head_dim = 256
            self.attention_arch = AttentionArch.MLA
            self.kv_lora_rank = self.hf_text_config.kv_lora_rank
            self.qk_rope_head_dim = self.hf_text_config.qk_rope_head_dim
        elif "KimiVLForConditionalGeneration" in self.hf_config.architectures:
            self.head_dim = 256
            self.attention_arch = AttentionArch.MLA
            self.kv_lora_rank = self.hf_text_config.kv_lora_rank
            self.qk_rope_head_dim = self.hf_text_config.qk_rope_head_dim
            self.v_head_dim = self.hf_text_config.v_head_dim
            self.qk_nope_head_dim = self.hf_text_config.qk_nope_head_dim
        elif "KimiLinearForCausalLM" in self.hf_config.architectures:
            self.head_dim = 72
            self.attention_arch = AttentionArch.MLA
            self.kv_lora_rank = self.hf_config.kv_lora_rank
            self.qk_rope_head_dim = self.hf_config.qk_rope_head_dim
            self.v_head_dim = self.hf_config.v_head_dim
            self.qk_nope_head_dim = self.hf_config.qk_nope_head_dim
        else:
            if (
                "MistralModel" in self.hf_config.architectures
                or "MixtralForCausalLM" in self.hf_config.architectures
                or "MistralForCausalLM" in self.hf_config.architectures
            ):
                if getattr(self, "head_dim", None) is None:
                    self.head_dim = (
                        self.hf_config.hidden_size // self.hf_config.num_attention_heads
                    )
                    # In transformers==4.52.3, the head_dim is null in MistralConfig
                    if (
                        not hasattr(self.hf_text_config, "head_dim")
                        or self.hf_text_config.head_dim is None
                    ):
                        setattr(self.hf_text_config, "head_dim", self.head_dim)

            self.attention_arch = AttentionArch.MHA
```
**File:** docs/platforms/ascend_npu_support_models.md (L31-97)
```markdown
| ZhipuAI/chatglm2-6b                       | ChatGLM                        |  **<span style="color: red;">×</span>**  | **<span style="color: green;">√</span>** |
| Shanghai_AI_Laboratory/internlm2-7b       | InternLM 2                     | **<span style="color: green;">√</span>** | **<span style="color: green;">√</span>** |
| LGAI-EXAONE/EXAONE-3.5-7.8B-Instruct      | ExaONE 3                       | **<span style="color: green;">√</span>** | **<span style="color: green;">√</span>** |
| xverse/XVERSE-MoE-A36B                    | XVERSE                         | **<span style="color: green;">√</span>** | **<span style="color: green;">√</span>** |
| HuggingFaceTB/SmolLM-1.7B                 | SmolLM                         | **<span style="color: green;">√</span>** | **<span style="color: green;">√</span>** |
| ZhipuAI/glm-4-9b-chat                     | GLM-4                          | **<span style="color: green;">√</span>** | **<span style="color: green;">√</span>** |
| XiaomiMiMo/MiMo-7B-RL                     | MiMo                           | **<span style="color: green;">√</span>** | **<span style="color: green;">√</span>** |
| arcee-ai/AFM-4.5B-Base                    | Arcee AFM-4.5B                 | **<span style="color: green;">√</span>** | **<span style="color: green;">√</span>** |
| Howeee/persimmon-8b-chat                  | Persimmon                      | **<span style="color: green;">√</span>** | **<span style="color: green;">√</span>** |
| inclusionAI/Ling-lite                     | Ling                           | **<span style="color: green;">√</span>** | **<span style="color: green;">√</span>** |
| ibm-granite/granite-3.1-8b-instruct       | Granite                        | **<span style="color: green;">√</span>** | **<span style="color: green;">√</span>** |
| ibm-granite/granite-3.0-3b-a800m-instruct | Granite MoE                    | **<span style="color: green;">√</span>** | **<span style="color: green;">√</span>** |
| databricks/dbrx-instruct                  | DBRX (Databricks)              |  **<span style="color: red;">×</span>**  |  **<span style="color: red;">×</span>**  |
| baichuan-inc/Baichuan2-13B-Chat           | Baichuan 2 (7B, 13B)           |  **<span style="color: red;">×</span>**  |  **<span style="color: red;">×</span>**  |
| baidu/ERNIE-4.5-21B-A3B-PT                | ERNIE-4.5 (4.5, 4.5MoE series) |  **<span style="color: red;">×</span>**  |  **<span style="color: red;">×</span>**  |
| openbmb/MiniCPM3-4B                       | MiniCPM (v3, 4B)               |  **<span style="color: red;">×</span>**  |  **<span style="color: red;">×</span>**  |
| openai/gpt-oss-120b                       | GPTOSS                         |  **<span style="color: red;">×</span>**  |  **<span style="color: red;">×</span>**  |

## Multimodal Language Models

| Models                                        | Model Family (Variants)   | A2 Supported                             |               A3 Supported               |
|-----------------------------------------------|---------------------------|------------------------------------------|:----------------------------------------:|
| Qwen2.5-VL-72B-Instruct-w8a8                  | Qwen-VL                   | **<span style="color: red;">×</span>**   | **<span style="color: green;">√</span>** |
| Qwen3-VL-30B-A3B-Instruct                     | Qwen-VL                   | **<span style="color: red;">×</span>**   | **<span style="color: green;">√</span>** |
| Qwen3-VL-8B-Instruct                          | Qwen-VL                   | **<span style="color: red;">×</span>**   | **<span style="color: green;">√</span>** |
| Qwen3-VL-4B-Instruct                          | Qwen-VL                   | **<span style="color: red;">×</span>**   | **<span style="color: green;">√</span>** |
| Qwen3-VL-235B-A22B-Instruct                   | Qwen-VL                   | **<span style="color: red;">×</span>**   | **<span style="color: green;">√</span>** |
| deepseek-ai/deepseek-vl2                      | DeepSeek-VL2              | **<span style="color: red;">×</span>**   | **<span style="color: green;">√</span>** |
| deepseek-ai/Janus-Pro-7B                      | Janus-Pro (1B, 7B)        | **<span style="color: green;">√</span>** | **<span style="color: green;">√</span>** |
| openbmb/MiniCPM-V-2_6                         | MiniCPM-V / MiniCPM-o     | **<span style="color: red;">×</span>**   | **<span style="color: green;">√</span>** |
| google/gemma-3-4b-it                          | Gemma 3 (Multimodal)      | **<span style="color: green;">√</span>** | **<span style="color: green;">√</span>** |
| mistralai/Mistral-Small-3.1-24B-Instruct-2503 | Mistral-Small-3.1-24B     | **<span style="color: red;">×</span>**   | **<span style="color: green;">√</span>** |
| microsoft/Phi-4-multimodal-instruct           | Phi-4-multimodal-instruct | **<span style="color: red;">×</span>**   | **<span style="color: green;">√</span>** |
| XiaomiMiMo/MiMo-VL-7B-RL                      | MiMo-VL (7B)              | **<span style="color: red;">×</span>**   | **<span style="color: green;">√</span>** |
| AI-ModelScope/llava-v1.6-34b                  | LLaVA (v1.5 & v1.6)       | **<span style="color: green;">√</span>** | **<span style="color: green;">√</span>** |
| lmms-lab/llava-next-72b                       | LLaVA-NeXT (8B, 72B)      | **<span style="color: green;">√</span>** | **<span style="color: green;">√</span>** |
| lmms-lab/llava-onevision-qwen2-7b-ov          | LLaVA-OneVision           | **<span style="color: red;">×</span>**   | **<span style="color: green;">√</span>** |
| Kimi/Kimi-VL-A3B-Instruct                     | Kimi-VL (A3B)             | **<span style="color: red;">×</span>**   | **<span style="color: green;">√</span>** |
| ZhipuAI/GLM-4.5V                              | GLM-4.5V (106B)           | **<span style="color: red;">×</span>**   | **<span style="color: green;">√</span>** |
| meta-llama/Llama-3.2-11B-Vision-Instruct      | Llama 3.2 Vision (11B)    | **<span style="color: red;">×</span>**   |  **<span style="color: red;">×</span>**  |

## Embedding Models

| Models                                    | Model Family             | A2 Supported                             |               A3 Supported               |
|-------------------------------------------|--------------------------|------------------------------------------|:----------------------------------------:|
| 	intfloat/e5-mistral-7b-instruct          | E5 (Llama/Mistral based) | **<span style="color: red;">×</span>**   | **<span style="color: green;">√</span>** |
| 	iic/gte_Qwen2-1.5B-instruct              | GTE-Qwen2                | **<span style="color: red;">×</span>**   | **<span style="color: green;">√</span>** |
| 	Qwen/Qwen3-Embedding-8B                  | Qwen3-Embedding          | **<span style="color: red;">×</span>**   |  **<span style="color: red;">×</span>**  |
| 	Alibaba-NLP/gme-Qwen2-VL-2B-Instruct     | GME (Multimodal)         | **<span style="color: red;">×</span>**   | **<span style="color: green;">√</span>** |
| 	AI-ModelScope/clip-vit-large-patch14-336 | CLIP                     | **<span style="color: green;">√</span>** | **<span style="color: green;">√</span>** |
| 	BAAI/bge-large-en-v1.5                   | BGE                      | **<span style="color: red;">×</span>**   |  **<span style="color: red;">×</span>**  |

## Reward Models

| Models                                      | Model Family              | A2 Supported                           |               A3 Supported               |
|---------------------------------------------|---------------------------|----------------------------------------|:----------------------------------------:|
| 	Skywork/Skywork-Reward-Llama-3.1-8B-v0.2   | Llama3.1 Reward           | **<span style="color: red;">×</span>** | **<span style="color: green;">√</span>** |
| 	Shanghai_AI_Laboratory/internlm2-7b-reward | InternLM 2 Reward         | **<span style="color: red;">×</span>** | **<span style="color: green;">√</span>** |
| 	Qwen/Qwen2.5-Math-RM-72B                   | Qwen2.5 Reward - Math     | **<span style="color: red;">×</span>** | **<span style="color: green;">√</span>** |
| 	jason9693/Qwen2.5-1.5B-apeach              | Qwen2.5 Reward - Sequence | **<span style="color: red;">×</span>** | **<span style="color: green;">√</span>** |
| 	Skywork/Skywork-Reward-Gemma-2-27B-v0.2    | Gemma 2-27B Reward        | **<span style="color: red;">×</span>** |  **<span style="color: red;">×</span>**  |

## Rerank Models

| Models                  | Model Family |              A2 Supported              |              A3 Supported              |
|-------------------------|--------------|:--------------------------------------:|:--------------------------------------:|
```
**File:** python/sglang/srt/server_args.py (L443-467)
```python
    # Expert parallelism
    ep_size: int = 1
    moe_a2a_backend: Literal["none", "deepep", "mooncake", "ascend_fuseep"] = "none"
    moe_runner_backend: str = "auto"
    flashinfer_mxfp4_moe_precision: Literal["default", "bf16"] = "default"
    enable_flashinfer_allreduce_fusion: bool = False
    deepep_mode: Literal["auto", "normal", "low_latency"] = "auto"
    ep_num_redundant_experts: int = 0
    ep_dispatch_algorithm: Optional[Literal["static", "dynamic", "fake"]] = None
    init_expert_location: str = "trivial"
    enable_eplb: bool = False
    eplb_algorithm: str = "auto"
    eplb_rebalance_num_iterations: int = 1000
    eplb_rebalance_layers_per_chunk: Optional[int] = None
    eplb_min_rebalancing_utilization_threshold: float = 1.0
    expert_distribution_recorder_mode: Optional[
        Literal["stat", "stat_approx", "per_pass", "per_token"]
    ] = None
    expert_distribution_recorder_buffer_size: Optional[int] = None
    enable_expert_distribution_metrics: bool = False
    deepep_config: Optional[str] = None
    moe_dense_tp_size: Optional[int] = None
    elastic_ep_backend: Literal[None, "mooncake"] = None
    mooncake_ib_device: Optional[str] = None

```
**File:** python/sglang/srt/model_executor/model_runner_kv_cache_mixin.py (L47-110)
```python
class ModelRunnerKVCacheMixin:
    def get_cell_size_per_token(self: ModelRunner, num_layers: int) -> int:
        kv_size = torch._utils._element_size(self.kv_cache_dtype)
        if self.use_mla_backend:
            cell_size = (
                (self.model_config.kv_lora_rank + self.model_config.qk_rope_head_dim)
                * num_layers
                * kv_size
            )
            if is_float4_e2m1fn_x2(self.kv_cache_dtype):
                # kv_scale_buffer
                scale_block_size = 16
                cell_size = (cell_size // 2) + (
                    (
                        (
                            self.model_config.kv_lora_rank
                            + self.model_config.qk_rope_head_dim
                        )
                        // scale_block_size
                    )
                    * num_layers
                    * kv_size
                )

            # Add indexer KV cache overhead for NSA models (DeepSeek V3.2)
            if is_deepseek_nsa(self.model_config.hf_config):
                index_head_dim = get_nsa_index_head_dim(self.model_config.hf_config)
                indexer_size_per_token = (
                    index_head_dim
                    + index_head_dim // NSATokenToKVPool.quant_block_size * 4
                )
                element_size = torch._utils._element_size(
                    NSATokenToKVPool.index_k_with_scale_buffer_dtype
                )
                cell_size += indexer_size_per_token * num_layers * element_size
        else:
            cell_size = (
                self.model_config.get_num_kv_heads(get_attention_tp_size())
                * (self.model_config.head_dim + self.model_config.v_head_dim)
                * num_layers
                * kv_size
            )

            if is_float4_e2m1fn_x2(self.kv_cache_dtype):
                # kv_scale_buffer
                scale_block_size = 16

                n = self.model_config.get_num_kv_heads(get_attention_tp_size())
                k = self.model_config.head_dim
                cell_size = (cell_size // 2) + (
                    (n * k * num_layers * 2 * kv_size) // scale_block_size
                )

            if "MiMoV2FlashForCausalLM" in self.model_config.hf_config.architectures:
                cell_size += (
                    self.model_config.get_swa_num_kv_heads(get_attention_tp_size())
                    * (
                        self.model_config.hf_text_config.swa_head_dim
                        + self.model_config.hf_text_config.swa_v_head_dim
                    )
                    * len(self.model_config.swa_attention_layer_ids)
                    * kv_size
                )
        return cell_size
```
## Q2
我能在不修改sglang源码的条件下，将我的新模型包装为一个插件module来实现嘛
### Answer
是的，您完全可以在不修改 SGLang 源码的条件下，将新模型包装为插件模块来实现。SGLang 提供了 `ModelRegistry` 机制来支持外部模型注册。

## 插件实现方式

### 1. 直接注册单个模型

最简单的方式是直接操作 `ModelRegistry.models` 字典：

```python
from sglang.srt.models.registry import ModelRegistry
from your_plugin_module import YourNewModel

# 注册单个模型
ModelRegistry.models["YourModelName"] = YourNewModel
``` sglang:115-120 

### 2. 批量注册多个模型

对于多个模型，可以创建一个导入函数：

```python
from functools import lru_cache
from sglang.srt.models.registry import ModelRegistry

@lru_cache()
def import_new_model_classes():
    model_arch_name_to_cls = {}
    # 填充您的模型类
    model_arch_name_to_cls["YourModel1"] = YourModel1Class
    model_arch_name_to_cls["YourModel2"] = YourModel2Class
    return model_arch_name_to_cls

ModelRegistry.models.update(import_new_model_classes())
``` sglang:122-132 

### 3. 通过环境变量加载外部包

SGLang 支持通过 `SGLANG_EXTERNAL_MODEL_PACKAGE` 环境变量自动加载外部模型包：

```bash
export SGLANG_EXTERNAL_MODEL_PACKAGE="your_plugin_package"
``` sglang:131-132 

## 完整插件示例

### 插件模块结构

创建一个独立的 Python 包，例如 `my_sglang_models/`：

```
my_sglang_models/
├── __init__.py
├── model1.py
├── model2.py
└── setup.py
```

### 模型实现示例

在 `model1.py` 中实现您的模型：

```python
import torch
from sglang.srt.models.llama import LlamaForCausalLM
from sglang.srt.layers.logits_processor import LogitsProcessorOutput

class MyCustomModel(LlamaForCausalLM):
    def __init__(self, config, quant_config=None, prefix=""):
        super().__init__(config=config, quant_config=quant_config, prefix=prefix)
    
    def forward(self, input_ids, positions, ...):
        # 调用父类方法
        res = super().forward(input_ids, positions, ...)
        
        # 自定义处理
        orig_logits = res.next_token_logits
        res.next_token_logits = torch.where(
            orig_logits > 0,
            orig_logits.sqrt(),
            orig_logits
        )
        
        return res

# 注册入口
EntryClass = [MyCustomModel]
``` sglang:166-225 

### 使用插件

```python
import asyncio
from functools import lru_cache
from sglang.srt.models.registry import ModelRegistry
import sglang as sgl

# 导入您的模型
from my_sglang_models.model1 import MyCustomModel

# 注册模型
@lru_cache()
def import_new_model_classes():
    return {"MyCustomModel": MyCustomModel}

ModelRegistry.models.update(import_new_model_classes())

# 使用模型
async def main():
    llm = sgl.Engine(model_path="./path/to/your/model")
    sampling_params = {"temperature": 0.2}
    outputs = await llm.async_generate(["Hello"], sampling_params)
    print(outputs)
    llm.shutdown()

if __name__ == "__main__":
    asyncio.run(main())
``` sglang:270-309 

## ModelRegistry 工作原理

`ModelRegistry` 是一个单例，负责管理所有可用的模型类：

```python
@dataclass
class _ModelRegistry:
    models: Dict[str, Union[Type[nn.Module], str]] = field(default_factory=dict)
    
    def resolve_model_cls(self, architectures):
        # 解析并返回对应的模型类
        for arch in architectures:
            model_cls = self._try_load_model_cls(arch)
            if model_cls is not None:
                return (model_cls, arch)
``` sglang:17-89 

## 注意事项

1. **模型配置**：确保模型的 `config.json` 中的 `architectures` 字段指向您注册的模型名称
2. **依赖管理**：插件包需要独立管理其依赖
3. **版本兼容**：确保插件与 SGLang 版本兼容
4. **测试**：建议在插件包中包含单元测试

通过这种方式，您可以完全独立地开发和分发模型插件，无需修改 SGLang 源码。

## Notes

- 插件模式特别适合企业内部模型或需要保密的模型实现
- SGLang 的 ModelRegistry 使用 `@lru_cache` 来优化模型类的导入 sglang:92-93 
- 对于多模态模型，同样可以通过插件方式实现，需要额外实现相应的处理器和视觉组件

Wiki pages you might want to explore:
- [Installation and Platform Support (sgl-project/sglang)](/wiki/sgl-project/sglang#2)
### Citations
**File:** docs/supported_models/support_new_models.md (L115-120)
```markdown
```python
from sglang.srt.models.registry import ModelRegistry
from sglang.srt.entrypoints.http_server import launch_server

# For a single model, add it to the registry:
ModelRegistry.models[model_name] = model_class
```
**File:** docs/supported_models/support_new_models.md (L122-132)
```markdown
# For multiple models, you can imitate the import_model_classes() function:
from functools import lru_cache

@lru_cache()
def import_new_model_classes():
    model_arch_name_to_cls = {}
    # Populate model_arch_name_to_cls with your new model classes.
    ...
    return model_arch_name_to_cls

ModelRegistry.models.update(import_new_model_classes())
```
**File:** docs/supported_models/support_new_models.md (L166-225)
```markdown
```python
class LlamaWrapper(LlamaForCausalLM):
    def __init__(
        self,
        config: LlamaConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__(config=config, quant_config=quant_config, prefix=prefix)
```

Now, we want to define the `forward` method, which is what will be called at inference time.
Note that the signature for `forward` is essentially the same for any model; you can take a look at the other models defined in the [`models` directory](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/models/) for references.
To see where exactly `forward` is called in the SGLang runtime's internals, take a look at [`forward_decode`](https://github.com/sgl-project/sglang/blob/bf72b80122fd888bf619d17b96fa3e323ab809fc/python/sglang/srt/model_executor/model_runner.py#L1705) and [`forward_extend`](https://github.com/sgl-project/sglang/blob/bf72b80122fd888bf619d17b96fa3e323ab809fc/python/sglang/srt/model_executor/model_runner.py#L1724) in the [`ModelRunner` class](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/model_executor/model_runner.py).

```python
    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        pp_proxy_tensors: Optional[PPProxyTensors] = None,
        input_embeds: Optional[torch.Tensor] = None,
        get_embedding: bool = False,
    ) -> LogitsProcessorOutput:
```

We now call the `__call__` method for `self.model` (which is a member variable that `LlamaForCausalLM` defines in its `__init__` method), which eventually calls `LlamaForCausalLM`'s `forward` method.
After that, we feed the `hidden_states` into our model's `LogitsProcessor` (again defined in `LlamaForCausalLM`).

```python
        hidden_states = self.model(
            input_ids,
            positions,
            forward_batch,
            input_embeds,
            pp_proxy_tensors=pp_proxy_tensors,
        )

        res: LogitsProcessorOutput = self.logits_processor(
            input_ids,
            hidden_states,
            self.lm_head,
            forward_batch,
        )
```

After receiving the logits for the next token, we can finally perform our biasing step.

```python
        orig_logits = res.next_token_logits
        res.next_token_logits = torch.where(
            orig_logits > 0,
            orig_logits.sqrt(),
            orig_logits
        )

        return res
```
```
**File:** docs/supported_models/support_new_models.md (L270-309)
```markdown
```python
@lru_cache()
def import_new_model_classes():
    model_arch_name_to_cls = {"LlamaWrapper": LlamaWrapper}
    return model_arch_name_to_cls

ModelRegistry.models.update(import_new_model_classes())
```

Lastly, when we create our `Engine`, we just pass in the path to the local model directory.
Then, our `LlamaWrapper` is ready to be served; for this walkthrough, we will use SGLang `Engine`'s non-streaming asynchronous generation endpoint.

```python
def main():
    llm = sgl.Engine(model_path="./llama_ckpt")
    sampling_params = {"temperature": 0.2, "top_k": 5}
    prompts = [
        "Write a short, neutral self-introduction for a fictional character. Hello, my name is",
        "Provide a concise factual statement about France’s capital city. The capital of France is",
        "Explain possible future trends in artificial intelligence. The future of AI is",
    ]

    asyncio.run(run_llm(llm, sampling_params, prompts))

    llm.shutdown()

async def run_llm(
    llm,
    sampling_params,
    prompts,
) -> None:
    outputs = await llm.async_generate(prompts, sampling_params)

    for prompt, output in zip(prompts, outputs):
        print(f"\nPrompt: {prompt}")
        print(f"Generated text: {output['text']}")

if __name__ == "__main__":
    main()
```
```
**File:** python/sglang/srt/models/registry.py (L17-89)
```python
@dataclass
class _ModelRegistry:
    # Keyed by model_arch
    models: Dict[str, Union[Type[nn.Module], str]] = field(default_factory=dict)

    def register(
        self, package_name: str, overwrite: bool = False, strict: bool = False
    ):
        new_models = import_model_classes(package_name, strict=strict)
        if overwrite:
            self.models.update(new_models)
        else:
            for arch, cls in new_models.items():
                if arch in self.models:
                    raise ValueError(
                        f"Model architecture {arch} already registered. Set overwrite=True to replace."
                    )
                self.models[arch] = cls

    def get_supported_archs(self) -> AbstractSet[str]:
        return self.models.keys()

    def _raise_for_unsupported(self, architectures: List[str]):
        all_supported_archs = self.get_supported_archs()

        if any(arch in all_supported_archs for arch in architectures):
            raise ValueError(
                f"Model architectures {architectures} failed "
                "to be inspected. Please check the logs for more details."
            )

        raise ValueError(
            f"Model architectures {architectures} are not supported for now. "
            f"Supported architectures: {all_supported_archs}"
        )

    def _try_load_model_cls(self, model_arch: str) -> Optional[Type[nn.Module]]:
        if model_arch not in self.models:
            return None

        return self.models[model_arch]

    def _normalize_archs(
        self,
        architectures: Union[str, List[str]],
    ) -> List[str]:
        if isinstance(architectures, str):
            architectures = [architectures]
        if not architectures:
            logger.warning("No model architectures are specified")

        # filter out support architectures
        normalized_arch = list(
            filter(lambda model: model in self.models, architectures)
        )

        # make sure Transformers backend is put at the last as a fallback
        if len(normalized_arch) != len(architectures):
            normalized_arch.append("TransformersForCausalLM")
        return normalized_arch

    def resolve_model_cls(
        self,
        architectures: Union[str, List[str]],
    ) -> Tuple[Type[nn.Module], str]:
        architectures = self._normalize_archs(architectures)

        for arch in architectures:
            model_cls = self._try_load_model_cls(arch)
            if model_cls is not None:
                return (model_cls, arch)

        return self._raise_for_unsupported(architectures)
```
**File:** python/sglang/srt/models/registry.py (L92-93)
```python
@lru_cache()
def import_model_classes(package_name: str, strict: bool = False):
```
**File:** python/sglang/srt/models/registry.py (L131-132)
```python
if external_pkg := envs.SGLANG_EXTERNAL_MODEL_PACKAGE.get():
    ModelRegistry.register(external_pkg, overwrite=True)
```
## Q3
sglang 当前支持 qwen3 next 嘛，主要涉及哪些文件
### Answer
是的，SGLang 当前支持 Qwen3 Next 模型。这是一个混合架构（Mamba + Transformer）的模型，SGLang 提供了完整的实现。

## 主要涉及的文件

### 1. 核心模型实现
- **`python/sglang/srt/models/qwen3_next.py`** - Qwen3Next 的主要模型实现，包含 `Qwen3NextModel` 和 `Qwen3NextForCausalLM` 类 sglang:1-53 sglang:776-854 
- **`python/sglang/srt/configs/qwen3_next.py`** - Qwen3Next 的配置类，定义模型架构参数 sglang:32-169 

### 2. 服务器配置支持
- **`python/sglang/srt/server_args.py`** - 在 `_handle_model_specific_adjustments()` 方法中有针对 Qwen3Next 的特殊配置处理，包括量化、MoE 后端和注意力后端的自动选择 sglang:1425-1483 

### 3. 文档和测试
- **`docs/basic_usage/qwen3.md`** - Qwen3-Next 的使用指南，包含启动命令和配置提示 sglang:1-39 
- **`test/srt/models/test_qwen3_next_models.py`** - Qwen3Next 的测试用例 sglang:39-124 

### 4. 相关支持文件
- **`python/sglang/srt/models/qwen3_next_mtp.py`** - Qwen3Next MTP（Multi-Token Prediction）推测解码支持 sglang:1-32 

## 关键特性

1. **混合架构支持** - 实现了 Mamba 和 Transformer 层的混合
2. **MambaRadixCache** - 支持 Mamba 层的前缀缓存
3. **EAGLE 推测解码** - 支持 EAGLE 算法加速推理
4. **专家并行** - 支持 MoE（Mixture of Experts）架构

## 使用示例

```bash
python3 -m sglang.launch_server \
  --model Qwen/Qwen3-Next-80B-A3B-Instruct \
  --tp 4
```

## Notes

- Qwen3 Next 在 SM100 架构上会自动使用 `flashinfer_trtllm` 作为 MoE 后端 sglang:1435-1443 
- 支持 `--mamba-scheduler-strategy extra_buffer` 来启用优化的 Mamba 缓存策略 sglang:18-22 

Wiki pages you might want to explore:
- [Programming Interfaces (sgl-project/sglang)](/wiki/sgl-project/sglang#11)
### Citations
**File:** python/sglang/srt/models/qwen3_next.py (L1-53)
```python
import enum
import logging
from typing import Any, Iterable, Optional, Set, Tuple

import torch
from torch import nn

from sglang.srt.compilation.piecewise_context_manager import get_forward_context
from sglang.srt.configs.qwen3_next import Qwen3NextConfig
from sglang.srt.distributed import get_pp_group
from sglang.srt.eplb.expert_distribution import get_global_expert_distribution_recorder
from sglang.srt.eplb.expert_location import ModelConfigForExpertLocation
from sglang.srt.layers.attention.fla.layernorm_gated import RMSNorm as RMSNormGated
from sglang.srt.layers.attention.mamba.mamba import mamba_v2_sharded_weight_loader
from sglang.srt.layers.communicator import LayerCommunicator, LayerScatterModes
from sglang.srt.layers.dp_attention import (
    get_attention_tp_rank,
    get_attention_tp_size,
    is_dp_attention_enabled,
)
from sglang.srt.layers.layernorm import GemmaRMSNorm
from sglang.srt.layers.linear import (
    ColumnParallelLinear,
    QKVParallelLinear,
    RowParallelLinear,
)
from sglang.srt.layers.logits_processor import LogitsProcessor
from sglang.srt.layers.moe.fused_moe_triton.layer import FusedMoE
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.layers.radix_attention import RadixAttention
from sglang.srt.layers.rotary_embedding import get_rope
from sglang.srt.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from sglang.srt.model_executor.cuda_graph_runner import get_is_capture_mode
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.model_loader.weight_utils import (
    default_weight_loader,
    sharded_weight_loader,
)
from sglang.srt.models.qwen2_moe import Qwen2MoeMLP, Qwen2MoeSparseMoeBlock
from sglang.srt.server_args import get_global_server_args
from sglang.srt.utils import (
    LazyValue,
    add_prefix,
    is_cuda,
    is_npu,
    make_layers,
    set_weight_attrs,
)
from sglang.srt.utils.custom_op import register_custom_op

```
**File:** python/sglang/srt/models/qwen3_next.py (L776-854)
```python
ALL_DECODER_LAYER_TYPES = {
    "attention": Qwen3HybridAttentionDecoderLayer,
    "linear_attention": Qwen3HybridLinearDecoderLayer,
}


class Qwen3NextModel(nn.Module):
    def __init__(
        self,
        config: Qwen3NextConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.config = config

        alt_stream = torch.cuda.Stream() if _is_cuda else None

        self.embed_tokens = VocabParallelEmbedding(
            config.vocab_size,
            config.hidden_size,
            org_num_embeddings=config.vocab_size,
            enable_tp=not is_dp_attention_enabled(),
        )

        def get_layer(idx: int, prefix: str):
            layer_class = ALL_DECODER_LAYER_TYPES[config.layers_block_type[idx]]
            return layer_class(
                config,
                idx,
                quant_config=quant_config,
                prefix=prefix,
                alt_stream=alt_stream,
            )

        self.layers = make_layers(
            config.num_hidden_layers, get_layer, prefix=f"{prefix}.layers"
        )

        self.norm = GemmaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.infer_count = 0

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        # mamba_cache_params: MambaCacheParams,
        inputs_embeds: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:

        # pass a sequence index tensor, that is required for
        # proper continuous batching computation including
        # chunked prefill
        if inputs_embeds is not None:
            hidden_states = inputs_embeds
        else:
            hidden_states = self.embed_tokens(input_ids)

        residual = None
        for i in range(len(self.layers)):
            layer = self.layers[i]
            with get_global_expert_distribution_recorder().with_current_layer(i):
                hidden_states, residual = layer(
                    layer_id=i,
                    positions=positions,
                    hidden_states=hidden_states,
                    residual=residual,
                    forward_batch=forward_batch,
                )

        if not forward_batch.forward_mode.is_idle():
            if residual is None:
                hidden_states = self.norm(hidden_states)
            else:
                hidden_states, _ = self.norm(hidden_states, residual)

        return hidden_states

```
**File:** python/sglang/srt/configs/qwen3_next.py (L32-169)
```python
class Qwen3NextConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`Qwen3NextModel`]. It is used to instantiate a
    Qwen3-Next model according to the specified arguments, defining the model architecture.
    Instantiating a configuration with the defaults will yield a similar configuration to that of
    Qwen3-Next-80B-A3B-Instruct [Qwen/Qwen3-Next-80B-A3B-Instruct](https://huggingface.co/Qwen/Qwen3-Next-80B-A3B-Instruct).

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.


    Args:
        vocab_size (`int`, *optional*, defaults to 151936):
            Vocabulary size of the model. Defines the number of different tokens that can be represented by the
            `inputs_ids`.
        hidden_size (`int`, *optional*, defaults to 2048):
            Dimension of the hidden representations.
        intermediate_size (`int`, *optional*, defaults to 5632):
            Dimension of the MLP representations.
        num_hidden_layers (`int`, *optional*, defaults to 48):
            Number of hidden layers in the Transformer encoder.
        num_attention_heads (`int`, *optional*, defaults to 16):
            Number of attention heads for each attention layer in the Transformer encoder.
        num_key_value_heads (`int`, *optional*, defaults to 2):
            This is the number of key_value heads that should be used to implement Grouped Query Attention. If
            `num_key_value_heads=num_attention_heads`, the model will use Multi Head Attention (MHA), if
            `num_key_value_heads=1` the model will use Multi Query Attention (MQA) otherwise GQA is used. When
            converting a multi-head checkpoint to a GQA checkpoint, each group key and value head should be constructed
            by meanpooling all the original heads within that group. For more details checkout [this
            paper](https://arxiv.org/pdf/2305.13245.pdf). If it is not specified, will default to `32`.
        hidden_act (`str`, *optional*, defaults to `"silu"`):
            The non-linear activation function in the decoder.
        max_position_embeddings (`int`, *optional*, defaults to 32768):
            The maximum sequence length that this model might ever be used with.
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        rms_norm_eps (`float`, *optional*, defaults to 1e-06):
            The epsilon used by the rms normalization layers.
        use_cache (`bool`, *optional*, defaults to `True`):
            Whether or not the model should return the last key/values attentions (not used by all models). Only
            relevant if `config.is_decoder=True`.
        tie_word_embeddings (`bool`, *optional*, defaults to `False`):
            Whether the model's input and output word embeddings should be tied.
        rope_theta (`float`, *optional*, defaults to 10000.0):
            The base period of the RoPE embeddings.
        rope_scaling (`Dict`, *optional*):
            Dictionary containing the scaling configuration for the RoPE embeddings. NOTE: if you apply new rope type
            and you expect the model to work on longer `max_position_embeddings`, we recommend you to update this value
            accordingly.
            Expected contents:
                `rope_type` (`str`):
                    The sub-variant of RoPE to use. Can be one of ['default', 'linear', 'dynamic', 'yarn', 'longrope',
                    'llama3'], with 'default' being the original RoPE implementation.
                `factor` (`float`, *optional*):
                    Used with all rope types except 'default'. The scaling factor to apply to the RoPE embeddings. In
                    most scaling types, a `factor` of x will enable the model to handle sequences of length x *
                    original maximum pre-trained length.
                `original_max_position_embeddings` (`int`, *optional*):
                    Used with 'dynamic', 'longrope' and 'llama3'. The original max position embeddings used during
                    pretraining.
                `attention_factor` (`float`, *optional*):
                    Used with 'yarn' and 'longrope'. The scaling factor to be applied on the attention
                    computation. If unspecified, it defaults to value recommended by the implementation, using the
                    `factor` field to infer the suggested value.
                `beta_fast` (`float`, *optional*):
                    Only used with 'yarn'. Parameter to set the boundary for extrapolation (only) in the linear
                    ramp function. If unspecified, it defaults to 32.
                `beta_slow` (`float`, *optional*):
                    Only used with 'yarn'. Parameter to set the boundary for interpolation (only) in the linear
                    ramp function. If unspecified, it defaults to 1.
                `short_factor` (`List[float]`, *optional*):
                    Only used with 'longrope'. The scaling factor to be applied to short contexts (<
                    `original_max_position_embeddings`). Must be a list of numbers with the same length as the hidden
                    size divided by the number of attention heads divided by 2
                `long_factor` (`List[float]`, *optional*):
                    Only used with 'longrope'. The scaling factor to be applied to long contexts (<
                    `original_max_position_embeddings`). Must be a list of numbers with the same length as the hidden
                    size divided by the number of attention heads divided by 2
                `low_freq_factor` (`float`, *optional*):
                    Only used with 'llama3'. Scaling factor applied to low frequency components of the RoPE
                `high_freq_factor` (`float`, *optional*):
                    Only used with 'llama3'. Scaling factor applied to high frequency components of the RoPE
        partial_rotary_factor (`float`, *optional*, defaults to 0.25):
            Percentage of the query and keys which will have rotary embedding.
        attention_bias (`bool`, *optional*, defaults to `False`):
            Whether to use a bias in the query, key, value and output projection layers during self-attention.
        attention_dropout (`float`, *optional*, defaults to 0.0):
            The dropout ratio for the attention probabilities.
        head_dim (`int`, *optional*, defaults to 256):
            Projection weights dimension in multi-head attention.
        linear_conv_kernel_dim (`int`, *optional*, defaults to 4):
            Kernel size of the convolution used in linear attention layers.
        linear_key_head_dim (`int`, *optional*, defaults to 128):
            Dimension of each key head in linear attention.
        linear_value_head_dim (`int`, *optional*, defaults to 128):
            Dimension of each value head in linear attention.
        linear_num_key_heads (`int`, *optional*, defaults to 16):
            Number of key heads used in linear attention layers.
        linear_num_value_heads (`int`, *optional*, defaults to 32):
            Number of value heads used in linear attention layers.
        decoder_sparse_step (`int`, *optional*, defaults to 1):
            The frequency of the MoE layer.
        moe_intermediate_size (`int`, *optional*, defaults to 512):
            Intermediate size of the routed expert.
        shared_expert_intermediate_size (`int`, *optional*, defaults to 512):
            Intermediate size of the shared expert.
        num_experts_per_tok (`int`, *optional*, defaults to 10):
            Number of selected experts.
        num_experts (`int`, *optional*, defaults to 512):
            Number of routed experts.
        norm_topk_prob (`bool`, *optional*, defaults to `True`):
            Whether to normalize the topk probabilities.
        output_router_logits (`bool`, *optional*, defaults to `False`):
            Whether or not the router logits should be returned by the model. Enabling this will also
            allow the model to output the auxiliary loss, including load balancing loss and router z-loss.
        router_aux_loss_coef (`float`, *optional*, defaults to 0.001):
            The aux loss factor for the total loss.
        mlp_only_layers (`list[int]`, *optional*, defaults to `[]`):
            Indicate which layers use Qwen3NextMLP rather than Qwen3NextSparseMoeBlock
            The list contains layer index, from 0 to num_layers-1 if we have num_layers layers
            If `mlp_only_layers` is empty, `decoder_sparse_step` is used to determine the sparsity.
        layer_types (`list[str]`, *optional*, defaults to None):
            Types of each layer (attention or linear).

    ```python
    >>> from transformers import Qwen3NextModel, Qwen3NextConfig

    >>> # Initializing a Qwen3Next style configuration
    >>> configuration =  Qwen3NextConfig()

    >>> # Initializing a model from the Qwen3-Next-80B-A3B style configuration
    >>> model = Qwen3NextModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```
    """

```
**File:** python/sglang/srt/server_args.py (L1425-1483)
```python
        elif model_arch in ["Qwen3NextForCausalLM"]:
            if is_sm100_supported():
                quantization_config = getattr(hf_config, "quantization_config", None)
                quant_method = (
                    quantization_config.get("quant_method")
                    if quantization_config is not None
                    else None
                )
                if self.quantization is None and quant_method is not None:
                    self.quantization = quant_method
                if (
                    (self.quantization == "fp8" or self.quantization == "modelopt_fp4")
                    and self.moe_a2a_backend == "none"
                    and self.moe_runner_backend == "auto"
                ):
                    self.moe_runner_backend = "flashinfer_trtllm"
                    logger.info(
                        "Use flashinfer_trtllm as MoE runner backend on sm100 for Qwen3NextForCausalLM"
                    )
                if self.attention_backend is None:
                    self.attention_backend = "triton"
                    logger.info(
                        "Use triton as attention backend on sm100 for Qwen3NextForCausalLM"
                    )
                if (
                    not self.disable_radix_cache
                    and self.attention_backend == "trtllm_mha"
                ):
                    logger.warning(
                        "Disabling radix cache since trtllm_mha does not support page_size = 1, which is required by MambaRadixCache. "
                        "Try to use --attention-backend triton if radix cache is necessary."
                    )
                    self.disable_radix_cache = True
                    self.disable_overlap_schedule = False

            # Mamba radix cache v2
            if self.enable_mamba_extra_buffer():
                assert (
                    is_cuda()
                ), "Mamba extra_buffer is only supported on CUDA devices with FLA backend"
                if self.speculative_num_draft_tokens is not None:
                    assert (
                        self.mamba_track_interval >= self.speculative_num_draft_tokens
                    ), f"mamba_track_interval {self.mamba_track_interval} must be greater than or equal to speculative_num_draft_tokens {self.speculative_num_draft_tokens}"

                if self.page_size is not None:
                    assert (
                        self.mamba_track_interval % self.page_size == 0
                    ), f"mamba_track_interval {self.mamba_track_interval} must be divisible by page_size {self.page_size}"
                    assert (
                        FLA_CHUNK_SIZE % self.page_size == 0
                    ), f"Page size for hybrid GDN model must be divisible by {FLA_CHUNK_SIZE}, got {self.page_size}"

            elif not self.disable_radix_cache:
                logger.warning(
                    "Disabling overlap schedule since MambaRadixCache no_buffer is not compatible with "
                    "overlap schedule currently, try to use --mamba-scheduler-strategy extra_buffer to enable overlap schedule"
                )
                self.disable_overlap_schedule = True
```
**File:** docs/basic_usage/qwen3.md (L1-39)
```markdown
# Qwen3-Next Usage

SGLang has supported Qwen3-Next-80B-A3B-Instruct and Qwen3-Next-80B-A3B-Thinking since [this PR](https://github.com/sgl-project/sglang/pull/10233).

## Launch Qwen3-Next with SGLang

To serve Qwen3-Next models on 4xH100/H200 GPUs:

```bash
python3 -m sglang.launch_server --model Qwen/Qwen3-Next-80B-A3B-Instruct --tp 4
```

### Configuration Tips
- `--max-mamba-cache-size`: Adjust `--max-mamba-cache-size` to increase mamba cache space and max running requests capability. It will decrease KV cache space as a trade-off. You can adjust it according to workload.
- `--mamba-ssm-dtype`: `bfloat16` or `float32`, use `bfloat16` to save mamba cache size and `float32` to get more accurate results. The default setting is `float32`.
- `--mamba-full-memory-ratio`: The ratio of mamba state memory to full kv cache memory. The default is 0.9.

### Mamba Radix Cache
SGLang supports prefix caching for Qwen3-Next models named `MambaRadixCache`, which improves inference speed by reusing computation results. There are two versions of `MambaRadixCache`:
- `no_buffer`: The default version, which is also other hybrid linear models' choice. When it is enabled, SGLang will automatically close overlap schedule for compatibility reasons.
- `extra_buffer`: An optimized version that is compatible with features like page size > 1, overlap schedule, and speculative decoding. It also supports storing mamba state in branching positions. However, it requires two extra mamba spaces for a ping-pong buffer for each request. To enable it, add the argument `--mamba-scheduler-strategy extra_buffer` when launching the server.

### EAGLE Speculative Decoding
**Description**: SGLang has supported Qwen3-Next models with [EAGLE speculative decoding](https://docs.sglang.io/advanced_features/speculative_decoding.html#EAGLE-Decoding).

**Usage**:
Add arguments `--speculative-algorithm`, `--speculative-num-steps`, `--speculative-eagle-topk` and `--speculative-num-draft-tokens` to enable this feature. For example:

``` bash
python3 -m sglang.launch_server \
  --model Qwen/Qwen3-Next-80B-A3B-Instruct \
  --tp 4 \
  --speculative-num-steps 3 \
  --speculative-eagle-topk 1 \
  --speculative-num-draft-tokens 4 \
  --speculative-algo NEXTN
```

```
**File:** test/srt/models/test_qwen3_next_models.py (L39-124)
```python
class TestQwen3Next(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = QWEN3_NEXT_MODEL
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=[
                "--tp-size",
                "4",
                "--chunked-prefill-size",
                "2048",
                "--mamba-scheduler-strategy",
                "extra_buffer",
                "--mamba-track-interval",
                "128",
            ],
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    def test_gsm8k(self):
        args = SimpleNamespace(
            num_shots=5,
            data_path=None,
            num_questions=200,
            max_new_tokens=512,
            parallel=128,
            host="http://127.0.0.1",
            port=int(self.base_url.split(":")[-1]),
        )
        metrics = run_eval(args)
        print(f"{metrics=}")
        self.assertGreaterEqual(
            metrics["accuracy"], ACC_THRESHOLDS[self.model]["gsm8k"]
        )

    def test_input_output_logprobs_match_prefill_cache_hit(self):
        test_input_output_logprobs_match_prefill_cache_hit_helper(
            self.base_url,
            ACC_THRESHOLDS,
            self.model,
            max_samples=32,
            max_new_tokens=512,
        )

    def test_input_output_logprobs_match_decode_cache_hit(self):
        test_input_output_logprobs_match_decode_cache_hit_helper(
            self.base_url,
            ACC_THRESHOLDS,
            self.model,
            max_samples=32,
            max_new_tokens=512,
        )

    def test_prefix_cache_branching(self):
        print("running test_prefix_cache_branching")
        requests.get(self.base_url + "/flush_cache")
        branching_pos = 257
        text_prefix = "hi" * branching_pos
        suffix_list = ["this" * 256, "here" * 256, "that" * 256]
        cache_hit_list = [False, False, True]

        # First request only prefill the entire sequence
        # Second request won't have cache hit, but will cache the branching point
        # Third request will have cache hit on the branching point
        for i, (suffix, cache_hit) in enumerate(
            zip(suffix_list, cache_hit_list, strict=True)
        ):
            result = send_request_helper(self.base_url, text_prefix + suffix)
            cached_tokens = result["meta_info"]["cached_tokens"]
            if cache_hit:
                expected_cached_tokens = branching_pos // 64 * 64
                assert (
                    cached_tokens == expected_cached_tokens
                ), f"{i=}, {cache_hit=}, {cached_tokens=} is not equal to {expected_cached_tokens=}, {branching_pos=}"
            else:
                assert (
                    cached_tokens == 0
                ), f"{i=}, {cache_hit=}, {cached_tokens=} is not 0"
        print("test_prefix_cache_branching passed")

```
**File:** python/sglang/srt/models/qwen3_next_mtp.py (L1-32)
```python
# Copyright 2023-2024 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Inference-only Qwen3Next MTP Speculative Decoding."""
import logging
from typing import Iterable, Optional, Tuple

import torch
from torch import nn
from transformers import PretrainedConfig

from sglang.srt.distributed import get_pp_group, get_tensor_model_parallel_world_size
from sglang.srt.layers.layernorm import GemmaRMSNorm
from sglang.srt.layers.logits_processor import LogitsProcessor
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.layers.vocab_parallel_embedding import ParallelLMHead
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.models.qwen3_next import Qwen3NextForCausalLM, Qwen3NextModel
from sglang.srt.server_args import get_global_server_args
from sglang.srt.utils import add_prefix

```
## Q4
如果我希望在qwen3 next的基础上，自定义修改为我需要的模型，以插件module的形式独立注册为一个新的模型，该如何做？
### Answer
您可以在 Qwen3 Next 的基础上创建自定义模型并以插件形式独立注册。以下是具体步骤：

## 快速指南

1. **继承 Qwen3NextForCausalLM** 创建自定义模型类
2. **重写需要修改的方法**（如 `forward` 方法）
3. **通过 ModelRegistry 注册** 新模型
4. **修改模型配置** 的 `architectures` 字段

## 详细实现

### 1. 创建自定义模型类

```python
# 在您的插件模块中，例如 my_qwen3_custom.py
import torch
from typing import Optional
from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, PPProxyTensors
from sglang.srt.models.qwen3_next import Qwen3NextForCausalLM
from sglang.srt.configs.qwen3_next import Qwen3NextConfig

class MyQwen3Custom(Qwen3NextForCausalLM):
    """基于 Qwen3 Next 的自定义模型"""
    
    def __init__(
        self,
        config: Qwen3NextConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__(config=config, quant_config=quant_config, prefix=prefix)
        # 添加您的自定义初始化代码
        self.custom_param = torch.nn.Linear(config.hidden_size, config.hidden_size)
    
    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        inputs_embeds: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        # 调用父类的 forward 方法
        hidden_states = self.model(input_ids, positions, forward_batch, inputs_embeds)
        
        # 获取原始 logits
        res = self.logits_processor(
            input_ids, hidden_states, self.lm_head, forward_batch
        )
        
        # 添加您的自定义处理
        # 例如：对特定 token 的 logits 进行调整
        custom_logits = self.custom_param(hidden_states)
        res.next_token_logits = res.next_token_logits + 0.1 * custom_logits
        
        return res

# 注册入口
EntryClass = MyQwen3Custom
``` sglang:863-915 

### 2. 注册模型到 ModelRegistry

```python
# 在您的插件主文件中
from functools import lru_cache
from sglang.srt.models.registry import ModelRegistry
from my_qwen3_custom import MyQwen3Custom

@lru_cache()
def import_custom_model_classes():
    return {
        "MyQwen3Custom": MyQwen3Custom,
    }

# 注册模型
ModelRegistry.models.update(import_custom_model_classes())
``` sglang:128-132 

### 3. 修改模型配置文件

在您的模型目录中修改 `config.json`：

```json
{
  "architectures": [
    "MyQwen3Custom"
  ],
  "model_type": "qwen3_next",
  // ... 其他配置保持不变
}
```

### 4. 完整的插件使用示例

```python
import asyncio
import sglang as sgl
from functools import lru_cache
from sglang.srt.models.registry import ModelRegistry
from my_qwen3_custom import MyQwen3Custom

# 注册自定义模型
@lru_cache()
def import_custom_model_classes():
    return {"MyQwen3Custom": MyQwen3Custom}

ModelRegistry.models.update(import_custom_model_classes())

async def main():
    # 使用自定义模型
    llm = sgl.Engine(model_path="./path/to/your/qwen3-next-model")
    
    sampling_params = {
        "temperature": 0.7,
        "max_new_tokens": 100
    }
    
    prompts = ["你好，请介绍一下你自己"]
    outputs = await llm.async_generate(prompts, sampling_params)
    
    for prompt, output in zip(prompts, outputs):
        print(f"Prompt: {prompt}")
        print(f"Generated: {output['text']}")
    
    llm.shutdown()

if __name__ == "__main__":
    asyncio.run(main())
``` sglang:270-309 

## 可扩展的修改点

### 1. 修改层结构

您可以重写 `Qwen3NextModel` 来添加自定义层：

```python
from sglang.srt.models.qwen3_next import Qwen3NextModel, ALL_DECODER_LAYER_TYPES

class MyQwen3NextModel(Qwen3NextModel):
    def __init__(self, config, quant_config=None, prefix=""):
        # 添加自定义层类型
        ALL_DECODER_LAYER_TYPES["custom_layer"] = MyCustomDecoderLayer
        
        # 修改层类型映射
        if hasattr(config, 'layers_block_type'):
            # 自定义层的类型分配逻辑
            pass
        
        super().__init__(config, quant_config, prefix)
``` sglang:776-813 

### 2. 修改 MoE 配置

如果需要修改专家路由逻辑：

```python
from sglang.srt.models.qwen2_moe import Qwen2MoeSparseMoeBlock

class MyCustomMoEBlock(Qwen2MoeSparseMoeBlock):
    def forward(self, hidden_states, forward_batch, use_reduce_scatter):
        # 自定义专家选择逻辑
        router_logits = self.gate(hidden_states)
        
        # 您的自定义路由算法
        custom_routing_weights = your_custom_routing(router_logits)
        
        # 继续使用原始的 MoE 前向传播
        return super().forward(hidden_states, forward_batch, use_reduce_scatter)
``` sglang:498-505 

### 3. 添加 MTP 支持

如果需要推测解码支持：

```python
from sglang.srt.models.qwen3_next_mtp import Qwen3NextForCausalLMMTP

class MyQwen3CustomMTP(Qwen3NextForCausalLMMTP):
    def __init__(self, config, quant_config=None, prefix=""):
        super().__init__(config, quant_config, prefix)
        # 添加您的 MTP 自定义逻辑
``` sglang:36-74 

## 注意事项

1. **保持接口兼容性**：确保您的自定义模型保持与 SGLang 运行时的接口兼容
2. **性能考虑**：自定义修改可能会影响性能，建议进行基准测试
3. **配置一致性**：确保配置文件与您的模型实现匹配
4. **依赖管理**：插件需要独立管理其依赖关系

## Notes

- Qwen3 Next 使用混合架构，包含 `Qwen3HybridAttentionDecoderLayer` 和 `Qwen3HybridLinearDecoderLayer` sglang:776-779 
- 模型通过 `EntryClass` 进行注册 sglang:1046-1046 
- 支持通过环境变量 `SGLANG_EXTERNAL_MODEL_PACKAGE` 自动加载外部模型包 sglang:131-132
### Citations
**File:** python/sglang/srt/models/qwen3_next.py (L498-505)
```python
        if self.is_layer_sparse:
            self.mlp = Qwen2MoeSparseMoeBlock(
                layer_id=layer_id,
                config=config,
                quant_config=quant_config,
                alt_stream=alt_stream,
                prefix=add_prefix("mlp", prefix),
            )
```
**File:** python/sglang/srt/models/qwen3_next.py (L776-813)
```python
ALL_DECODER_LAYER_TYPES = {
    "attention": Qwen3HybridAttentionDecoderLayer,
    "linear_attention": Qwen3HybridLinearDecoderLayer,
}


class Qwen3NextModel(nn.Module):
    def __init__(
        self,
        config: Qwen3NextConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.config = config

        alt_stream = torch.cuda.Stream() if _is_cuda else None

        self.embed_tokens = VocabParallelEmbedding(
            config.vocab_size,
            config.hidden_size,
            org_num_embeddings=config.vocab_size,
            enable_tp=not is_dp_attention_enabled(),
        )

        def get_layer(idx: int, prefix: str):
            layer_class = ALL_DECODER_LAYER_TYPES[config.layers_block_type[idx]]
            return layer_class(
                config,
                idx,
                quant_config=quant_config,
                prefix=prefix,
                alt_stream=alt_stream,
            )

        self.layers = make_layers(
            config.num_hidden_layers, get_layer, prefix=f"{prefix}.layers"
        )
```
**File:** python/sglang/srt/models/qwen3_next.py (L863-915)
```python
class Qwen3NextForCausalLM(nn.Module):
    fall_back_to_pt_during_load = False

    def __init__(
        self,
        config: Qwen3NextConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.config = config
        self.pp_group = get_pp_group()
        assert self.pp_group.is_first_rank and self.pp_group.is_last_rank
        self.quant_config = quant_config
        self.model = Qwen3NextModel(
            config, quant_config, prefix=add_prefix("model", prefix)
        )
        self.lm_head = ParallelLMHead(
            config.vocab_size,
            config.hidden_size,
            quant_config=quant_config,
            org_num_embeddings=config.vocab_size,
            prefix=add_prefix("lm_head", prefix),
            use_attn_tp_group=get_global_server_args().enable_dp_lm_head,
        )
        self.logits_processor = LogitsProcessor(config)

        self._routed_experts_weights_of_layer = LazyValue(
            lambda: {
                layer_id: layer.mlp.get_moe_weights()
                for layer_id, layer in enumerate(self.model.layers)
                if isinstance(layer.mlp, Qwen2MoeSparseMoeBlock)
            }
        )

    @property
    def routed_experts_weights_of_layer(self):
        return self._routed_experts_weights_of_layer.value

    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        inputs_embeds: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        hidden_states = self.model(input_ids, positions, forward_batch, inputs_embeds)

        return self.logits_processor(
            input_ids, hidden_states, self.lm_head, forward_batch
        )
```
**File:** python/sglang/srt/models/qwen3_next.py (L1046-1046)
```python
EntryClass = Qwen3NextForCausalLM
```
**File:** python/sglang/srt/models/registry.py (L128-132)
```python
ModelRegistry = _ModelRegistry()
ModelRegistry.register("sglang.srt.models")

if external_pkg := envs.SGLANG_EXTERNAL_MODEL_PACKAGE.get():
    ModelRegistry.register(external_pkg, overwrite=True)
```
**File:** docs/supported_models/support_new_models.md (L270-309)
```markdown
```python
@lru_cache()
def import_new_model_classes():
    model_arch_name_to_cls = {"LlamaWrapper": LlamaWrapper}
    return model_arch_name_to_cls

ModelRegistry.models.update(import_new_model_classes())
```

Lastly, when we create our `Engine`, we just pass in the path to the local model directory.
Then, our `LlamaWrapper` is ready to be served; for this walkthrough, we will use SGLang `Engine`'s non-streaming asynchronous generation endpoint.

```python
def main():
    llm = sgl.Engine(model_path="./llama_ckpt")
    sampling_params = {"temperature": 0.2, "top_k": 5}
    prompts = [
        "Write a short, neutral self-introduction for a fictional character. Hello, my name is",
        "Provide a concise factual statement about France’s capital city. The capital of France is",
        "Explain possible future trends in artificial intelligence. The future of AI is",
    ]

    asyncio.run(run_llm(llm, sampling_params, prompts))

    llm.shutdown()

async def run_llm(
    llm,
    sampling_params,
    prompts,
) -> None:
    outputs = await llm.async_generate(prompts, sampling_params)

    for prompt, output in zip(prompts, outputs):
        print(f"\nPrompt: {prompt}")
        print(f"Generated text: {output['text']}")

if __name__ == "__main__":
    main()
```
```
**File:** python/sglang/srt/models/qwen3_next_mtp.py (L36-74)
```python
class Qwen3NextForCausalLMMTP(Qwen3NextForCausalLM):

    def __init__(
        self,
        config: PretrainedConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        nn.Module.__init__(self)
        self.config = config
        self.tp_size = get_tensor_model_parallel_world_size()
        self.quant_config = quant_config
        # if not set, model load will be broken in Qwen3NextForCausalLM load_weights()
        self.pp_group = get_pp_group()
        # self.determine_num_fused_shared_experts("Qwen3NextForCausalLMMTP")

        # currently based on the provided ckpt, we:
        # (1) do not use_dedicated_mtp_embeddings provided in ckpt since not provided and directly use the target model embeddings
        # (2) hardcode bias=False since not provided
        self.fc = nn.Linear(2 * config.hidden_size, config.hidden_size, bias=False)
        RMSNorm_cls = GemmaRMSNorm
        self.pre_fc_norm_embedding = RMSNorm_cls(
            config.hidden_size, config.rms_norm_eps
        )
        self.pre_fc_norm_hidden = RMSNorm_cls(config.hidden_size, config.rms_norm_eps)
        config.num_hidden_layers = 1
        config.full_attention_interval = 1
        self.model = Qwen3NextModel(
            config, quant_config, prefix=add_prefix("model", prefix)
        )
        self.lm_head = ParallelLMHead(
            config.vocab_size,
            config.hidden_size,
            quant_config=quant_config,
            prefix=add_prefix("model.shared_head.head", prefix),
            use_attn_tp_group=get_global_server_args().enable_dp_lm_head,
        )
        self.logits_processor = LogitsProcessor(config)

```
## Q5
请你给出完整的创建plugin流程方案和注意事项
### Answer
# 完整的SGLang插件创建流程方案

## 一、项目准备阶段

### 1.1 创建插件项目结构

```
my_sglang_plugin/
├── setup.py                 # 包安装配置
├── pyproject.toml          # 现代Python项目配置
├── README.md               # 项目说明
├── my_sglang_plugin/
│   ├── __init__.py         # 包初始化
│   ├── models/
│   │   ├── __init__.py
│   │   ├── custom_model.py # 自定义模型实现
│   │   └── config.py       # 模型配置
│   └── processors/         # 多模态处理器(可选)
│       ├── __init__.py
│       └── custom_processor.py
└── tests/
    ├── __init__.py
    └── test_custom_model.py
```

### 1.2 配置项目依赖

在 `pyproject.toml` 中定义依赖：

```toml
[build-system]
requires = ["setuptools>=61.0", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "my-sglang-plugin"
version = "0.1.0"
description = "Custom SGLang models plugin"
dependencies = [
    "sglang>=0.3.0",
    "torch",
    "transformers",
]

[project.entry-points."sglang.models"]
custom_model = "my_sglang_plugin.models.custom_model:EntryClass"
```

## 二、模型实现阶段

### 2.1 继承基础模型类

```python
# my_sglang_plugin/models/custom_model.py
import torch
from typing import Optional
from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.models.qwen3_next import Qwen3NextForCausalLM

class MyCustomModel(Qwen3NextForCausalLM):
    """自定义模型实现"""
    
    def __init__(
        self,
        config,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__(config=config, quant_config=quant_config, prefix=prefix)
        # 添加自定义层
        self.custom_layer = torch.nn.Linear(config.hidden_size, config.hidden_size)
    
    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        inputs_embeds: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> LogitsProcessorOutput:
        # 调用父类前向传播
        hidden_states = self.model(input_ids, positions, forward_batch, inputs_embeds)
        
        # 获取logits
        res = self.logits_processor(input_ids, hidden_states, self.lm_head, forward_batch)
        
        # 自定义处理
        custom_features = self.custom_layer(hidden_states)
        res.next_token_logits = res.next_token_logits + 0.1 * custom_features
        
        return res

# 注册入口
EntryClass = [MyCustomModel]
``` sglang:166-225 

### 2.2 实现多模态处理器（可选）

```python
# my_sglang_plugin/processors/custom_processor.py
from sglang.srt.multimodal.processors import BaseMultimodalProcessor

class CustomMultimodalProcessor(BaseMultimodalProcessor):
    def process(self, inputs):
        # 实现自定义的多模态数据处理
        pass
```

## 三、注册机制阶段

### 3.1 手动注册方式

```python
# 使用插件时注册
from functools import lru_cache
from sglang.srt.models.registry import ModelRegistry
from my_sglang_plugin.models.custom_model import MyCustomModel

@lru_cache()
def import_custom_model_classes():
    return {"MyCustomModel": MyCustomModel}

ModelRegistry.models.update(import_custom_model_classes())
``` sglang:128-132 

### 3.2 环境变量自动加载

```bash
export SGLANG_EXTERNAL_MODEL_PACKAGE="my_sglang_plugin"
``` sglang:131-132 

## 四、测试验证阶段

### 4.1 单元测试

```python
# tests/test_custom_model.py
import unittest
import sglang as sgl
from sglang.srt.models.registry import ModelRegistry
from my_sglang_plugin.models.custom_model import MyCustomModel

class TestCustomModel(unittest.TestCase):
    def setUp(self):
        # 注册模型
        ModelRegistry.models["MyCustomModel"] = MyCustomModel
    
    def test_model_loading(self):
        # 测试模型加载
        engine = sgl.Engine(model_path="./test_model")
        self.assertIsNotNone(engine)
        engine.shutdown()
```

### 4.2 精度对比测试

```bash
# HuggingFace参考输出
python3 scripts/playground/reference_hf.py --model-path [model] --model-type text

# SGLang输出
python3 -m sglang.bench_one_batch --correct --model [model]
``` sglang:53-63 

## 五、部署使用阶段

### 5.1 离线Engine使用

```python
import asyncio
import sglang as sgl

async def main():
    # 注册插件模型
    from my_sglang_plugin.models.custom_model import MyCustomModel
    from sglang.srt.models.registry import ModelRegistry
    ModelRegistry.models["MyCustomModel"] = MyCustomModel
    
    # 创建Engine
    llm = sgl.Engine(model_path="./path/to/model")
    
    # 生成文本
    sampling_params = {"temperature": 0.7, "max_new_tokens": 100}
    outputs = await llm.async_generate(["Hello"], sampling_params)
    print(outputs)
    
    llm.shutdown()

if __name__ == "__main__":
    asyncio.run(main())
``` sglang:270-309 

### 5.2 HTTP服务器部署

```bash
# 设置环境变量
export SGLANG_EXTERNAL_MODEL_PACKAGE="my_sglang_plugin"

# 启动服务器
python3 -m sglang.launch_server \
  --model-path ./path/to/model \
  --port 30000
```

## 六、注意事项

### 6.1 开发注意事项

1. **接口兼容性**
   - 确保 `forward` 方法签名与SGLang要求一致 sglang:863-915 
   - 保持与 `LogitsProcessorOutput` 的兼容性

2. **性能考虑**
   - 使用 `@torch.no_grad()` 装饰器减少内存占用
   - 避免在前向传播中创建不必要的张量
   - 考虑使用CUDA图优化

3. **配置管理**
   - 修改模型的 `config.json` 中的 `architectures` 字段
   - 确保配置参数与模型实现匹配

### 6.2 部署注意事项

1. **依赖版本**
   - 明确指定SGLang版本兼容性
   - 使用虚拟环境隔离依赖

2. **模型路径**
   - 确保模型权重路径正确
   - 检查配置文件完整性

3. **资源管理**
   - 适当设置GPU内存限制
   - 监控模型加载和推理性能

### 6.3 维护注意事项

1. **版本控制**
   - 使用语义化版本号
   - 维护变更日志

2. **文档完善**
   - 提供详细的API文档
   - 包含使用示例和最佳实践

3. **测试覆盖**
   - 编写全面的单元测试
   - 进行集成测试验证

## Notes

- 插件模式特别适合企业内部模型或需要保密的模型实现 sglang:110-136 
- SGLang的ModelRegistry使用`@lru_cache`优化模型类导入 sglang:92-93 
- 对于多模态模型，需要额外实现相应的处理器和视觉组件 sglang:27-43 
- 插件包可以通过PyPI分发，便于团队共享和使用

Wiki pages you might want to explore:
- [Overview (sgl-project/sglang)](/wiki/sgl-project/sglang#1)
### Citations
**File:** docs/supported_models/support_new_models.md (L27-43)
```markdown
3. **Multimodal Data Processor**:
   Define a new `Processor` class that inherits from `BaseMultimodalProcessor` and register this processor as your
   model’s dedicated processor.
   See [multimodal_processor.py](https://github.com/sgl-project/sglang/tree/main/python/sglang/srt/multimodal/processors)
   for more details.

4. **Handle Multimodal Tokens**:
   Implement a `pad_input_ids` function for your new model. In this function, multimodal tokens in the prompt should be
   expanded (if necessary) and padded with multimodal-data-hashes so that SGLang can recognize different multimodal data
   with `RadixAttention`.

5. **Handle Image Feature Extraction**:
   Implement a `get_image_feature` function for your new model, which extracts image features from raw image data and converts them into the embeddings used by the language model.

6. **Adapt to Vision Attention**:
   Adapt the multi-headed `Attention` of ViT with SGLang’s `VisionAttention`.

```
**File:** docs/supported_models/support_new_models.md (L53-63)
```markdown
For interactive debugging, compare the outputs of Hugging Face/Transformers and SGLang. The following two commands
should give the same text output and very similar prefill logits:

- Get the reference output:
  ```bash
  python3 scripts/playground/reference_hf.py --model-path [new model] --model-type {text,mllm}
  ```
- Get the SGLang output:
  ```bash
  python3 -m sglang.bench_one_batch --correct --model [new model]
  ```
```
**File:** docs/supported_models/support_new_models.md (L110-136)
```markdown
In addition to the methods above, you can register your new model with the `ModelRegistry` before launching the server.
This allows you to integrate your model without modifying the source code.

For example:

```python
from sglang.srt.models.registry import ModelRegistry
from sglang.srt.entrypoints.http_server import launch_server

# For a single model, add it to the registry:
ModelRegistry.models[model_name] = model_class

# For multiple models, you can imitate the import_model_classes() function:
from functools import lru_cache

@lru_cache()
def import_new_model_classes():
    model_arch_name_to_cls = {}
    # Populate model_arch_name_to_cls with your new model classes.
    ...
    return model_arch_name_to_cls

ModelRegistry.models.update(import_new_model_classes())

# Launch the server with your server arguments:
launch_server(server_args)
```
```
**File:** docs/supported_models/support_new_models.md (L166-225)
```markdown
```python
class LlamaWrapper(LlamaForCausalLM):
    def __init__(
        self,
        config: LlamaConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__(config=config, quant_config=quant_config, prefix=prefix)
```

Now, we want to define the `forward` method, which is what will be called at inference time.
Note that the signature for `forward` is essentially the same for any model; you can take a look at the other models defined in the [`models` directory](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/models/) for references.
To see where exactly `forward` is called in the SGLang runtime's internals, take a look at [`forward_decode`](https://github.com/sgl-project/sglang/blob/bf72b80122fd888bf619d17b96fa3e323ab809fc/python/sglang/srt/model_executor/model_runner.py#L1705) and [`forward_extend`](https://github.com/sgl-project/sglang/blob/bf72b80122fd888bf619d17b96fa3e323ab809fc/python/sglang/srt/model_executor/model_runner.py#L1724) in the [`ModelRunner` class](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/model_executor/model_runner.py).

```python
    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        pp_proxy_tensors: Optional[PPProxyTensors] = None,
        input_embeds: Optional[torch.Tensor] = None,
        get_embedding: bool = False,
    ) -> LogitsProcessorOutput:
```

We now call the `__call__` method for `self.model` (which is a member variable that `LlamaForCausalLM` defines in its `__init__` method), which eventually calls `LlamaForCausalLM`'s `forward` method.
After that, we feed the `hidden_states` into our model's `LogitsProcessor` (again defined in `LlamaForCausalLM`).

```python
        hidden_states = self.model(
            input_ids,
            positions,
            forward_batch,
            input_embeds,
            pp_proxy_tensors=pp_proxy_tensors,
        )

        res: LogitsProcessorOutput = self.logits_processor(
            input_ids,
            hidden_states,
            self.lm_head,
            forward_batch,
        )
```

After receiving the logits for the next token, we can finally perform our biasing step.

```python
        orig_logits = res.next_token_logits
        res.next_token_logits = torch.where(
            orig_logits > 0,
            orig_logits.sqrt(),
            orig_logits
        )

        return res
```
```
**File:** docs/supported_models/support_new_models.md (L270-309)
```markdown
```python
@lru_cache()
def import_new_model_classes():
    model_arch_name_to_cls = {"LlamaWrapper": LlamaWrapper}
    return model_arch_name_to_cls

ModelRegistry.models.update(import_new_model_classes())
```

Lastly, when we create our `Engine`, we just pass in the path to the local model directory.
Then, our `LlamaWrapper` is ready to be served; for this walkthrough, we will use SGLang `Engine`'s non-streaming asynchronous generation endpoint.

```python
def main():
    llm = sgl.Engine(model_path="./llama_ckpt")
    sampling_params = {"temperature": 0.2, "top_k": 5}
    prompts = [
        "Write a short, neutral self-introduction for a fictional character. Hello, my name is",
        "Provide a concise factual statement about France’s capital city. The capital of France is",
        "Explain possible future trends in artificial intelligence. The future of AI is",
    ]

    asyncio.run(run_llm(llm, sampling_params, prompts))

    llm.shutdown()

async def run_llm(
    llm,
    sampling_params,
    prompts,
) -> None:
    outputs = await llm.async_generate(prompts, sampling_params)

    for prompt, output in zip(prompts, outputs):
        print(f"\nPrompt: {prompt}")
        print(f"Generated text: {output['text']}")

if __name__ == "__main__":
    main()
```
```
**File:** python/sglang/srt/models/registry.py (L92-93)
```python
@lru_cache()
def import_model_classes(package_name: str, strict: bool = False):
```
**File:** python/sglang/srt/models/registry.py (L128-132)
```python
ModelRegistry = _ModelRegistry()
ModelRegistry.register("sglang.srt.models")

if external_pkg := envs.SGLANG_EXTERNAL_MODEL_PACKAGE.get():
    ModelRegistry.register(external_pkg, overwrite=True)
```