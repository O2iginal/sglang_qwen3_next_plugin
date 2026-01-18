# DeepWiki Q&A with Code Context for Repository: sgl-project/sglang
## Q1
如何编写plugin module用来将自定义模型支持加入sglang？请给出完整方案和注意事项
### Answer
## 编写Plugin Module为SGLang添加自定义模型支持的完整方案

### 核心方案概述

SGLang支持通过两种方式添加自定义模型：1) 修改源码添加模型实现；2) 使用ModelRegistry注册外部模型。核心流程包括创建模型实现文件、注册模型类、配置模型类型检测，以及针对多模态模型的特殊处理。 sglang:1-180 

---

## 详细实施方案

### 1. 创建模型实现文件

在`python/sglang/srt/models/`目录下创建新模型文件，继承现有模型类：

```python
# 示例：基于Llama的自定义模型
from sglang.srt.models.llama import LlamaForCausalLM
from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.layers.quantization.base_config import QuantizationConfig

class CustomModel(LlamaForCausalLM):
    def __init__(self, config, quant_config=None, prefix=""):
        super().__init__(config=config, quant_config=quant_config, prefix=prefix)
    
    def forward(self, input_ids, positions, ...):
        # 自定义前向传播逻辑
        pass
    
    def forward_batch(self, forward_batch):
        # 批处理前向传播
        pass

# 注册入口类
EntryClass = CustomModel
```

### 2. 模型注册机制

SGLang使用`ModelRegistry`自动发现和注册模型： sglang:92-133 

```python
# 自动导入模型类
@lru_cache()
def import_model_classes(package_name: str, strict: bool = False):
    model_arch_name_to_cls = {}
    package = importlib.import_module(package_name)
    for _, name, ispkg in pkgutil.iter_modules(package.__path__, package_name + "."):
        if not ispkg:
            module = importlib.import_module(name)
            if hasattr(module, "EntryClass"):
                entry = module.EntryClass
                model_arch_name_to_cls[entry.__name__] = entry
    return model_arch_name_to_cls
```

### 3. 多模态模型特殊处理

对于多模态模型，需要额外步骤： sglang:14-46 

1. **注册多模态检测**：修改`is_multimodal_model`函数
2. **创建处理器**：继承`BaseMultimodalProcessor`
3. **处理多模态token**：实现`pad_input_ids`函数
4. **图像特征提取**：实现`get_image_feature`函数
5. **适配视觉注意力**：使用`VisionAttention`

### 4. 外部模型注册（无需修改源码）

使用ModelRegistry动态注册： sglang:108-137 

```python
from sglang.srt.models.registry import ModelRegistry
from sglang.srt.entrypoints.http_server import launch_server

# 注册单个模型
ModelRegistry.models[model_name] = model_class

# 批量注册
@lru_cache()
def import_new_model_classes():
    model_arch_name_to_cls = {}
    # 填充模型类映射
    return model_arch_name_to_cls

ModelRegistry.models.update(import_new_model_classes())
launch_server(server_args)
```

### 5. 测试和验证

**交互式调试**： sglang:51-64 

```bash
# 获取参考输出
python3 scripts/playground/reference_hf.py --model-path [new model] --model-type {text,mllm}

# 获取SGLang输出
python3 -m sglang.bench_one_batch --correct --model [new model]
```

**添加到测试套件**：在`test_generation_models.py`的`ALL_OTHER_MODELS`列表中添加模型，并运行基准测试。

---

## 重要注意事项

### 1. 架构适配要求

- **替换Attention**：使用`RadixAttention`替代vLLM的Attention，确保传递`layer_id` sglang:96-98 
- **替换LogitsProcessor**：使用SGLang的`LogitsProcessor`
- **移除Sample**：SGLang使用独立的采样逻辑
- **添加forward_batch方法**：支持批处理推理

### 2. 配置处理

模型配置通过`ModelConfig`类处理，包括：
- 模型类型检测（生成模型、多模态模型、嵌入模型等）
- 量化配置解析
- 注意力架构检测（MHA vs MLA）

### 3. 性能优化

- 使用SGLang的优化层（如`RMSNorm`、`SiluAndMul`）
- 支持量化方法（FP8、AWQ、GPTQ等）
- 利用RadixAttention的prefix caching功能

### 4. 兼容性考虑

- 确保模型实现不依赖vLLM组件
- 支持transformers fallback机制
- 处理远程模型加载和自定义配置

## Notes

- SGLang支持70+种模型架构，主要通过模型注册系统实现 sglang:26-60 
- 对于复杂模型（如MoE模型），需要特别注意专家并行化和路由逻辑
- 多模态模型需要额外的处理器和特征提取逻辑
- 建议参考现有模型实现（如Llama、Qwen）作为起点 sglang:8-12 

Wiki pages you might want to explore:
- [Model Support and Configuration (sgl-project/sglang)](/wiki/sgl-project/sglang#6)
### Citations
**File:** docs/supported_models/support_new_models.md (L1-180)
```markdown
# How to Support New Models

This document explains how to add support for new language models and multimodal large language models (MLLMs) in
SGLang. It also covers how to test new models and register external implementations.

## How to Support a New Language Model

To support a new model in SGLang, you only need to add a single file under
the [SGLang Models Directory](https://github.com/sgl-project/sglang/tree/main/python/sglang/srt/models). You can learn
from existing model implementations and create a new file for your model. For most models, you should be able to find a
similar model to start with (e.g., starting from Llama). Also refer how
to [port a Model from vLLM to SGLang](#port-a-model-from-vllm-to-sglang)

## How to Support a New Multimodal Large Language Model

To support a new multimodal large language model (MLLM) in SGLang, there are several key components in addition to the
standard LLM support:

1. **Register your new model as multimodal**:
   Extend `is_multimodal_model`
   in [model_config.py](https://github.com/sgl-project/sglang/blob/0ab3f437aba729b348a683ab32b35b214456efc7/python/sglang/srt/configs/model_config.py#L561)
   to return `True` for your model.

2. **Register a new chat-template**:
   Only when your default chat-template is unable to accept images as input: Register a new chat template in [conversation.py](https://github.com/sgl-project/sglang/tree/main/python/sglang/srt/conversation.py) and the corresponding matching function.

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

You can refer to [Qwen2VL](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/models/qwen2_vl.py) or
other mllm implementations. These models demonstrate how to correctly handle both multimodal and textual inputs.

## Testing and Debugging

Please note all your testing and benchmarking results in PR description.

### Interactive Debugging

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

### Add the Model to the Test Suite

To ensure the new model is well maintained, add it to the test suite by including it in the `ALL_OTHER_MODELS` list in
the [test_generation_models.py](https://github.com/sgl-project/sglang/blob/main/test/srt/models/test_generation_models.py)
file, test the new model on your local machine and report the results on demonstrative benchmarks (GSM8K, MMLU, MMMU,
MMMU-Pro, etc.) in your PR. \\
For VLMs, also include a test in `test_vision_openai_server_{x}.py` (e.g. [test_vision_openai_server_a.py](https://github.com/sgl-project/sglang/blob/main/test/srt/test_vision_openai_server_a.py), [test_vision_openai_server_b.py](https://github.com/sgl-project/sglang/blob/main/test/srt/test_vision_openai_server_b.py)).


This is an example command to run to test a new model on your local machine:

```bash
ONLY_RUN=Qwen/Qwen2-1.5B python3 -m unittest test_generation_models.TestGenerationModels.test_others
```

### Benchmark

- **(Required) MMMU**: follow MMMU benchmark [README.md](https://github.com/sgl-project/sglang/blob/main/benchmark/mmmu/README.md) to get SGLang vs. HF Transformer accuracy comparison. The accuracy score from SGLang run should not be much lower than that from HF Transformer run. Similarly, follow https://docs.sglang.io/developer_guide/benchmark_and_profiling.html to get performance comparison: TTFT and throughput must meet or exceed baselines (e.g., HF Transformer).
- **(Optional) Other evals**: If you ran other evals, please note the results in PR description.

## Port a Model from vLLM to SGLang

The [vLLM Models Directory](https://github.com/vllm-project/vllm/tree/main/vllm/model_executor/models) is a valuable
resource, as vLLM covers many models. SGLang reuses vLLM’s interface and some layers, making it easier to port models
from vLLM to SGLang.

To port a model from vLLM to SGLang:

- Compare these two files for guidance:
    - [SGLang Llama Implementation](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/models/llama.py)
    - [vLLM Llama Implementation](https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/models/llama.py)
- The major differences include:
    - **Replace vLLM’s `Attention` with `RadixAttention`** (ensure you pass `layer_id` to `RadixAttention`).
    - **Replace vLLM’s `LogitsProcessor` with SGLang’s `LogitsProcessor`.**
    - **Replace the multi-headed `Attention` of ViT with SGLang’s `VisionAttention`.**
    - **Replace other vLLM layers** (such as `RMSNorm`, `SiluAndMul`) with SGLang layers.
    - **Remove `Sample`.**
    - **Change the `forward()` functions** and add a `forward_batch()` method.
    - **Add `EntryClass`** at the end.
    - **Ensure that the new implementation uses only SGLang components** and does not rely on any vLLM components.

Note: make sure you add your new model to the supported models list in the supported models documentation.

## Registering an External Model Implementation

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

## Example: Implementing and Serving a Llama Wrapper Model

Below is an introductory, step-by-step walkthrough on how to implement a new model end-to-end in SGLang and then run it via the [Offline Engine](https://github.com/sgl-project/sglang/blob/main/docs/basic_usage/offline_engine_api.ipynb).

### Implementing Our Model

To keep things simple, this new model will be a simple wrapper around [Llama 3.1-8B-Instruct](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct), and our goal will be just to bias the output logits for each `forward` call by taking the square root of each individual logit.

Let's start by defining our model in a file called `llama_wrapper.py`.
The first step is to import the necessary libraries from SRT, which is SGLang's internal backend.

```python
# In the file `llama_wrapper.py`

import torch
from transformers import LlamaConfig
from typing import Optional
from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, PPProxyTensors

from sglang.srt.models.llama import LlamaForCausalLM
```

Next, we declare a new `class` for our model and have it inherit from `LlamaForCausalLM`, which allows our model to access `LlamaForCausalLM`'s predefined modules and layers, such as `LlamaAttention` and `LlamaMLP`.
Note that almost all model implementations take in `config` and `quant_config` as arguments for their `__init__` method; `config` and `quant_config` are passed in via [`model_loader/loader.py`](https://github.com/sgl-project/sglang/blob/bf72b80122fd888bf619d17b96fa3e323ab809fc/python/sglang/srt/model_loader/loader.py#L219).
Because we have inherited from `LlamaForCausalLM`, we can pass our parameters directly to its constructor, which will set the member variables for us.

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

```
**File:** python/sglang/srt/models/registry.py (L92-133)
```python
@lru_cache()
def import_model_classes(package_name: str, strict: bool = False):
    model_arch_name_to_cls = {}
    package = importlib.import_module(package_name)
    for _, name, ispkg in pkgutil.iter_modules(package.__path__, package_name + "."):
        if not ispkg:
            if name.split(".")[-1] in envs.SGLANG_DISABLED_MODEL_ARCHS.get():
                logger.debug(f"Skip loading {name} due to SGLANG_DISABLED_MODEL_ARCHS")
                continue

            try:
                module = importlib.import_module(name)
            except Exception as e:
                if strict:
                    raise
                logger.warning(f"Ignore import error when loading {name}: {e}")
                continue
            if hasattr(module, "EntryClass"):
                entry = module.EntryClass
                if isinstance(
                    entry, list
                ):  # To support multiple model classes in one module
                    for tmp in entry:
                        assert (
                            tmp.__name__ not in model_arch_name_to_cls
                        ), f"Duplicated model implementation for {tmp.__name__}"
                        model_arch_name_to_cls[tmp.__name__] = tmp
                else:
                    assert (
                        entry.__name__ not in model_arch_name_to_cls
                    ), f"Duplicated model implementation for {entry.__name__}"
                    model_arch_name_to_cls[entry.__name__] = entry

    return model_arch_name_to_cls


ModelRegistry = _ModelRegistry()
ModelRegistry.register("sglang.srt.models")

if external_pkg := envs.SGLANG_EXTERNAL_MODEL_PACKAGE.get():
    ModelRegistry.register(external_pkg, overwrite=True)

```
**File:** docs/supported_models/generative_models.md (L26-60)
```markdown
| Model Family (Variants)             | Example HuggingFace Identifier                     | Description                                                                            |
|-------------------------------------|--------------------------------------------------|----------------------------------------------------------------------------------------|
| **DeepSeek** (v1, v2, v3/R1)        | `deepseek-ai/DeepSeek-R1`                        | Series of advanced reasoning-optimized models (including a 671B MoE) trained with reinforcement learning; top performance on complex reasoning, math, and code tasks. [SGLang provides Deepseek v3/R1 model-specific optimizations](../basic_usage/deepseek.md) and [Reasoning Parser](../advanced_features/separate_reasoning.ipynb)|
| **Kimi K2** (Thinking, Instruct)    | `moonshotai/Kimi-K2-Instruct`                    | Moonshot AI's 1 trillion parameter MoE model (32B active) with 128K–256K context; state-of-the-art agentic intelligence with stable long-horizon agency across 200–300 sequential tool calls. Features MLA attention and native INT4 quantization. [See Reasoning Parser docs](../advanced_features/separate_reasoning.ipynb)|
| **Kimi Linear** (48B-A3B)           | `moonshotai/Kimi-Linear-48B-A3B-Instruct`        | Moonshot AI's hybrid linear attention model (48B total, 3B active) with 1M token context; features Kimi Delta Attention (KDA) for up to 6× faster decoding and 75% KV cache reduction vs full attention. |
| **GPT-OSS**       | `openai/gpt-oss-20b`, `openai/gpt-oss-120b`       | OpenAI’s latest GPT-OSS series for complex reasoning, agentic tasks, and versatile developer use cases.|
| **Qwen** (3, 3MoE, 3Next, 2.5, 2 series)       | `Qwen/Qwen3-0.6B`, `Qwen/Qwen3-30B-A3B` `Qwen/Qwen3-Next-80B-A3B-Instruct `      | Alibaba’s latest Qwen3 series for complex reasoning, language understanding, and generation tasks; Support for MoE variants along with previous generation 2.5, 2, etc. [SGLang provides Qwen3 specific reasoning parser](../advanced_features/separate_reasoning.ipynb)|
| **Llama** (2, 3.x, 4 series)        | `meta-llama/Llama-4-Scout-17B-16E-Instruct`       | Meta's open LLM series, spanning 7B to 400B parameters (Llama 2, 3, and new Llama 4) with well-recognized performance. [SGLang provides Llama-4 model-specific optimizations](../basic_usage/llama4.md)  |
| **Mistral** (Mixtral, NeMo, Small3) | `mistralai/Mistral-7B-Instruct-v0.2`             | Open 7B LLM by Mistral AI with strong performance; extended into MoE (“Mixtral”) and NeMo Megatron variants for larger scale. |
| **Gemma** (v1, v2, v3)              | `google/gemma-3-1b-it`                            | Google’s family of efficient multilingual models (1B–27B); Gemma 3 offers a 128K context window, and its larger (4B+) variants support vision input. |
| **Phi** (Phi-1.5, Phi-2, Phi-3, Phi-4, Phi-MoE series) | `microsoft/Phi-4-multimodal-instruct`, `microsoft/Phi-3.5-MoE-instruct` | Microsoft’s Phi family of small models (1.3B–5.6B); Phi-4-multimodal (5.6B) processes text, images, and speech, Phi-4-mini is a high-accuracy text model and Phi-3.5-MoE is a mixture-of-experts model. |
| **MiniCPM** (v3, 4B)               | `openbmb/MiniCPM3-4B`                            | OpenBMB’s series of compact LLMs for edge devices; MiniCPM 3 (4B) achieves GPT-3.5-level results in text tasks. |
| **OLMo** (2, 3) | `allenai/OLMo-3-1125-32B`, `allenai/OLMo-3-32B-Think`, `allenai/OLMo-2-1124-7B-Instruct` | Allen AI’s series of Open Language Models designed to enable the science of language models. |
| **OLMoE** (Open MoE)               | `allenai/OLMoE-1B-7B-0924`                       | Allen AI’s open Mixture-of-Experts model (7B total, 1B active parameters) delivering state-of-the-art results with sparse expert activation. |
| **MiniMax-M2** (M2, M2.1)                     | `minimax/MiniMax-M2`, `minimax/MiniMax-M2.1`           | MiniMax’s SOTA LLM for coding & agentic workflows. |
| **StableLM** (3B, 7B)               | `stabilityai/stablelm-tuned-alpha-7b`            | StabilityAI’s early open-source LLM (3B & 7B) for general text generation; a demonstration model with basic instruction-following ability. |
| **Command-R** (Cohere)              | `CohereForAI/c4ai-command-r-v01`                 | Cohere’s open conversational LLM (Command series) optimized for long context, retrieval-augmented generation, and tool use. |
| **DBRX** (Databricks)              | `databricks/dbrx-instruct`                       | Databricks’ 132B-parameter MoE model (36B active) trained on 12T tokens; competes with GPT-3.5 quality as a fully open foundation model. |
| **Grok** (xAI)                     | `xai-org/grok-1`                                | xAI’s grok-1 model known for vast size(314B parameters) and high quality; integrated in SGLang for high-performance inference. |
| **ChatGLM** (GLM-130B family)       | `THUDM/chatglm2-6b`                              | Zhipu AI’s bilingual chat model (6B) excelling at Chinese-English dialogue; fine-tuned for conversational quality and alignment. |
| **InternLM 2** (7B, 20B)           | `internlm/internlm2-7b`                          | Next-gen InternLM (7B and 20B) from SenseTime, offering strong reasoning and ultra-long context support (up to 200K tokens). |
| **ExaONE 3** (Korean-English)      | `LGAI-EXAONE/EXAONE-3.5-7.8B-Instruct`           | LG AI Research’s Korean-English model (7.8B) trained on 8T tokens; provides high-quality bilingual understanding and generation. |
| **Baichuan 2** (7B, 13B)           | `baichuan-inc/Baichuan2-13B-Chat`                | BaichuanAI’s second-generation Chinese-English LLM (7B/13B) with improved performance and an open commercial license. |
| **XVERSE** (MoE)                   | `xverse/XVERSE-MoE-A36B`                         | Yuanxiang’s open MoE LLM (XVERSE-MoE-A36B: 255B total, 36B active) supporting ~40 languages; delivers 100B+ dense-level performance via expert routing. |
| **SmolLM** (135M–1.7B)            | `HuggingFaceTB/SmolLM-1.7B`                      | Hugging Face’s ultra-small LLM series (135M–1.7B params) offering surprisingly strong results, enabling advanced AI on mobile/edge devices. |
| **GLM-4** (Multilingual 9B)        | `ZhipuAI/glm-4-9b-chat`                          | Zhipu’s GLM-4 series (up to 9B parameters) – open multilingual models with support for 1M-token context and even a 5.6B multimodal variant (Phi-4V). |
| **MiMo** (7B series)               | `XiaomiMiMo/MiMo-7B-RL`                         | Xiaomi's reasoning-optimized model series, leverages Multiple-Token Prediction for faster inference. |
| **ERNIE-4.5** (4.5, 4.5MoE series) | `baidu/ERNIE-4.5-21B-A3B-PT`                    | Baidu's ERNIE-4.5 series which consists of MoE with 47B and 3B active parameters, with the largest model having 424B total parameters, as well as a 0.3B dense model. |
| **Arcee AFM-4.5B**               | `arcee-ai/AFM-4.5B-Base`                         | Arcee's foundational model series for real world reliability and edge deployments. |
| **Persimmon** (8B)               | `adept/persimmon-8b-chat`                         | Adept’s open 8B model with a 16K context window and fast inference; trained for broad usability and licensed under Apache 2.0. |
| **Solar** (10.7B)               | `upstage/SOLAR-10.7B-Instruct-v1.0`                         | Upstage's 10.7B parameter model, optimized for instruction-following tasks. This architecture incorporates a depth-up scaling methodology, enhancing model performance. |
| **Tele FLM** (52B-1T)               | `CofeAI/Tele-FLM`                         | BAAI & TeleAI's multilingual model, available in 52-billion and 1-trillion parameter variants. It is a decoder-only transformer trained on ~2T tokens |
| **Ling** (16.8B–290B) | `inclusionAI/Ling-lite`, `inclusionAI/Ling-plus` | InclusionAI’s open MoE models. Ling-Lite has 16.8B total / 2.75B active parameters, and Ling-Plus has 290B total / 28.8B active parameters. They are designed for high performance on NLP and complex reasoning tasks. |
| **Granite 3.0, 3.1** (IBM)               | `ibm-granite/granite-3.1-8b-instruct`                          | IBM's open dense foundation models optimized for reasoning, code, and business AI use cases. Integrated with Red Hat and watsonx systems. |
| **Granite 3.0 MoE** (IBM)               | `ibm-granite/granite-3.0-3b-a800m-instruct`                          | IBM’s Mixture-of-Experts models offering strong performance with cost-efficiency. MoE expert routing designed for enterprise deployment at scale. |
```
## Q2
如果我希望支持sglang  python -m sglang.launch_server时使用自定义模型呢
### Answer
要在 `python -m sglang.launch_server` 时使用自定义模型，主要通过 `--model-path` 参数指定模型路径，并确保自定义模型已正确注册到 SGLang 的模型注册系统中。 sglang:5-12 

## 基本使用方法

### 1. 指定模型路径
```bash
python3 -m sglang.launch_server \
  --model-path /path/to/your/custom/model \
  --host 0.0.0.0 \
  --port 30000
```

### 2. 使用 HuggingFace 模型ID
```bash
python3 -m sglang.launch_server \
  --model-path your-username/your-custom-model \
  --trust-remote-code
```

## 重要配置参数

### 必要参数
- `--model-path`: 指定自定义模型路径或 HuggingFace ID sglang:85-86 
- `--trust-remote-code`: 如果模型包含自定义代码，需要添加此标志 sglang:92-92 

### 常用可选参数
- `--model-impl`: 指定模型实现方式 (`auto`, `sglang`, `transformers`) sglang:97-97 
- `--tokenizer-path`: 单独指定分词器路径
- `--context-length`: 设置模型上下文长度

## 实际使用示例

### 标准模型启动
```bash
python3 -m sglang.launch_server \
  --model-path meta-llama/Llama-3.2-1B-Instruct \
  --host 0.0.0.0 \
  --port 30000
``` sglang:7-12 

### 多GPU部署
```bash
python3 -m sglang.launch_server \
  --model-path your-custom-model \
  --tp 8 \
  --trust-remote-code
``` sglang:28-32 

### 量化模型
```bash
python3 -m sglang.launch_server \
  --model-path your-custom-model \
  --quantization fp8 \
  --trust-remote-code
``` sglang:56-57 

## 不同平台的启动示例

### NVIDIA GPU
```bash
python3 -m sglang.launch_server \
  --model-path deepseek-ai/DeepSeek-V3 \
  --tp 16 \
  --trust-remote-code
``` sglang:256-256 

### AMD ROCm
```bash
python3 -m sglang.launch_server \
  --model-path meta-llama/Llama-3.1-8B-Instruct \
  --attention-backend triton
``` sglang:149-154 

### CPU部署
```bash
python -m sglang.launch_server \
  --model <MODEL_ID_OR_PATH> \
  --trust-remote-code \
  --device cpu \
  --host 0.0.0.0 \
  --tp 6
``` sglang:169-177 

## 注意事项

1. **模型注册**: 确保自定义模型已通过 `ModelRegistry` 注册或放置在正确的模型目录下
2. **依赖检查**: 验证所有必要的依赖库已安装
3. **内存配置**: 根据模型大小调整 `--mem-fraction-static` 参数 sglang:40-44 
4. **权限设置**: 对于 HuggingFace 模型，确保设置了正确的 `HF_TOKEN`

## Notes

- 启动服务器时会自动检测模型类型并选择合适的实现
- 如果遇到模型加载问题，可以尝试使用 `--model-impl transformers` 作为后备方案
- 对于大型模型，建议使用多GPU部署以提高性能

Wiki pages you might want to explore:
- [Installation and Deployment (sgl-project/sglang)](/wiki/sgl-project/sglang#2)
### Citations
**File:** docs/supported_models/generative_models.md (L5-12)
```markdown
## Example launch Command

```shell
python3 -m sglang.launch_server \
  --model-path meta-llama/Llama-3.2-1B-Instruct \  # example HF/local path
  --host 0.0.0.0 \
  --port 30000 \
```
```
**File:** docs/advanced_features/server_arguments.md (L28-32)
```markdown
- To enable multi-GPU tensor parallelism, add `--tp 2`. If it reports the error "peer access is not supported between these two devices", add `--enable-p2p-check` to the server launch command.

  ```bash
  python -m sglang.launch_server --model-path meta-llama/Meta-Llama-3-8B-Instruct --tp 2
  ```
```
**File:** docs/advanced_features/server_arguments.md (L40-44)
```markdown
- If you see out-of-memory errors during serving, try to reduce the memory usage of the KV cache pool by setting a smaller value of `--mem-fraction-static`. The default value is `0.9`.

  ```bash
  python -m sglang.launch_server --model-path meta-llama/Meta-Llama-3-8B-Instruct --mem-fraction-static 0.7
  ```
```
**File:** docs/advanced_features/server_arguments.md (L56-57)
```markdown
- To enable fp8 weight quantization, add `--quantization fp8` on a fp16 checkpoint or directly load a fp8 checkpoint without specifying any arguments.
- To enable fp8 kv cache quantization, add `--kv-cache-dtype fp8_e5m2`.
```
**File:** docs/advanced_features/server_arguments.md (L85-86)
```markdown
| `--model-path`<br>`--model` | The path of the model weights. This can be a local folder or a Hugging Face repo ID. | `None` | Type: str |
| `--tokenizer-path` | The path of the tokenizer. | `None` | Type: str |
```
**File:** docs/advanced_features/server_arguments.md (L92-92)
```markdown
| `--trust-remote-code` | Whether or not to allow for custom models defined on the Hub in their own modeling files. | `False` | bool flag (set to enable) |
```
**File:** docs/advanced_features/server_arguments.md (L97-97)
```markdown
| `--model-impl` | Which implementation of the model to use. * "auto" will try to use the SGLang implementation if it exists and fall back to the Transformers implementation if no SGLang implementation is available. * "sglang" will use the SGLang model implementation. * "transformers" will use the Transformers model implementation. | `auto` | Type: str |
```
**File:** benchmark/deepseek_v3/README.md (L256-256)
```markdown
python3 -m sglang.launch_server --model-path /path/to/DeepSeek-V3-BF16 --tp 32 --dist-init-addr 10.0.0.1:5000 --nnodes 4 --node-rank 0 --trust-remote-code --host 0.0.0.0 --port 30000
```
**File:** docs/platforms/amd_gpu.md (L149-154)
```markdown
    python3 -m sglang.launch_server \
    --model-path meta-llama/Meta-Llama-3.1-8B-Instruct \ # <- here
    --tp 8 \
    --trust-remote-code \
    --host 0.0.0.0 \
    --port 30000
```
**File:** docs/platforms/cpu_server.md (L169-177)
```markdown
```bash
python -m sglang.launch_server   \
    --model <MODEL_ID_OR_PATH>   \
    --trust-remote-code          \
    --disable-overlap-schedule   \
    --device cpu                 \
    --host 0.0.0.0               \
    --tp 6
```
```
## Q3
如何确定“自定义模型已正确注册到 SGLang 的模型注册系统中”？
### Answer
## 如何验证自定义模型是否正确注册到SGLang

### 程序化验证方法

#### 1. 检查ModelRegistry中的模型

```python
from sglang.srt.models.registry import ModelRegistry

# 查看所有已注册的模型架构
print("支持的模型架构:", ModelRegistry.get_supported_archs())

# 检查特定模型是否已注册
model_name = "YourCustomModel"
if model_name in ModelRegistry.models:
    print(f"模型 {model_name} 已注册")
    print(f"模型类: {ModelRegistry.models[model_name]}")
else:
    print(f"模型 {model_name} 未注册")
``` sglang:36-51 

#### 2. 使用resolve_model_cls方法测试

```python
from sglang.srt.models.registry import ModelRegistry

# 尝试解析模型类
try:
    model_cls, arch_name = ModelRegistry.resolve_model_cls(["YourCustomModel"])
    print(f"成功解析模型: {model_cls}, 架构名称: {arch_name}")
except ValueError as e:
    print(f"模型解析失败: {e}")
``` sglang:78-90 

### 实际验证步骤

#### 1. 检查模型文件的EntryClass

确保你的模型文件末尾有正确的EntryClass定义： sglang:2062-2064 

```python
# 在你的模型文件末尾
EntryClass = YourCustomModel
# 或对于多个模型类
EntryClass = [Model1, Model2]
```

#### 2. 验证外部注册（如果使用）

如果你使用外部注册方式，检查注册是否成功： sglang:115-136 

```python
from sglang.srt.models.registry import ModelRegistry

# 注册后立即验证
ModelRegistry.models["YourCustomModel"] = YourCustomModel
assert "YourCustomModel" in ModelRegistry.models
```

#### 3. 启动服务器验证

使用launch_server的实际测试：

```bash
# 尝试启动服务器，观察日志
python3 -m sglang.launch_server \
  --model-path /path/to/your/model \
  --trust-remote-code
```

成功启动的标志：
- 没有报错"Model architectures [...] are not supported"
- 日志显示成功加载模型
- 服务器正常启动并监听端口

### 调试技巧

#### 1. 启用详细日志

```python
import logging
logging.getLogger("sglang.srt.models.registry").setLevel(logging.DEBUG)
```

#### 2. 检查模型导入过程 sglang:92-125 

```python
# 手动测试模型导入
from sglang.srt.models.registry import import_model_classes

# 测试从sglang.srt.models包导入
models = import_model_classes("sglang.srt.models")
print(f"发现的模型: {list(models.keys())}")
```

### 常见问题排查

1. **EntryClass未定义** - 确保模型文件有EntryClass
2. **模块导入失败** - 检查模型文件位置和语法
3. **架构名称不匹配** - 确保config.json中的architectures字段与EntryClass名称一致
4. **外部注册时机** - 确保在launch_server调用前完成注册

## Notes

- SGLang的模型注册系统支持自动发现和手动注册两种方式 sglang:128-132 
- 对于多模态模型，还需要检查多模态处理器的注册情况
- 使用`--model-impl transformers`可以作为后备方案来验证模型本身是否可用
### Citations
**File:** python/sglang/srt/models/registry.py (L36-51)
```python
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
```
**File:** python/sglang/srt/models/registry.py (L78-90)
```python
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
**File:** python/sglang/srt/models/registry.py (L92-125)
```python
@lru_cache()
def import_model_classes(package_name: str, strict: bool = False):
    model_arch_name_to_cls = {}
    package = importlib.import_module(package_name)
    for _, name, ispkg in pkgutil.iter_modules(package.__path__, package_name + "."):
        if not ispkg:
            if name.split(".")[-1] in envs.SGLANG_DISABLED_MODEL_ARCHS.get():
                logger.debug(f"Skip loading {name} due to SGLANG_DISABLED_MODEL_ARCHS")
                continue

            try:
                module = importlib.import_module(name)
            except Exception as e:
                if strict:
                    raise
                logger.warning(f"Ignore import error when loading {name}: {e}")
                continue
            if hasattr(module, "EntryClass"):
                entry = module.EntryClass
                if isinstance(
                    entry, list
                ):  # To support multiple model classes in one module
                    for tmp in entry:
                        assert (
                            tmp.__name__ not in model_arch_name_to_cls
                        ), f"Duplicated model implementation for {tmp.__name__}"
                        model_arch_name_to_cls[tmp.__name__] = tmp
                else:
                    assert (
                        entry.__name__ not in model_arch_name_to_cls
                    ), f"Duplicated model implementation for {entry.__name__}"
                    model_arch_name_to_cls[entry.__name__] = entry

    return model_arch_name_to_cls
```
**File:** python/sglang/srt/models/registry.py (L128-132)
```python
ModelRegistry = _ModelRegistry()
ModelRegistry.register("sglang.srt.models")

if external_pkg := envs.SGLANG_EXTERNAL_MODEL_PACKAGE.get():
    ModelRegistry.register(external_pkg, overwrite=True)
```
**File:** python/sglang/srt/models/deepseek_janus_pro.py (L2062-2064)
```python
AutoModel.register(config_class=MultiModalityConfig, model_class=MultiModalityCausalLM)
EntryClass = [MultiModalityCausalLM]

```
**File:** docs/supported_models/support_new_models.md (L115-136)
```markdown
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
## Q4
那么如何编写自定义模块，用来实现pip install custom_module之后就完成了登记注册呢
### Answer
## 创建pip安装后自动注册的自定义模型模块

要创建一个通过 `pip install custom_module` 安装后自动完成模型注册的模块，你需要利用SGLang的外部模型包发现机制。

### 核心原理

SGLang通过环境变量 `SGLANG_EXTERNAL_MODEL_PACKAGE` 来发现和加载外部模型包 sglang:131-132 。当设置此环境变量后，SGLang会自动扫描指定包中的模型模块。

### 完整实现方案

#### 1. 创建Python包结构

```
custom_sglang_models/
├── setup.py
├── pyproject.toml
├── custom_sglang_models/
│   ├── __init__.py
│   ├── models/
│   │   ├── __init__.py
│   │   ├── my_custom_model.py
│   │   └── another_model.py
│   └── processors/
│       ├── __init__.py
│       └── my_processor.py
```

#### 2. 实现模型文件

在 `custom_sglang_models/models/my_custom_model.py` 中：

```python
import torch
from sglang.srt.models.llama import LlamaForCausalLM
from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.layers.quantization.base_config import QuantizationConfig

class MyCustomModel(LlamaForCausalLM):
    def __init__(
        self,
        config,
        quant_config: QuantizationConfig = None,
        prefix: str = "",
    ):
        super().__init__(config=config, quant_config=quant_config, prefix=prefix)
    
    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch,
        pp_proxy_tensors=None,
        input_embeds: torch.Tensor = None,
        get_embedding: bool = False,
    ) -> LogitsProcessorOutput:
        # 调用父类实现
        hidden_states = self.model(
            input_ids,
            positions,
            forward_batch,
            input_embeds,
            pp_proxy_tensors=pp_proxy_tensors,
        )
        
        res = self.logits_processor(
            input_ids,
            hidden_states,
            self.lm_head,
            forward_batch,
        )
        
        # 自定义逻辑
        res.next_token_logits = res.next_token_logits * 0.9
        
        return res

# 关键：定义EntryClass
EntryClass = MyCustomModel
```

#### 3. 配置setup.py

```python
from setuptools import setup, find_packages

setup(
    name="custom_sglang_models",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "sglang",
        "torch",
        "transformers",
    ],
    # 使用entry_points自动设置环境变量
    entry_points={
        'console_scripts': [
            'setup-custom-sglang=custom_sglang_models.setup:main',
        ],
    },
    python_requires=">=3.8",
)
```

#### 4. 创建自动设置脚本

在 `custom_sglang_models/setup.py` 中：

```python
import os
import sys

def main():
    """设置环境变量以启用SGLang外部模型发现"""
    package_name = "custom_sglang_models"
    
    # 获取当前安装的包路径
    import custom_sglang_models
    package_path = os.path.dirname(custom_sglang_models.__file__)
    
    # 设置环境变量
    env_var = "SGLANG_EXTERNAL_MODEL_PACKAGE"
    current_value = os.environ.get(env_var, "")
    
    if package_name not in current_value:
        if current_value:
            new_value = f"{current_value},{package_name}"
        else:
            new_value = package_name
        
        os.environ[env_var] = new_value
        print(f"设置 {env_var}={new_value}")
        print("请将此环境变量添加到您的shell配置文件中")
        print(f"export {env_var}='{new_value}'")

if __name__ == "__main__":
    main()
```

#### 5. 创建包初始化文件

在 `custom_sglang_models/__init__.py` 中：

```python
"""
自定义SGLang模型包

安装后请运行：
python -m custom_sglang_models.setup

或手动设置环境变量：
export SGLANG_EXTERNAL_MODEL_PACKAGE=custom_sglang_models
"""

__version__ = "0.1.0"

# 可选：在导入时自动注册
try:
    from sglang.srt.models.registry import ModelRegistry
    from .models.my_custom_model import MyCustomModel
    from .models.another_model import AnotherModel
    
    # 动态注册模型
    ModelRegistry.models["MyCustomModel"] = MyCustomModel
    ModelRegistry.models["AnotherModel"] = AnotherModel
    
except ImportError:
    # SGLang未安装时的处理
    pass
```

### 安装和使用流程

#### 1. 安装包

```bash
pip install custom_sglang_models
```

#### 2. 设置环境变量

```bash
# 方法1：使用提供的脚本
python -m custom_sglang_models.setup

# 方法2：手动设置
export SGLANG_EXTERNAL_MODEL_PACKAGE=custom_sglang_models

# 方法3：添加到shell配置文件
echo 'export SGLANG_EXTERNAL_MODEL_PACKAGE=custom_sglang_models' >> ~/.bashrc
source ~/.bashrc
```

#### 3. 使用自定义模型

```bash
python3 -m sglang.launch_server \
  --model-path /path/to/your/model \
  --trust-remote-code
```

### 高级配置

#### 支持多模态模型

如果包含多模态模型，还需要设置多模态处理器包：

```python
# 在setup.py中
env_vars = {
    "SGLANG_EXTERNAL_MODEL_PACKAGE": "custom_sglang_models",
    "SGLANG_EXTERNAL_MM_PROCESSOR_PACKAGE": "custom_sglang_models",
}
```

#### 使用pyproject.toml（现代方式）

```toml
[build-system]
requires = ["setuptools>=61.0", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "custom_sglang_models"
version = "0.1.0"
description = "Custom models for SGLang"
dependencies = [
    "sglang",
    "torch",
    "transformers",
]

[project.scripts]
setup-custom-sglang = "custom_sglang_models.setup:main"
```

### 验证安装

安装后可以通过以下方式验证：

```python
from sglang.srt.models.registry import ModelRegistry

# 检查模型是否已注册
print("已注册的模型:", list(ModelRegistry.models.keys()))

# 测试模型解析
try:
    model_cls, arch_name = ModelRegistry.resolve_model_cls(["MyCustomModel"])
    print(f"成功解析: {model_cls}")
except ValueError as e:
    print(f"解析失败: {e}")
```

## Notes

- SGLang的模型发现机制依赖于 `EntryClass` 属性 sglang:109-124 
- 环境变量 `SGLANG_EXTERNAL_MODEL_PACKAGE` 支持多个包，用逗号分隔
- 对于多模态模型，还需要设置 `SGLANG_EXTERNAL_MM_PROCESSOR_PACKAGE` 环境变量 sglang:14-15 
- 这种方式无需修改SGLang源码，完全通过外部扩展实现
### Citations
**File:** python/sglang/srt/models/registry.py (L109-124)
```python
            if hasattr(module, "EntryClass"):
                entry = module.EntryClass
                if isinstance(
                    entry, list
                ):  # To support multiple model classes in one module
                    for tmp in entry:
                        assert (
                            tmp.__name__ not in model_arch_name_to_cls
                        ), f"Duplicated model implementation for {tmp.__name__}"
                        model_arch_name_to_cls[tmp.__name__] = tmp
                else:
                    assert (
                        entry.__name__ not in model_arch_name_to_cls
                    ), f"Duplicated model implementation for {entry.__name__}"
                    model_arch_name_to_cls[entry.__name__] = entry

```
**File:** python/sglang/srt/models/registry.py (L131-132)
```python
if external_pkg := envs.SGLANG_EXTERNAL_MODEL_PACKAGE.get():
    ModelRegistry.register(external_pkg, overwrite=True)
```
**File:** test/registered/model_loading/test_external_models.py (L14-15)
```python
        envs.SGLANG_EXTERNAL_MODEL_PACKAGE.set("sglang.test.external_models")
        envs.SGLANG_EXTERNAL_MM_PROCESSOR_PACKAGE.set("sglang.test.external_models")
```