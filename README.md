# SGLang Qwen3 Next Plugin

这是一个 SGLang 插件，用于自定义修改 Qwen3 Next 模型的推理行为。通过插件机制，可以在不修改 SGLang 源码的情况下替换原有的模型实现。

## 功能特性

- ✅ 基于 SGLang v0.5.2 的 Qwen3 Next 实现
- ✅ 插件化架构，无需修改 SGLang 源码
- ✅ 支持多种加载方式（包装脚本、环境变量、手动注册）
- ✅ 自动修复常见配置问题（expert location、full_attention_interval）
- ✅ 可扩展的自定义修改接口

## 安装

```bash
# 开发模式安装
pip install -e .

# 或者常规安装
pip install .
```

## 使用方法

### 方式 1: 使用包装脚本（推荐，最简单）

**推荐使用此方式，无需设置环境变量，插件会自动注册！**

```bash
# 直接使用包装脚本启动
python -m sglang_qwen3_next_plugin.launch_server \
  --model-path <your_qwen3_next_model_path> \
  --port 30110 \
  --tp 1
```

**完整示例：**
```bash
source /path/to/venv/bin/activate
hf_path=/path/to/your/model
python -m sglang_qwen3_next_plugin.launch_server \
  --model-path $hf_path \
  --port 30110 \
  --tp 1
```

启动时，您应该会看到绿色的成功提示信息，表明插件已成功加载：

```
================================================================================
  ✓ SGLang Qwen3 Next Plugin Loaded Successfully!
  ✓ Custom model implementation is active.
================================================================================
```

### 方式 2: 环境变量 + 原始命令

如果希望使用原始的 `sglang.launch_server` 命令：

```bash
# 设置环境变量（必须在导入 SGLang 之前）
export SGLANG_EXTERNAL_MODEL_PACKAGE="sglang_qwen3_next_plugin"

# 启动 SGLang 服务器
python3 -m sglang.launch_server \
  --model-path <your_qwen3_next_model_path> \
  --port 30110 \
  --tp 1
```

**注意**：环境变量必须在启动命令的同一 shell 会话中设置，或者添加到 `~/.bashrc` 或 `~/.zshrc` 中。

### 方式 3: 手动注册（用于调试或编程使用）

在代码中手动注册插件模型：

```python
import sglang_qwen3_next_plugin
from sglang_qwen3_next_plugin import register

# 注册模型（会自动覆盖原始模型）
register()

# 然后正常使用 SGLang
from sglang.srt.models.registry import ModelRegistry
# 验证注册
assert "Qwen3NextForCausalLM" in ModelRegistry.models
```

## 模型配置要求

确保您的模型目录中的 `config.json` 包含正确的架构名称：

```json
{
  "architectures": [
    "Qwen3NextForCausalLM"
  ],
  ...
}
```

## 自定义修改

本插件对 Qwen3 Next 模型进行了以下自定义修改（所有修改都在代码中标注了 `MODIFIED` 注释）：

1. **LayerNorm 替换**: 将 `GemmaRMSNorm` 替换为普通 `RMSNorm`
2. **移除 QK Norm**: 删除了 `q_norm` 和 `k_norm` 相关逻辑和参数
3. **Attention Bias 设置**: 
   - `qkv_proj` 默认开启 `bias=True`
   - `o_proj` 默认关闭 `bias=False`
4. **支持普通 MLP**: 当 `num_experts=0` 时，使用普通 MLP 而非 MoE
5. **默认关闭 Gate**: `attn_output_gate` 默认设置为 `False`
6. **修复 Expert Location**: 当 `num_experts=0` 时，`num_logical_experts` 设置为 1，避免空数组错误
7. **修复 full_attention_interval**: 当 `full_attention_interval=0` 时，自动修复以避免除零错误

所有修改都在 `sglang_qwen3_next_plugin/qwen3_next.py` 文件中，并在 `Qwen3NextForCausalLM.__init__` 方法中添加了彩色打印，作为插件加载成功的标识。

## 项目结构

```
sglang_qwen3_next_plugin/
├── pyproject.toml          # 项目配置
├── README.md               # 本文件
├── plan.md                 # 开发计划
├── sglang_qwen3_next_plugin/
│   ├── __init__.py         # 包初始化，包含自动注册逻辑
│   ├── qwen3_next.py       # 自定义的 Qwen3 Next 模型实现（包含 EntryClass）✅
│   ├── launch_server.py    # 包装脚本，用于自动注册插件
│   ├── models/             # 保留用于参考
│   │   ├── __init__.py
│   │   └── qwen3_next.py
│   └── _0_5_2/            # 旧版本目录（保留用于参考）
│       ├── __init__.py
│       └── qwen3_next.py
```

**重要**：模型文件 `qwen3_next.py` 必须放在包根目录下，因为 SGLang 的 `import_model_classes` 函数只会扫描包的直接子模块，不会递归扫描嵌套的子包。文件中的 `EntryClass` 会被自动发现。

## 开发

### 添加自定义功能

1. 编辑 `sglang_qwen3_next_plugin/qwen3_next.py`
2. 在 `Qwen3NextForCausalLM` 类中添加您的自定义逻辑
3. 确保文件末尾有 `EntryClass = Qwen3NextForCausalLM` 定义
4. 重新安装插件：`pip install -e .`
5. 测试您的修改

### 调试

如果插件未加载，请检查：

1. **使用包装脚本（推荐）**：使用 `python -m sglang_qwen3_next_plugin.launch_server` 而不是原始命令
2. **插件包是否正确安装**：运行 `pip install -e .` 重新安装
3. **模型文件位置**：确保 `qwen3_next.py` 在包根目录下（`sglang_qwen3_next_plugin/qwen3_next.py`）
4. **EntryClass 定义**：确保 `qwen3_next.py` 文件末尾有 `EntryClass = Qwen3NextForCausalLM`
5. **模型的 `config.json`**：确保 `architectures` 字段为 `["Qwen3NextForCausalLM"]`
6. **SGLang 版本**：当前基于 v0.5.2

**验证插件是否加载：**

启动时应该看到绿色的成功提示：
```
================================================================================
  ✓ SGLang Qwen3 Next Plugin Loaded Successfully!
  ✓ Custom model implementation is active.
================================================================================
```

如果看到这个提示，说明插件已成功加载。

**为什么模型文件要在包根目录？**

SGLang 的 `import_model_classes` 函数使用 `pkgutil.iter_modules` 扫描包，它只会扫描**直接子模块**，不会递归扫描嵌套的子包，也不会包含 `__init__.py`。因此：
- ✅ `sglang_qwen3_next_plugin.qwen3_next` 会被扫描到（包根目录）
- ❌ `sglang_qwen3_next_plugin.models.qwen3_next` 不会被扫描到（`models` 是子包）
- ❌ `sglang_qwen3_next_plugin._0_5_2.qwen3_next` 不会被扫描到（`_0_5_2` 是子包）

## 版本兼容性

- SGLang: >= 0.5.0
- Python: >= 3.10

## 许可证

请参考 SGLang 的许可证要求。
