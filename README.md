# SGLang Qwen3 Next Plugin

这是一个 SGLang 插件，用于自定义修改 Qwen3 Next 模型的推理行为。通过插件机制，可以在不修改 SGLang 源码的情况下替换原有的模型实现。

## 功能特性

- ✅ 基于 SGLang v0.5.2 的 Qwen3 Next 实现
- ✅ 插件化架构，无需修改 SGLang 源码
- ✅ 支持通过环境变量自动加载
- ✅ 可扩展的自定义修改接口

## 安装

```bash
# 开发模式安装
pip install -e .

# 或者常规安装
pip install .
```

## 使用方法

### 方式 1: 环境变量自动加载（推荐）

设置环境变量后，SGLang 会自动加载插件：

```bash
export SGLANG_EXTERNAL_MODEL_PACKAGE="sglang_qwen3_next_plugin"

# 启动 SGLang 服务器
python3 -m sglang.launch_server \
  --model <your_qwen3_next_model_path> \
  --tp 1
```

启动时，您应该会看到绿色的成功提示信息，表明插件已成功加载：

```
================================================================================
  ✓ SGLang Qwen3 Next Plugin Loaded Successfully!
  ✓ Custom model implementation is active.
================================================================================
```

### 方式 2: 手动注册

在代码中手动注册插件模型：

```python
from sglang.srt.models.registry import ModelRegistry
from sglang_qwen3_next_plugin import Qwen3NextForCausalLM

# 注册模型
ModelRegistry.models["Qwen3NextForCausalLM"] = Qwen3NextForCausalLM

# 然后正常使用 SGLang
import sglang as sgl
llm = sgl.Engine(model_path="./path/to/model")
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

当前插件在 `Qwen3NextForCausalLM.__init__` 方法中添加了彩色打印，作为插件加载成功的标识。

要添加更多自定义修改，请编辑：
- `sglang_qwen3_next_plugin/_0_5_2/qwen3_next.py`

## 项目结构

```
sglang_qwen3_next_plugin/
├── pyproject.toml          # 项目配置
├── README.md               # 本文件
├── plan.md                 # 开发计划
├── sglang_qwen3_next_plugin/
│   ├── __init__.py         # 包初始化，模型注册
│   └── _0_5_2/
│       ├── __init__.py
│       └── qwen3_next.py  # 自定义的 Qwen3 Next 模型实现
```

## 开发

### 添加自定义功能

1. 编辑 `sglang_qwen3_next_plugin/_0_5_2/qwen3_next.py`
2. 在 `Qwen3NextForCausalLM` 类中添加您的自定义逻辑
3. 重新安装插件：`pip install -e .`
4. 测试您的修改

### 调试

如果插件未加载，请检查：

1. 环境变量是否正确设置
2. 插件包是否正确安装
3. 模型的 `config.json` 中 `architectures` 字段是否正确
4. SGLang 版本是否兼容（当前基于 v0.5.2）

## 版本兼容性

- SGLang: >= 0.5.0
- Python: >= 3.10

## 许可证

请参考 SGLang 的许可证要求。
