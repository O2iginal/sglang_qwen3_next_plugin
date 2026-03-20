# SGLang Qwen3 Next Plugin

这是一个 SGLang 插件，用于自定义修改 Qwen3 Next 模型的推理行为。通过插件机制，可以在不修改 SGLang 源码的情况下替换原有的模型实现。

## 功能特性

- ✅ 基于 SGLang v0.5.2 的 Qwen3 Next 实现
- ✅ 插件化架构，无需修改 SGLang 源码
- ✅ 主入口支持 `SGLANG_EXTERNAL_MODEL_PACKAGE=sglang_qwen3_next_plugin`
- ✅ 已接通 checkpoint 显式 `layer_types`
- ✅ 已修复 Hybrid GDN 在线性注意力路径上的关键运行时兼容问题
- ✅ 提供可复用的生成验收脚本

## 安装

```bash
# 开发模式安装
pip install -e .

# 或者常规安装
pip install .
```

## 使用方法

### 环境变量 + 原始命令

当前推荐并作为主验收标准的启动方式：

```bash
source /path/to/venv/bin/activate
export SGLANG_EXTERNAL_MODEL_PACKAGE="sglang_qwen3_next_plugin"

python -m sglang.launch_server \
  --model-path <your_qwen3_next_model_path> \
  --port 30110 \
  --tp 1
```

注意：
- 环境变量必须在 Python 进程启动前设置
- 当前仓库中仍保留少量接入层兼容逻辑，因为目标环境里的 `sglang 0.5.2` 本地实现并不会直接消费这个环境变量
- 若未来上游版本原生支持外部模型包，应优先删除兼容层并回归上游机制

启动时，您应该看到绿色成功提示。当前实现会做到“每个进程只打印一次”，避免多层重复刷屏。

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

本插件当前包含两类修改：

1. 结构性 checkpoint 对齐
   - 将 `GemmaRMSNorm` 替换为普通 `RMSNorm`
   - 移除 `q_norm` / `k_norm`
   - `qkv_proj` 开启 bias，`o_proj` 保持无 bias
   - `attn_output_gate` 默认按 checkpoint 需要关闭
   - `num_experts=0` 时走普通 MLP 路径

2. SGLang 0.5.2 运行时兼容补丁
   - 显式接通 `layer_types`，避免错误依赖 `full_attention_interval`
   - 将 Hybrid GDN backend 切到稳定 `causal_conv1d_fn`
   - 在 `ModelRunner.init_memory_pool()` 同步真实运行 dtype，修复 `conv_state` 分配类型错误

详细原因与升级复核点请看：
- `progress.md`
- `changelog.md`

## 项目结构

```
sglang_qwen3_next_plugin/
├── pyproject.toml          # 项目配置
├── README.md               # 本文件
├── progress.md             # 中文进度记录
├── changelog.md            # 相对上游的变更与升级复核点
├── sitecustomize.py        # 环境变量主入口接入层
├── scripts/
│   ├── check_plugin_import.py
│   ├── run_acceptance.py
│   ├── validate_generation.py
│   └── validate_logprob.py
├── sglang_qwen3_next_plugin/
│   ├── __init__.py         # 仅导出 EntryClass
│   ├── env_override.py     # 插件发现兼容层
│   ├── qwen3_next.py       # 当前插件实现
│   └── qwen3_next_upstream.py
└── docs/
    └── superpowers/
```

说明：
- 当前文档以“主入口 + 上游基线 + 插件实现 + 中文记录”这套结构为准
- 历史试验目录不再作为当前方案依赖

## 验证

### 验证插件导入契约

```bash
source /path/to/venv/bin/activate
XDG_CACHE_HOME=/tmp HOME=/tmp python test_plugin.py
```

### 验证插件是否接管 registry

```bash
source /path/to/venv/bin/activate
SGLANG_EXTERNAL_MODEL_PACKAGE=sglang_qwen3_next_plugin \
XDG_CACHE_HOME=/tmp HOME=/tmp \
python scripts/check_plugin_import.py
```

### 验证自然语言生成

服务启动后，执行：

```bash
source /path/to/venv/bin/activate
python scripts/validate_generation.py --host 127.0.0.1 --port 30110
```

该脚本会发送中英文最小样例并输出：
- 生成文本
- `output_ids` 前缀
- `PASS` / `FAIL`

### 验证 logprob / 近似 lm loss

服务启动后，执行：

```bash
source /path/to/venv/bin/activate
python scripts/validate_logprob.py --host 127.0.0.1 --port 30110
```

该脚本会：
- 请求服务返回 prompt token 的 logprob
- 计算平均负对数似然 `avg_nll`
- 给出一个宽松阈值判断，排除“模型路径明显坏掉”的情况

当前实测量级：
- 中文样例：`avg_nll ≈ 4.30`
- 英文样例：`avg_nll ≈ 3.11`

### 一条命令串联验收

仅做本地契约与 registry 检查：

```bash
source /path/to/venv/bin/activate
python scripts/run_acceptance.py --skip-runtime
```

对已启动服务做完整验收：

```bash
source /path/to/venv/bin/activate
python scripts/run_acceptance.py --host 127.0.0.1 --port 30110
```

## 开发

### 添加自定义功能

1. 编辑 `sglang_qwen3_next_plugin/qwen3_next.py`
2. 在 `Qwen3NextForCausalLM` 类中添加您的自定义逻辑
3. 确保文件末尾有 `EntryClass = Qwen3NextForCausalLM` 定义
4. 重新安装插件：`pip install -e .`
5. 测试您的修改

### 调试

如果插件未加载，请检查：

1. 插件包是否已重新安装：`pip install -e .`
2. 环境变量是否在 Python 启动前设置
3. `config.json` 的 `architectures` 是否为 `["Qwen3NextForCausalLM"]`
4. `scripts/check_plugin_import.py` 是否显示 registry 已指向插件实现
5. `scripts/validate_generation.py` 与 `scripts/validate_logprob.py` 是否仍返回 `PASS`

说明：
- 当前 README 只保留真实仍在使用的链路
- 更细的排障过程和根因链请看 `progress.md`

## 版本兼容性

- SGLang: >= 0.5.0
- Python: >= 3.10

## 许可证

请参考 SGLang 的许可证要求。
