# SGLang Qwen3 Next Plugin 设计文档

## 目标

围绕用户提供的测试环境与测试 checkpoint，构建一个可长期维护的 SGLang 外部插件，使其通过原生插件入口接管 `Qwen3NextForCausalLM` 的实现，并在 `sglang==0.5.2` 下完成正确的自然语言推理。

本设计不以当前仓库中的历史 patch 为真值，而是以目标 venv 中的上游文件为起点：

- 上游基线文件：`/mnt/ssd/yulan/pretrain-linear-moe-dev-worktree/slime/.venv/lib/python3.12/site-packages/sglang/srt/models/qwen3_next.py`

## 范围

本轮工作覆盖以下内容：

- 以上游 `qwen3_next.py` 为基线重建插件结构
- 通过 `SGLANG_EXTERNAL_MODEL_PACKAGE=sglang_qwen3_next_plugin` 验证原生插件发现机制
- 逐条恢复并论证必要的模型结构定制
- 将运行时兼容补丁与模型语义改动分层管理
- 使用测试 checkpoint 进行生成验证，必要时补充 `lm loss` 验证
- 输出中文进度文档与中文变更记录文档，支撑未来 SGLang 版本升级

不在本轮范围内的事项：

- 保持历史 patch 完全不变
- 以包装脚本替代原生插件入口作为主方案
- 引入与当前问题无关的大规模重构

## 总体方案

### 1. 代码组织

代码分为三层：

- `sglang_qwen3_next_plugin/qwen3_next_upstream.py`
  - 保存从目标 venv 拷出的上游 `qwen3_next.py`
  - 仅允许最小必要的注释或导入适配
  - 不承载用户自定义逻辑
- `sglang_qwen3_next_plugin/qwen3_next.py`
  - 作为插件实际导出的模型实现
  - 基于上游基线恢复必要定制
  - 每一类偏离上游的改动都必须有清晰注释，并同步记录到 `changelog.md`
- `sglang_qwen3_next_plugin/__init__.py`
  - 只负责暴露插件入口与 `EntryClass`
  - 避免混入不透明的自动注册、退出钩子、全局 monkey patch

### 2. 改动分层

所有改动按优先级分三层处理：

- `P0` 插件接管层
  - 只验证 SGLang 能发现并实例化插件模型
  - 不改变模型数学行为
- `P1` 模型结构定制层
  - 只保留与用户 checkpoint 结构确实对应的改动
  - 例如 LayerNorm 类型、Q/K norm、bias 配置、MoE/MLP 路径差异等
- `P2` 运行时兼容层
  - 只在 `P1` 自洽后仍有问题时引入
  - 例如某些运行时 guard、SGLang 接口兼容、最小必要的补丁
  - 默认优先避免全局 monkey patch

### 3. 原生插件入口

主验收入口固定为：

```bash
export SGLANG_EXTERNAL_MODEL_PACKAGE=sglang_qwen3_next_plugin
python -m sglang.launch_server --model-path <checkpoint> --tp 1
```

包装脚本可以保留为调试辅助工具，但不能作为主方案成立的前提。

## 验证设计

验证按四级执行：

### 1. 插件发现与接管验证

- 插件包可在目标 venv 中安装
- 启动时 SGLang 确实导入插件包
- 被实例化的是插件中的 `Qwen3NextForCausalLM`
- 通过轻量标识避免误以为仍在使用上游实现

### 2. 权重与结构对齐验证

- 权重加载不出现关键参数缺失或形状不匹配
- 若确有需要跳过的参数，必须明确记录，不允许静默吞掉
- 逐项解释每类结构定制为什么与 checkpoint 对应

### 3. 自然语言生成验证

- 对测试 checkpoint 发起多条自然语言 prompt
- 输出应为可读自然语言，而非乱码、异常重复、空输出或明显坏掉的 token 模式

### 4. `lm loss` 兜底验证

- 当生成结果不足以稳定判断时，补充自然语言样本的 `lm loss`
- 记录样本、配置与结果，形成可复用基线

## 文档设计

### 1. `progress.md`

持续记录每个主要阶段的：

- 目标
- 现状
- 关键发现
- 下一步

### 2. `changelog.md`

持续记录每个偏离上游的改动：

- 改动点
- 原因
- 是否属于 `P1` 或 `P2`
- 对 checkpoint 的对应关系
- 升级 SGLang 时的复核方法

## 风险与原则

### 风险

- 当前仓库存在历史调试残留，可能混淆真正必要的改动
- 自定义 checkpoint 可能与上游实现有多处结构差异，不宜一次性全部恢复
- 某些问题可能来自 SGLang 运行时，而非模型定义本身

### 原则

- 以上游基线为真值，不以历史 patch 为真值
- 先证明插件接管，再证明模型正确
- 每次只引入一类变化，并立刻验证
- 所有偏离上游的逻辑都必须可解释、可记录、可回归
