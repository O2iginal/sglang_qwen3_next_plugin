# SGLang Qwen3 Next 多版本插件适配设计文档

## 目标

在当前已经适配 `sglang==0.5.2` 的基础上，把仓库升级为可维护的多版本插件工程，使其能够继续支持旧版本，同时为新环境中的官方 `sglang` 主包提供同样的 Qwen3 Next 自定义能力。

本轮设计的核心不是“再打一轮补丁”，而是把以下两类逻辑彻底拆开：

- 与用户 checkpoint 结构绑定的公共自定义逻辑
- 与某个特定 `sglang` 版本实现缺口绑定的兼容补丁

此外，本轮还包含一个工程化目标：

- 在仓库中新增一个项目级 skill，用于指导未来新增 `sglang` 版本时如何抽取上游基线、分类补丁、更新文档并执行验收

## 已知前提

- 现有仓库已经在 `sglang==0.5.2` 下跑通真实生成与 logprob 验收。
- 现有 `0.5.2` 适配中包含两类内容：
  - checkpoint 结构性定制
  - `0.5.2` 运行时兼容补丁
- 用户指定的新目标环境为：
  - `/mnt/ssd/yulan/pretrain-linear-moe-dev-worktree/slime/.venv-sgl0_5_9`
- 该环境当前尚未安装 `sglang` 主包，只装有 `sgl_kernel` 与 `sglang_router`，因此必须先安装官方 `sglang` 主包，再以其真实代码为新版本基线。

## 范围

本轮工作包括：

- 安装并核实新环境中的官方 `sglang` 主包
- 提取新版本的上游 `qwen3_next.py` 与相关配置基线
- 将仓库重组为“上游基线层 + 公共自定义层 + 版本兼容层 + 入口装配层”
- 明确区分哪些补丁仅属于 `0.5.2`
- 为新版本补齐必要但最小的 compat
- 更新中文文档与验收脚本，使其能说明当前命中的版本和补丁
- 新增项目级 skill，固化未来升级 `sglang` 版本的流程

本轮工作不包括：

- 将不同版本拆成多个独立仓库
- 把所有历史 patch 原封不动复制到新版本
- 以包装启动脚本取代 `python -m sglang.launch_server`

## 总体方案

### 1. 仓库分层

建议将仓库逻辑拆成四层。

#### 上游基线层

保存不同 `sglang` 版本的原始模型实现快照，只做最小导入修正，不承载你的自定义逻辑。

建议路径：

- `sglang_qwen3_next_plugin/upstream/sglang_0_5_2/...`
- `sglang_qwen3_next_plugin/upstream/sglang_0_5_9/...`

必要时一并保存：

- `qwen3_next.py`
- `configs/qwen3_next.py`
- 其他直接影响模型构造与配置解析的上游文件

#### 公共自定义层

只放与你的 checkpoint 架构绑定、原则上跨版本都应成立的逻辑，例如：

- `GemmaRMSNorm -> RMSNorm`
- 去掉 `q_norm` / `k_norm`
- 投影层 bias 配置变化
- `attn_output_gate` 默认值变化
- `num_experts=0` 时切到普通 MLP 路径

这层的目标是尽量不依赖具体 `sglang` 版本细节。

#### 版本兼容层

只放某个 `sglang` 版本的运行时缺口或接口缺口补丁，例如：

- `layer_types` 未被正确保存或消费
- `full_attention_interval=0` 导致错误推导
- `ModelRunner.init_memory_pool()` dtype 同步问题
- Hybrid linear attention backend 需要切换稳定实现
- 插件发现机制的版本差异

建议按版本拆成独立模块：

- `sglang_qwen3_next_plugin/compat/sglang_0_5_2.py`
- `sglang_qwen3_next_plugin/compat/sglang_0_5_9.py`

#### 入口装配层

包根负责：

- 检测当前 `sglang.__version__`
- 选择对应上游基线
- 应用公共自定义层
- 应用当前版本 compat
- 统一导出 `EntryClass`

这样后续升级时，不需要重写整个插件，只需要新增：

- 一个新版本上游快照目录
- 一个新的 compat 模块
- 一组针对该版本的事实结论

### 2. 补丁分类原则

所有现有补丁都必须重新归因，不能因为它在 `0.5.2` 中工作过，就默认继续保留。

判定规则：

- 如果改动描述的是 checkpoint 的真实结构或参数语义，则归入公共自定义层
- 如果改动只是在修补某个 `sglang` 版本的配置解析、运行时 dtype、kernel/backend 或插件接入缺口，则归入版本 compat

当前已知可优先视为公共自定义层的候选项：

- `GemmaRMSNorm -> RMSNorm`
- 去掉 `q_norm` / `k_norm`
- 投影层 bias 变化
- `attn_output_gate` 变化
- `num_experts=0` 走普通 MLP

当前已知可优先视为 `0.5.2` compat 的候选项：

- 显式保存并消费 `layer_types`
- 删除旧的 `full_attention_interval=1` 错误兜底
- Hybrid backend 切到稳定 `causal_conv1d_fn`
- `ModelRunner.init_memory_pool()` dtype 同步
- 插件发现的 env 接入兼容层

### 3. 新版本适配策略

本轮不直接把 `0.5.2` 的 compat 全量移植到新版本，而是按以下顺序推进：

1. 在新环境安装官方 `sglang` 主包
2. 记录真实版本号、模型文件路径、配置文件路径、插件入口能力
3. 抽取新版本上游基线
4. 先只移植公共自定义层
5. 在不带旧 compat 的情况下尝试新版本启动与加载
6. 只有在新版本真实缺同类能力时，才新增 `compat/sglang_0_5_9.py`

目标不是“让新版本也打上旧 patch”，而是明确回答每一项旧补丁：

- 新版本是否已经原生解决
- 若未解决，是否仍是同一个根因
- 若是，是否需要最小 compat

### 4. 文档与验收组织

文档需要同步升级成多版本视角。

#### `progress.md`

新增一节记录：

- 新版本环境事实
- 当前多版本结构设计
- 旧补丁的分类与去留判断
- 新版本适配过程中的关键发现

#### `changelog.md`

新增版本维度，明确写出：

- 哪些改动是公共 custom
- 哪些改动只对 `0.5.2` 生效
- 哪些改动在新版本继续保留
- 哪些改动被新版本上游吸收，可删除

#### 验收脚本

验收链路最终要支持：

- 检查当前环境的 `sglang` 版本
- 检查插件是否命中正确的版本 compat
- 继续执行 registry 接管、生成验收、logprob 验收

### 5. 项目级 skill 设计

新增一个项目级 skill，用来指导“给本仓库新增一个 `sglang` 版本适配”的标准流程。

该 skill 应固定覆盖以下内容：

1. 新增版本前必须先采集哪些环境事实
2. 应抽取哪些上游文件作为基线
3. 如何区分公共自定义与版本 compat
4. 何时允许新增 compat，何时应删除旧 compat
5. 如何更新 `progress.md`、`changelog.md`、spec、plan
6. 哪些本地脚本必须跑通，才能宣布该版本适配完成

它不是通用 SGLang 教程，而是本仓库的升级操作规程。

## 验证设计

### 1. 新版本事实基线验证

验证项：

- `sglang.__version__`
- `sglang.srt.models.qwen3_next` 实际路径
- `sglang.srt.configs.qwen3_next` 实际路径
- 是否原生支持 `SGLANG_EXTERNAL_MODEL_PACKAGE`

### 2. 多版本装配验证

验证项：

- 插件可根据当前 `sglang` 版本命中正确分支
- `EntryClass` 仍指向统一导出的 `Qwen3NextForCausalLM`
- 不同版本的 compat 不会互相污染

### 3. 结构对齐验证

验证项：

- 新版本在仅应用公共自定义层时，哪些地方已经能工作
- 对于失败点，能否明确归因为 compat 缺口

### 4. 运行时与质量验证

继续沿用现有验收标准：

- 自然语言生成不是乱码、空串或坏 token 模式
- logprob / 近似 `lm loss` 在健康区间

## 风险

- 新版本的模型文件布局、配置类命名或插件发现机制可能已经变化，不能假设与 `0.5.2` 相同
- 若新版本内部结构改动较大，公共自定义层可能需要重新抽象，而不是简单复制旧逻辑
- 当前新环境还未装 `sglang` 主包，若安装后发现版本并非用户预期，需要立即以真实版本为准更新文档和目录命名

## 原则

- 以目标环境里的真实上游代码为准，不以历史记忆为准
- 公共 custom 与版本 compat 必须显式分层
- 任何 compat 都要先证明必要性，再保留
- 每完成一个主要阶段都更新中文文档
- 项目级 skill 必须反映这套真实工作流，而不是抽象口号
