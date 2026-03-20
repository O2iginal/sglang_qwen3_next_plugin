# SGLang Qwen3 Next Plugin 进度记录

## 2026-03-20

### Phase 0: 上下文恢复
- 目标：从 `plan.md`、`.specstory` 与当前仓库状态中恢复用户的核心需求，再按“先设计、后实现”的方式推进。
- 已识别核心需求：在 `sglang==0.5.2` 下做一个真正可用的外部插件，使其接管用户自定义 Qwen3 Next checkpoint 的 `Qwen3NextForCausalLM` 推理实现，而不是只做一个打印成功横幅的 demo。
- 历史尝试只作为参考。由于早期关于包扫描与注册机制的结论曾多次变化，因此当前代码与真实运行时行为才是最终依据。
- 当前仓库状态已不止一个最小插件：
  - `sglang_qwen3_next_plugin/__init__.py` 里存在包级注册逻辑
  - `qwen3_next.py` 是带多处 `MODIFIED` 标记的 fork
  - 仓库内还保留过旧的启动包装与测试脚本
- 主要风险：仓库中很可能混杂了“真正的结构性修改”和“临时排障补丁”，如果不先拆开，最后很难形成稳定方案。

### 当前状态
- 进行中：在实现前先澄清准确的目标行为与验收标准。
- 新增文档要求：维护专门的 `changelog.md`，逐条记录插件相对上游 SGLang 的偏差，方便未来升级时系统性回放。
- 下一步：先提出聚焦问题，再形成方案选择与具体设计。

### Phase 1: 上游基线恢复
- 已确认唯一权威上游基线文件：
  - `/mnt/ssd/yulan/pretrain-linear-moe-dev-worktree/slime/.venv/lib/python3.12/site-packages/sglang/srt/models/qwen3_next.py`
- 当前插件 fork 相对上游虽然改动面不算大，但语义上都很关键：
  - layer norm replacement (`GemmaRMSNorm` -> `RMSNorm`)
  - non-MoE support when `num_experts=0`
  - disable `attn_output_gate` by default
  - remove `q_norm` / `k_norm`
  - change projection bias defaults
  - checkpoint loading skips for removed parameters
  - guard for `full_attention_interval == 0`
  - custom expert-location behavior for non-MoE
  - plugin-load indicator print
- 初步架构结论：
  - 插件加载/注册必须与模型逻辑定制拆开看
  - 当前 patch 中有些可能是 checkpoint 必需设计，有些则更像运行时绕错补丁
  - 后续应以上游 venv 文件为起点，逐项重新论证每个改动是否必要

### Phase 2: 验收标准澄清
- 需要支持的主启动路径：
  - `python -m sglang.launch_server` with `SGLANG_EXTERNAL_MODEL_PACKAGE=sglang_qwen3_next_plugin`
- 包装启动脚本只能作为调试辅助，不能替代原生插件入口成为主方案。
- 最终验收必须包含功能验证，而不只是插件发现：
  - 给定测试 checkpoint 能进行自然语言生成，且输出不是明显乱码；或者
  - 自然语言样本上的 `lm loss` 落在健康运行路径应有的范围内
- 文档要求：
  - 从此刻起仓库文档统一使用中文

### Phase 3: 设计与实施计划落盘
- 已写入中文设计文档：
  - `docs/superpowers/specs/2026-03-20-sglang-qwen3-next-plugin-design.md`
- 已写入中文实施计划：
  - `docs/superpowers/plans/2026-03-20-sglang-qwen3-next-plugin.md`
- 计划执行原则已经固定：
  - 以上游 venv 中的 `qwen3_next.py` 为唯一基线
  - 先做最小插件接管验证，再逐项恢复结构定制
  - 只有在必要时才引入运行时兼容补丁
- 说明：
  - 计划审阅子 agent 流程本轮未启用，后续由当前会话直接执行并人工校验关键节点

### Phase 4: 最小插件基线重建
- 已完成最小入口收缩：
  - `sglang_qwen3_next_plugin/__init__.py` 现在只负责导出 `Qwen3NextForCausalLM` 与 `EntryClass`
  - 已移除早期自动注册、`atexit`、全局 monkey patch 等不透明入口逻辑
- 已建立上游基线文件：
  - `sglang_qwen3_next_plugin/qwen3_next_upstream.py`
  - 来源：目标 venv 中的 `sglang/srt/models/qwen3_next.py`
- 已新增并通过最小契约测试：
  - `test_plugin.py`
  - 当前验证通过项：
    - 包根成功导出 `EntryClass`
    - `EntryClass.__name__ == "Qwen3NextForCausalLM"`
    - 上游基线模块存在且可导入
- 当前结论：
  - 现在已经具备“上游基线 + 最小插件入口”的可维护骨架
  - 下一步进入 SGLang 原生插件发现路径验证，而不是继续依赖包内副作用注册

### Phase 5: 原生环境变量入口打通
- 关键发现：
  - 当前目标环境中的 `sglang 0.5.2` 本地实现并未直接支持 `SGLANG_EXTERNAL_MODEL_PACKAGE`
  - `registry.py` 只扫描 `sglang.srt.models.*`
- 为满足用户要求的主入口，已引入最小接入层：
  - `sitecustomize.py`
  - `sglang_qwen3_next_plugin/env_override.py`
- 实现方式：
  - 在 Python 启动早期检测 `SGLANG_EXTERNAL_MODEL_PACKAGE=sglang_qwen3_next_plugin`
  - 注册一个最小 finder，使插件模块可在早期导入
  - 将 `sglang.srt.models.qwen3_next` 显式别名到插件实现
- 已通过验证脚本：
  - `scripts/check_plugin_import.py`
- 已验证结论：
  - 当环境变量在进程启动前设置时，`ModelRegistry.models["Qwen3NextForCausalLM"]`
    解析到 `sglang_qwen3_next_plugin.qwen3_next.Qwen3NextForCausalLM`
- 当前结论：
  - 用户要求的主入口路径已具备可行实现
  - 下一步应转向 checkpoint 结构对齐，判断哪些模型改动是必要的 `P1`

### Phase 6: 真实根因链排查与修复
- 已确认旧的“临时修复 `full_attention_interval=1`”会破坏 checkpoint 明确给出的层布局：
  - 目标 checkpoint 的 `config.json` 显式提供了 `layer_types`
  - 当前 venv 中的 `sglang/srt/configs/qwen3_next.py` 没有保存 `layer_types`，`layers_block_type` 也只按 `full_attention_interval` 推导
  - 当 `full_attention_interval=0` 时，原实现会报除零；把它强行改成 `1` 虽然能启动，但会把真实的混合层布局改坏
- 已实施修复：
  - 在插件中为 `Qwen3NextConfig` 补上 `layer_types` 保存逻辑
  - 重写 `layers_block_type` 属性，优先使用 checkpoint 显式提供的 `layer_types`
  - 删除先前“把 `full_attention_interval=0` 改成 `1`”的错误兜底
- 修复后出现的新现象：
  - 服务不再走错误层布局
  - 请求真正进入 `linear_attention` 路径后，触发 SGLang `hybrid_linear_attn_backend` 的 Triton `causal_conv1d` 编译错误
- 已定位并修复第 2 个运行时缺口：
  - `hybrid_linear_attn_backend.py` 默认硬连 Triton 版 `causal_conv1d_fn`
  - 该路径在当前环境下对本模型触发 `bf16/fp16` 类型分支错误
  - 插件已将其切换为同目录稳定版 `sglang.srt.layers.attention.mamba.causal_conv1d.causal_conv1d_fn`
- 已定位并修复第 3 个运行时缺口：
  - `model_runner` 分配 Mamba/Hybrid GDN cache 时，使用的是 `hf_config.hybrid_gdn_params`
  - 但真实运行 dtype 已被外层 `model_config.dtype` 下采样到 `torch.float16`
  - 原始 `hf_config` 仍保留 `torch.float32`，导致 `conv_state` 按错误 dtype 分配
  - 插件已在 `ModelRunner.init_memory_pool()` 入口同步：
    - `hf_config.torch_dtype = self.model_config.dtype`
    - `hf_config.dtype = self.model_config.dtype`
  - 同时保留 `hybrid_gdn_params` 侧的 dtype 兼容逻辑，确保后续读取时能得到正确 conv_state dtype
- 当前本地回归测试新增并通过：
  - 显式 `layer_types` 会覆盖 `full_attention_interval` 推导
  - `hybrid linear attention backend` 已切到稳定 `causal_conv1d_fn`
  - `hybrid_gdn_params` 会随模型 dtype 选择 `conv_state` dtype
  - `ModelRunner.init_memory_pool` 已添加 hybrid GDN dtype 同步补丁

### Phase 7: 服务端实测恢复
- 使用主验收入口重启服务：
  - `python -m sglang.launch_server`
  - `SGLANG_EXTERNAL_MODEL_PACKAGE=sglang_qwen3_next_plugin`
  - 测试 checkpoint：
    - `/mnt/ssd/yulan/cache/models/O2iginal/all_mcore_checkpoints/Dist-mathcode10b-s1randg-sch1-CPT-200b-stage2-r640k-GDN2.9b-A7-12_20_21_23_46_48_49-sl32768bs128lr2e5-2e5/merged_10ckpts_iter_38144-hf_to_iter_47683-hf_mean`
- 实测结果：
  - 服务已能穿过真实 `linear_attention` 路径
  - `/generate` 不再返回空串，也不再出现 `output_ids` 大片为 `0` / `<unk>` 的坏输出模式
  - 中文 prompt：
    - 输入：`你好，请用一句话介绍你自己。`
    - 输出摘要：返回了可读自然语言片段，内容形态接近回答模板，不再是乱码或空输出
  - 英文数学 prompt：
    - 输入：`What is 2 + 3? Answer briefly.`
    - 输出：`2 + 3 = 5`
  - 补充中文 prompt：
    - 输入：`请用两句话解释什么是机器学习。`
    - 输出摘要：返回了结构正常的中文解释性文本，如“机器学习是一种人工智能的分支……”
  - 补充英文 prompt：
    - 输入：`Write one short sentence about the sky.`
    - 输出摘要：返回了连贯英文自然语言，如 “The sky was a deep shade of blue...”
- 当前阶段结论：
  - 插件已从“只能加载但输出坏掉”推进到“按真实混合层布局运行，并能返回可读自然语言”
  - 这满足了当前核心验收标准中的关键部分
  - 后续可继续做更严格的质量验证，例如更长 prompt、更多样例与 `lm loss`

### Phase 8: 工程化收尾
- 已新增自动化生成验收脚本：
  - `scripts/validate_generation.py`
  - 用途：
    - 启动服务后快速验证中英文最小样例
    - 输出 `PASS/FAIL`、生成文本与 `output_ids` 前缀
    - 作为后续升级 SGLang 时的最小回归入口
- 已收敛插件成功横幅打印：
  - 原本在 `Qwen3NextForCausalLM.__init__` 中每次实例化都会打印大块绿色横幅
  - 现在改为每个进程只打印一次，减少多进程环境下的日志噪音
- 已更新 `README.md`：
  - 文档主入口改为用户要求的 `SGLANG_EXTERNAL_MODEL_PACKAGE` 路径
  - 移除了与当前真实方案不一致的旧包装脚本推荐语义
  - 补充了：
    - `test_plugin.py`
    - `scripts/check_plugin_import.py`
    - `scripts/validate_generation.py`
  - 并明确区分：
    - 结构性 checkpoint 对齐
    - `sglang 0.5.2` 运行时兼容补丁

### Phase 9: logprob / 近似 lm loss 验证落地
- 已新增脚本：
  - `scripts/validate_logprob.py`
- 验证方式：
  - 调用本地 `/generate`
  - 打开 `return_logprob=true`
  - 设置 `logprob_start_len=0`
  - 使用 prompt token logprob 计算平均负对数似然 `avg_nll`
- 当前实测结果：
  - 中文样例 `请用两句话解释什么是机器学习。`
    - `avg_nll = 4.3043`
    - `approx_ppl = 74.0148`
  - 英文样例 `What is 2 + 3? Answer briefly.`
    - `avg_nll = 3.1062`
    - `approx_ppl = 22.3354`
- 当前阶段结论：
  - 除了“可读自然语言生成”外，服务端 logprob 数值也已经落在正常可解释范围
  - 这满足了用户此前提出的第二类验收标准

### Phase 10: 串联验收入口
- 已新增脚本：
  - `scripts/run_acceptance.py`
- 当前支持两种模式：
  - `--skip-runtime`
    - 只检查本地契约与 registry 接管
  - 默认完整模式
    - 串联 `test_plugin.py`
    - `scripts/check_plugin_import.py`
    - `scripts/validate_generation.py`
    - `scripts/validate_logprob.py`
- 当前已实测：
  - `python scripts/run_acceptance.py --skip-runtime`
  - 返回 `ALL CHECKS PASSED`
- 完整服务验收也已实测：
  - `python scripts/run_acceptance.py --host 127.0.0.1 --port 30110`
  - 返回 `ALL CHECKS PASSED`
  - 串联通过内容包括：
    - 插件导入契约
    - registry 接管
    - 自然语言生成
    - logprob / 近似 lm loss
- 目的：
  - 后续升级 SGLang 后，可以先跑轻量验收，再跑完整服务验收
