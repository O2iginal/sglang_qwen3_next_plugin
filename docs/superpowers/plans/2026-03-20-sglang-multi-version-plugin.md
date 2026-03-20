# SGLang Qwen3 Next 多版本插件适配实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 将当前仅适配 `sglang==0.5.2` 的 Qwen3 Next 插件升级为单仓库多版本适配结构，并完成新版本官方 `sglang` 主包环境的接入、补丁归因、验收与项目级 skill 编写。

**Architecture:** 先把新版本环境事实采集清楚，再按“上游基线层 + 公共自定义层 + 版本兼容层 + 入口装配层”重组插件。实现过程中把现有 `0.5.2` 补丁逐项分类，明确哪些是 checkpoint 语义定制、哪些只是 `0.5.2` 的历史 compat，然后只为新版本补最小必要的 compat。

**Tech Stack:** Python 3.12、sglang、PyTorch、Transformers、editable install、本地验收脚本、项目级 superpowers skill

---

### Task 1: 建立多版本设计与计划文档基线

**Files:**
- Create: `docs/superpowers/specs/2026-03-20-sglang-multi-version-plugin-design.md`
- Create: `docs/superpowers/plans/2026-03-20-sglang-multi-version-plugin.md`
- Modify: `progress.md`

- [ ] **Step 1: 写入本轮设计文档**

确认设计文档包含以下小节：

```md
- 仓库四层结构
- 公共 custom 与版本 compat 的分类原则
- 新版本适配策略
- 项目级 skill 设计
```

- [ ] **Step 2: 将当前阶段结论追加到 `progress.md`**

记录：

```md
- 新环境当前未安装 sglang 主包
- 本轮决定采用单仓库多版本适配
- 下一步先安装官方主包并采集事实基线
```

- [ ] **Step 3: 自检文档是否与当前用户约束一致**

运行：`sed -n '1,260p' docs/superpowers/specs/2026-03-20-sglang-multi-version-plugin-design.md`
Expected: 明确使用官方 `sglang` 主包、保留多版本支持、文档使用中文

### Task 2: 在新环境安装官方 `sglang` 主包并采集事实基线

**Files:**
- Modify: `progress.md`
- Modify: `changelog.md`

- [ ] **Step 1: 安装官方 `sglang` 主包到目标环境**

Run: `source /mnt/ssd/yulan/pretrain-linear-moe-dev-worktree/slime/.venv-sgl0_5_9/bin/activate && python -m pip install sglang`
Expected: `sglang` 主包成功安装，可被 `python -c "import sglang"` 导入

- [ ] **Step 2: 记录新版本事实信息**

Run:

```bash
source /mnt/ssd/yulan/pretrain-linear-moe-dev-worktree/slime/.venv-sgl0_5_9/bin/activate
python - <<'PY'
import inspect
import sglang
import sglang.srt.models.qwen3_next as model_mod
import sglang.srt.configs.qwen3_next as config_mod
print(sglang.__version__)
print(inspect.getsourcefile(model_mod))
print(inspect.getsourcefile(config_mod))
PY
```

Expected: 输出真实版本号与新上游文件路径

- [ ] **Step 3: 检查新版本是否原生支持外部模型包**

Run: `grep -R "SGLANG_EXTERNAL_MODEL_PACKAGE" <new-sglang-site-packages>/sglang -n`
Expected: 明确知道新版本是否还需要 `sitecustomize.py` / `env_override.py`

- [ ] **Step 4: 将事实结论写入 `progress.md`**

记录：

```md
- 新版本号
- qwen3_next.py / config 文件路径
- 外部插件入口是否原生支持
```

### Task 3: 提取新版本上游基线并建立目录结构

**Files:**
- Create: `sglang_qwen3_next_plugin/upstream/sglang_0_5_9/qwen3_next.py`
- Create: `sglang_qwen3_next_plugin/upstream/sglang_0_5_9/configs_qwen3_next.py`
- Modify: `README.md`

- [ ] **Step 1: 为新版本建立上游目录**

目录至少包含：

```text
sglang_qwen3_next_plugin/upstream/sglang_0_5_9/
```

- [ ] **Step 2: 拷贝新版本上游模型与配置基线**

把新环境中的上游文件复制到仓库，并在文件头加最小说明注释：

```python
# copied from installed sglang <version> baseline
```

- [ ] **Step 3: 检查与 `0.5.2` 基线的结构差异**

Run: `git diff --no-index sglang_qwen3_next_plugin/qwen3_next_upstream.py sglang_qwen3_next_plugin/upstream/sglang_0_5_9/qwen3_next.py`
Expected: 得到足够明确的新旧上游差异块

- [ ] **Step 4: 在 `README.md` 中补充多版本目录说明**

说明：

```md
- `upstream/sglang_0_5_2`
- `upstream/sglang_0_5_9`
- 后续版本按同样方式新增
```

### Task 4: 重构插件为多版本装配结构

**Files:**
- Create: `sglang_qwen3_next_plugin/upstream/__init__.py`
- Create: `sglang_qwen3_next_plugin/compat/__init__.py`
- Create: `sglang_qwen3_next_plugin/compat/sglang_0_5_2.py`
- Create: `sglang_qwen3_next_plugin/compat/sglang_0_5_9.py`
- Create: `sglang_qwen3_next_plugin/versioning.py`
- Modify: `sglang_qwen3_next_plugin/__init__.py`
- Modify: `sglang_qwen3_next_plugin/qwen3_next.py`

- [ ] **Step 1: 写一个最小版本检测单元**

实现能力：

```python
def detect_sglang_version() -> str: ...
def get_version_key() -> str: ...
```

- [ ] **Step 2: 定义上游基线选择逻辑**

实现能力：

```python
def load_upstream_variant(version_key: str): ...
```

- [ ] **Step 3: 定义 compat 装配接口**

每个 compat 模块至少导出：

```python
def apply_runtime_compat() -> list[str]:
    ...
```

- [ ] **Step 4: 统一包根导出**

`__init__.py` 需要：

```python
from .qwen3_next import Qwen3NextForCausalLM
EntryClass = Qwen3NextForCausalLM
```

- [ ] **Step 5: 运行现有契约测试**

Run: `python test_plugin.py`
Expected: PASS，且 `EntryClass` 仍正常导出

### Task 5: 把现有 `0.5.2` 补丁逐项分类

**Files:**
- Modify: `changelog.md`
- Modify: `progress.md`

- [ ] **Step 1: 列出当前 `0.5.2` 的所有偏离上游项**

Run: `git diff --no-index sglang_qwen3_next_plugin/qwen3_next_upstream.py sglang_qwen3_next_plugin/qwen3_next.py`
Expected: 得到完整差异列表

- [ ] **Step 2: 逐项归类为公共 custom 或 `0.5.2` compat**

按如下模板写入 `changelog.md`：

```md
### 变更项：<name>
- 类型：公共 custom / 0.5.2 compat / 待确认
- 新版本是否仍需要：待验证
- 说明：...
```

- [ ] **Step 3: 给每个 `0.5.2` compat 增加新版本验证结论字段**

字段示例：

```md
- 0.5.9 状态：未验证 / 已删除 / 继续保留 / 已被上游吸收
```

### Task 6: 先在新版本上只移植公共自定义层

**Files:**
- Modify: `sglang_qwen3_next_plugin/qwen3_next.py`
- Modify: `README.md`

- [ ] **Step 1: 以新版本上游基线为底建立新版本模型装配路径**

确保先不带任何 `0.5.2` compat。

- [ ] **Step 2: 只迁移公共 custom**

优先顺序：

```md
1. RMSNorm 差异
2. num_experts=0 普通 MLP 路径
3. q_norm / k_norm 差异
4. projection bias
5. attn_output_gate
```

- [ ] **Step 3: 记录每一项迁移后的证据**

记录内容：

```md
- 是否与 checkpoint 结构更一致
- 是否引入新的导入或构造错误
- 是否仍需要 compat
```

### Task 7: 仅为新版本添加必要 compat

**Files:**
- Modify: `sglang_qwen3_next_plugin/compat/sglang_0_5_9.py`
- Modify: `sitecustomize.py`
- Modify: `sglang_qwen3_next_plugin/env_override.py`
- Modify: `changelog.md`

- [ ] **Step 1: 先验证新版本是否仍需要旧的 env 接入层**

Run: `python scripts/check_plugin_import.py`
Expected: 明确新版本是否原生支持外部模型包

- [ ] **Step 2: 若新版本仍缺失能力，再最小化补 compat**

compat 范围只允许包括：

```md
- 插件发现差异
- 配置字段兼容
- 运行时 dtype / backend 问题
```

- [ ] **Step 3: 明确记录哪些 `0.5.2` compat 没被带到新版本**

在 `changelog.md` 中写出“删除理由”，避免以后误加回去。

### Task 8: 扩展验收链路为多版本视角

**Files:**
- Modify: `scripts/check_plugin_import.py`
- Modify: `scripts/run_acceptance.py`
- Modify: `scripts/validate_generation.py`
- Modify: `scripts/validate_logprob.py`
- Modify: `README.md`

- [ ] **Step 1: 在验收输出中显示当前 `sglang` 版本**

Expected output 示例：

```text
sglang_version=0.5.9
compat=sglang_0_5_9
```

- [ ] **Step 2: 保证轻量验收与完整验收都适用于多版本**

Run:

```bash
python scripts/run_acceptance.py --skip-runtime
python scripts/run_acceptance.py --host 127.0.0.1 --port 30110
```

Expected: 至少轻量模式在两个版本环境都能运行；完整模式在对应服务启动后返回 PASS

- [ ] **Step 3: 文档补充多版本验收说明**

说明不同环境的激活方式与预期输出。

### Task 9: 编写项目级 skill

**Files:**
- Create: `.codex/skills/sglang-plugin-version-adaptation/SKILL.md` 或仓库约定的项目级 skill 路径
- Modify: `README.md`
- Modify: `progress.md`

- [ ] **Step 1: 先确认本仓库项目级 skill 的实际落点**

Run: `find .. -maxdepth 3 -type d | grep -E '/skills$|/\\.codex'`
Expected: 找到当前仓库适合放置项目级 skill 的位置

- [ ] **Step 2: 编写 skill 内容**

必须包含：

```md
- 新增版本前的事实采集清单
- 上游基线提取步骤
- 公共 custom / 版本 compat 分类规则
- 文档更新要求
- 验收脚本清单
```

- [ ] **Step 3: 在 README 中补充 skill 的用途**

说明这是本仓库升级 `sglang` 版本时的标准作业流程。

### Task 10: 完成多版本验证并准备收尾

**Files:**
- Modify: `progress.md`
- Modify: `changelog.md`
- Modify: `README.md`

- [ ] **Step 1: 在 `0.5.2` 环境重新跑轻量验收**

Run: `source /mnt/ssd/yulan/pretrain-linear-moe-dev-worktree/slime/.venv/bin/activate && python scripts/run_acceptance.py --skip-runtime`
Expected: PASS

- [ ] **Step 2: 在新版本环境跑轻量验收**

Run: `source /mnt/ssd/yulan/pretrain-linear-moe-dev-worktree/slime/.venv-sgl0_5_9/bin/activate && python scripts/run_acceptance.py --skip-runtime`
Expected: PASS

- [ ] **Step 3: 若服务已启动，再跑完整生成与 logprob 验收**

Run:

```bash
python scripts/validate_generation.py --host 127.0.0.1 --port 30110
python scripts/validate_logprob.py --host 127.0.0.1 --port 30110
```

Expected: PASS，输出为可读自然语言且 logprob 健康

- [ ] **Step 4: 更新最终文档结论**

写清：

```md
- 新版本是否原生支持外部模型包
- 哪些补丁只属于 0.5.2
- 哪些补丁在新版本继续保留
- 项目级 skill 路径
```
