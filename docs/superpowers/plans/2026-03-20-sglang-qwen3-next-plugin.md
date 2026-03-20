# SGLang Qwen3 Next Plugin 实施计划

> **对执行型 agent 的要求：** 必须配合 `superpowers:subagent-driven-development`（推荐）或 `superpowers:executing-plans` 按任务逐项实施本计划。任务步骤使用复选框 `- [ ]` 语法跟踪。

**目标：** 基于目标 venv 中的上游 `qwen3_next.py` 重建一个可长期维护的 SGLang 外部插件，并通过原生插件入口、生成验证与必要的 `lm loss` 验证确认其逻辑正确。

**总体方案：** 先建立干净的上游基线与最小插件入口，只验证插件被 SGLang 原生机制发现；然后逐条恢复真正需要的模型结构定制；最后在必要时加入最小运行时兼容补丁，并用测试 checkpoint 做生成与质量验证。

**技术栈：** Python 3.12、sglang 0.5.2、PyTorch、Transformers、Triton、editable package install

---

### 任务 1：建立中文文档与实施基线

**涉及文件：**
- 新增：`docs/superpowers/specs/2026-03-20-sglang-qwen3-next-plugin-design.md`
- 新增：`docs/superpowers/plans/2026-03-20-sglang-qwen3-next-plugin.md`
- 修改：`progress.md`
- 修改：`changelog.md`

- [ ] **Step 1: 记录已确认的目标与验收标准**

将以下内容写入文档：

```md
- 主入口必须是 `SGLANG_EXTERNAL_MODEL_PACKAGE`
- 验收必须覆盖自然语言生成或 `lm loss`
- 所有文档使用中文
```

- [ ] **Step 2: 更新进度文档**

运行：`sed -n '1,260p' progress.md`
预期：包含当前阶段、基线来源、验收标准

- [ ] **Step 3: 更新变更记录模板**

在 `changelog.md` 中为后续上游差异整理出固定字段：

```md
- 改动点
- 原因
- 分类（P1/P2）
- 升级复核方式
```

### 任务 2：以上游 venv 文件重建最小插件骨架

**涉及文件：**
- 新增：`sglang_qwen3_next_plugin/qwen3_next_upstream.py`
- 修改：`sglang_qwen3_next_plugin/qwen3_next.py`
- 修改：`sglang_qwen3_next_plugin/__init__.py`
- 修改：`README.md`
- 测试：`test_plugin.py`

- [ ] **Step 1: 为插件发现写一个失败前提检查**

在 `test_plugin.py` 或新测试脚本中先检查：

```python
import importlib
mod = importlib.import_module("sglang_qwen3_next_plugin")
assert hasattr(mod, "EntryClass")
assert mod.EntryClass.__name__ == "Qwen3NextForCausalLM"
```

- [ ] **Step 2: 运行检查并记录当前失败或不透明行为**

运行：`python test_plugin.py`
预期：明确显示当前入口暴露状态，不允许只输出笼统成功

- [ ] **Step 3: 拷贝上游文件到 `qwen3_next_upstream.py`**

保持内容尽量原样，仅加说明注释：

```python
# copied from target venv sglang 0.5.2 baseline
```

- [ ] **Step 4: 让插件导出最小可识别模型实现**

`__init__.py` 中只保留清晰入口：

```python
from .qwen3_next import Qwen3NextForCausalLM

EntryClass = Qwen3NextForCausalLM
__all__ = ["Qwen3NextForCausalLM", "EntryClass"]
```

- [ ] **Step 5: 运行导入检查**

运行：`python test_plugin.py`
预期：PASS，明确显示 `EntryClass` 与插件模型类

### 任务 3：验证原生插件接管路径

**涉及文件：**
- 修改：`sglang_qwen3_next_plugin/qwen3_next.py`
- 新增：`scripts/check_plugin_import.py`
- 修改：`progress.md`

- [ ] **Step 1: 添加最小插件接管标识**

在插件模型初始化处加入轻量、明确但不污染行为的标识：

```python
print("[sglang_qwen3_next_plugin] using plugin Qwen3NextForCausalLM")
```

- [ ] **Step 2: 编写原生插件路径检查脚本**

脚本应完成：

```python
import os
os.environ["SGLANG_EXTERNAL_MODEL_PACKAGE"] = "sglang_qwen3_next_plugin"
from sglang.srt.models.registry import import_model_classes
```

并验证能发现插件模型。

- [ ] **Step 3: 运行脚本确认插件接管**

运行：`python scripts/check_plugin_import.py`
预期：输出中明确包含插件模型类名称与模块路径

- [ ] **Step 4: 将发现机制结论写入进度文档**

写明：

```md
- 原生插件入口是否成立
- 是否需要额外注册逻辑
- 当前是否仍存在不透明行为
```

### 任务 4：梳理当前 fork 相对上游的差异并分类

**涉及文件：**
- 修改：`changelog.md`
- 修改：`progress.md`

- [ ] **Step 1: 生成上游与当前插件的差异列表**

运行：`git diff --no-index <upstream> sglang_qwen3_next_plugin/qwen3_next.py`
预期：得到完整差异块

- [ ] **Step 2: 将差异写入 `changelog.md`**

按以下模板逐项整理：

```md
### 变更项：移除 q_norm / k_norm
- 分类：P1
- 原因：待验证
- 对应 checkpoint：待验证
- 升级复核：检查上游是否仍保留该路径
```

- [ ] **Step 3: 标出需要先移除再验证的可疑项**

例如：

```md
- atexit 自动注册
- monkey patch profile_max_num_token
- 静默跳过未知权重
```

### 任务 5：逐项恢复真正必要的模型结构定制

**涉及文件：**
- 修改：`sglang_qwen3_next_plugin/qwen3_next.py`
- 修改：`changelog.md`
- 测试：`scripts/check_plugin_import.py`

- [ ] **Step 1: 从纯上游版本开始，只保留 P0 能力**

确保 `qwen3_next.py` 先尽量接近上游，仅保留插件标识与必要入口。

- [ ] **Step 2: 每次只恢复一类 P1 改动**

候选顺序：

```md
1. num_experts=0 -> 普通 MLP
2. LayerNorm 差异
3. q_norm/k_norm 差异
4. projection bias 差异
5. attn_output_gate 差异
```

- [ ] **Step 3: 每恢复一项就记录证据**

记录内容：

```md
- 加回后是否更接近 checkpoint 结构
- 是否修复了实际报错
- 是否引入了新的不匹配
```

- [ ] **Step 4: 对不成立的改动立即回退**

运行：`git diff -- sglang_qwen3_next_plugin/qwen3_next.py`
预期：差异保持最小且有解释

### 任务 6：只在必要时引入最小运行时兼容补丁

**涉及文件：**
- 修改：`sglang_qwen3_next_plugin/qwen3_next.py`
- 新增或修改：`sglang_qwen3_next_plugin/runtime_compat.py`
- 修改：`changelog.md`

- [ ] **Step 1: 确认问题是否真的属于 P2**

判断标准：

```md
- P1 已自洽
- 仍然被 SGLang 运行时或接口兼容问题阻塞
```

- [ ] **Step 2: 优先把 P2 与模型定义解耦**

如果必须加兼容逻辑，优先放入独立模块，避免污染模型主体。

- [ ] **Step 3: 明确禁止静默吞错**

任何跳过参数或自动修正配置的逻辑都必须：

```python
logger.warning("why this compatibility path is used")
```

- [ ] **Step 4: 更新 `changelog.md`**

区分：

```md
- P1: checkpoint 结构定制
- P2: 运行时兼容补丁
```

### 任务 7：使用测试 checkpoint 做启动与生成验证

**涉及文件：**
- 新增：`scripts/run_generation_smoke.py`
- 修改：`progress.md`
- 修改：`README.md`

- [ ] **Step 1: 写启动与请求脚本**

脚本应完成：

```python
- 设置 SGLANG_EXTERNAL_MODEL_PACKAGE
- 启动 sglang.launch_server
- 发送至少数条自然语言请求
- 收集返回文本
```

- [ ] **Step 2: 用用户提供的 checkpoint 运行冒烟验证**

运行：`python scripts/run_generation_smoke.py --model-path <checkpoint>`
预期：返回可读自然语言，不是明显乱码或异常重复

- [ ] **Step 3: 记录样例结果**

在 `progress.md` 中写明：

```md
- prompt
- 输出摘要
- 是否通过自然语言验证
```

### 任务 8：必要时补充 `lm loss` 验证

**涉及文件：**
- 新增：`scripts/eval_lm_loss.py`
- 修改：`progress.md`
- 修改：`README.md`

- [ ] **Step 1: 实现最小 `lm loss` 评估脚本**

脚本至少要支持：

```python
- 输入模型路径
- 输入若干自然语言样本
- 输出平均 loss
```

- [ ] **Step 2: 在生成结论不稳定时运行**

运行：`python scripts/eval_lm_loss.py --model-path <checkpoint> --input <samples>`
预期：得到可记录的数值基线

- [ ] **Step 3: 记录“正常值”上下文**

写入：

```md
- 样本来源
- 推理配置
- 平均 loss
- 为什么认为该值正常或异常
```

### 任务 9：整理最终交付与升级指引

**涉及文件：**
- 修改：`README.md`
- 修改：`changelog.md`
- 修改：`progress.md`

- [ ] **Step 1: 更新 README**

包含：

```md
- 原生插件使用方法
- 测试环境
- 测试 checkpoint
- 已验证通过的路径
```

- [ ] **Step 2: 固化升级指引**

在 `changelog.md` 中补齐每个偏离上游项的升级检查方法。

- [ ] **Step 3: 写最终阶段总结**

在 `progress.md` 中明确：

```md
- 完成项
- 未完成项
- 风险
- 建议的下一步
```
