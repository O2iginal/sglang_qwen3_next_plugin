# SGLang Qwen3 Next Plugin 创建计划

## 一、项目目标
创建一个 SGLang 插件，用于自定义修改 Qwen3 Next 模型的推理行为，通过插件机制替换 SGLang 原有的模型实现。

## 二、实现步骤

### 阶段 1: 项目结构搭建 ✅
1. ✅ 完善插件包结构
   - ✅ 创建 `__init__.py` 用于模型注册
   - ✅ 创建 `_0_5_2/__init__.py` 确保包结构完整
   - ✅ 更新 `pyproject.toml` 配置
   - ✅ 确保包可以正确导入

### 阶段 2: 模型修改 ✅
2. ✅ 在 `qwen3_next.py` 中添加标识
   - ✅ 在 `Qwen3NextForCausalLM.__init__` 中添加彩色打印
   - ✅ 使用 ANSI 颜色代码（绿色）显示插件已加载
   - ✅ 确保打印在模型初始化时执行
   - ✅ 打印格式：绿色边框 + 成功提示信息

### 阶段 3: 模型注册 ✅
3. ✅ 实现模型注册机制
   - ✅ 在 `__init__.py` 中导入自定义模型类
   - ✅ 使用 `EntryClass` 导出模型类（列表格式）
   - ✅ 确保 ModelRegistry 可以自动发现

### 阶段 4: 配置更新 ✅
4. ✅ 更新项目配置
   - ✅ 完善 `pyproject.toml` 依赖和元数据
   - ✅ 添加必要的依赖项（sglang>=0.5.0, torch, transformers）
   - ✅ 添加构建系统配置
   - ✅ 创建 README.md 使用说明

### 阶段 5: 测试验证 ⏳
5. ⏳ 测试插件加载
   - ⏳ 通过环境变量 `SGLANG_EXTERNAL_MODEL_PACKAGE` 加载
   - ⏳ 验证彩色打印是否出现
   - ⏳ 确认模型可以正常推理
   - ✅ 创建测试脚本 `test_plugin.py` 用于验证结构

## 三、技术要点

### 模型注册方式
- 方式1: 通过环境变量自动加载
  ```bash
  export SGLANG_EXTERNAL_MODEL_PACKAGE="sglang_qwen3_next_plugin"
  ```
- 方式2: 手动注册（在代码中）
  ```python
  from sglang.srt.models.registry import ModelRegistry
  from sglang_qwen3_next_plugin import Qwen3NextForCausalLM
  ModelRegistry.models["Qwen3NextForCausalLM"] = Qwen3NextForCausalLM
  ```

### 模型配置要求
- 模型的 `config.json` 中 `architectures` 字段需要包含 `"Qwen3NextForCausalLM"`
- 确保模型权重路径正确

### 彩色打印实现
- 使用 ANSI 转义序列实现彩色输出
- 在 `__init__` 方法中添加，确保每次模型初始化都会打印
- 格式：`\033[颜色码m文本\033[0m`

## 四、文件结构

```
sglang_qwen3_next_plugin/
├── pyproject.toml          # 项目配置
├── README.md              # 项目说明
├── plan.md                # 本计划文件
├── sglang_qwen3_next_plugin/
│   ├── __init__.py        # 包初始化，模型注册
│   └── 0_5_2/
│       └── qwen3_next.py  # 自定义的 Qwen3 Next 模型实现
```

## 五、验证方法

1. 安装插件包：
   ```bash
   pip install -e .
   ```

2. 设置环境变量并启动服务器：
   ```bash
   export SGLANG_EXTERNAL_MODEL_PACKAGE="sglang_qwen3_next_plugin"
   python3 -m sglang.launch_server --model <model_path> --tp 1
   ```

3. 检查输出：
   - 应该看到彩色打印信息，表明插件模型已加载
   - 模型应该可以正常进行推理

## 六、注意事项

1. **版本兼容性**: 确保插件与 SGLang 版本兼容（当前基于 v0.5.2）
2. **依赖管理**: 插件需要依赖 sglang，但不应直接修改 sglang 源码
3. **模型配置**: 使用插件时，模型的 config.json 需要正确配置 architectures
4. **调试**: 如果插件未加载，检查环境变量和包安装是否正确
