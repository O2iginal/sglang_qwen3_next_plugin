"""
基于环境变量的 SGLang 模型覆盖入口。

SGLang 0.5.2 当前安装版本的 registry 只会扫描 `sglang.srt.models.*`，
并不会读取 `SGLANG_EXTERNAL_MODEL_PACKAGE`。

为了让用户要求的启动方式成立，这里在 Python 进程启动早期将
`sglang.srt.models.qwen3_next` 显式别名到插件实现。
"""

from __future__ import annotations

import importlib
import os
import sys


PLUGIN_PACKAGE = "sglang_qwen3_next_plugin"
TARGET_MODULE = "sglang.srt.models.qwen3_next"
PLUGIN_MODULE = "sglang_qwen3_next_plugin.qwen3_next"


def install_env_override() -> bool:
    external_pkg = os.environ.get("SGLANG_EXTERNAL_MODEL_PACKAGE")
    if external_pkg != PLUGIN_PACKAGE:
        return False

    if TARGET_MODULE in sys.modules:
        return False

    plugin_module = importlib.import_module(PLUGIN_MODULE)
    sys.modules[TARGET_MODULE] = plugin_module
    return True
