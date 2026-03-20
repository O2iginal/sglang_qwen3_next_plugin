"""
通过 Python 启动钩子为 SGLang 0.5.2 提供环境变量插件覆盖能力。

当前目标环境中的 SGLang registry 不会读取 `SGLANG_EXTERNAL_MODEL_PACKAGE`。
这里在 Python 启动早期注册一个极小的 finder，使插件模块可以被提前导入，
然后把 `sglang.srt.models.qwen3_next` 显式别名到插件实现。
"""

from __future__ import annotations

import importlib
import importlib.abc
import importlib.machinery
import importlib.util
import os
import sys
from pathlib import Path


PLUGIN_PACKAGE = "sglang_qwen3_next_plugin"
PLUGIN_SUBMODULE = f"{PLUGIN_PACKAGE}.qwen3_next"
TARGET_MODULE = "sglang.srt.models.qwen3_next"

_ROOT = Path(__file__).resolve().parent
_PKG_DIR = _ROOT / PLUGIN_PACKAGE
_INIT_FILE = _PKG_DIR / "__init__.py"
_MODEL_FILE = _PKG_DIR / "qwen3_next.py"


class _PluginFinder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname: str, path=None, target=None):
        if fullname == PLUGIN_PACKAGE:
            return importlib.util.spec_from_file_location(
                fullname,
                _INIT_FILE,
                submodule_search_locations=[str(_PKG_DIR)],
            )
        if fullname == PLUGIN_SUBMODULE:
            return importlib.util.spec_from_file_location(fullname, _MODEL_FILE)
        return None


def _install_plugin_finder() -> None:
    if not any(isinstance(finder, _PluginFinder) for finder in sys.meta_path):
        sys.meta_path.insert(0, _PluginFinder())


def _install_override() -> None:
    if os.environ.get("SGLANG_EXTERNAL_MODEL_PACKAGE") != PLUGIN_PACKAGE:
        return

    _install_plugin_finder()
    plugin_module = importlib.import_module(PLUGIN_SUBMODULE)
    sys.modules[TARGET_MODULE] = plugin_module


_install_override()
