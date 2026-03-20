#!/usr/bin/env python3
"""
验证原生启动前的插件发现路径。

期望行为：
- 设置 SGLANG_EXTERNAL_MODEL_PACKAGE=sglang_qwen3_next_plugin
- 导入 SGLang registry 后，Qwen3NextForCausalLM 应来自插件包，而非上游 sglang.srt.models
"""

import os
import sys


def main() -> int:
    os.environ.setdefault("HOME", "/tmp")
    os.environ.setdefault("XDG_CACHE_HOME", "/tmp")
    assert (
        os.environ.get("SGLANG_EXTERNAL_MODEL_PACKAGE") == "sglang_qwen3_next_plugin"
    ), "必须在进程启动前设置 SGLANG_EXTERNAL_MODEL_PACKAGE=sglang_qwen3_next_plugin"

    from sglang.srt.models.registry import ModelRegistry

    model_cls = ModelRegistry.models.get("Qwen3NextForCausalLM")
    assert model_cls is not None, "registry 中缺少 Qwen3NextForCausalLM"

    print(f"resolved class: {model_cls}")
    print(f"resolved module: {model_cls.__module__}")

    assert model_cls.__module__.startswith("sglang_qwen3_next_plugin"), (
        "Qwen3NextForCausalLM 未指向插件实现"
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except AssertionError as exc:
        print(f"ASSERTION FAILED: {exc}")
        raise SystemExit(1)
    except Exception as exc:
        print(f"UNEXPECTED ERROR: {exc}")
        raise SystemExit(1)
