#!/usr/bin/env python3
"""
串联当前插件的最小验收链路。

顺序：
1. 检查插件导入契约
2. 检查 registry 接管
3. 检查自然语言生成
4. 检查 logprob / 近似 lm loss
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def run_step(name: str, cmd: list[str], extra_env: dict[str, str] | None = None) -> None:
    env = os.environ.copy()
    env.setdefault("HOME", "/tmp")
    env.setdefault("XDG_CACHE_HOME", "/tmp")
    if extra_env:
        env.update(extra_env)

    print(f"=== {name} ===")
    print("cmd:", " ".join(cmd))
    result = subprocess.run(cmd, cwd=REPO_ROOT, env=env)
    if result.returncode != 0:
        raise SystemExit(result.returncode)
    print()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=30110)
    parser.add_argument(
        "--skip-runtime",
        action="store_true",
        help="只检查本地契约与 registry，不检查服务端生成/logprob。",
    )
    args = parser.parse_args()

    run_step(
        "plugin_contract",
        [sys.executable, str(REPO_ROOT / "test_plugin.py")],
    )
    run_step(
        "plugin_registry",
        [sys.executable, str(REPO_ROOT / "scripts" / "check_plugin_import.py")],
        extra_env={"SGLANG_EXTERNAL_MODEL_PACKAGE": "sglang_qwen3_next_plugin"},
    )

    if not args.skip_runtime:
        run_step(
            "generation_validation",
            [
                sys.executable,
                str(REPO_ROOT / "scripts" / "validate_generation.py"),
                "--host",
                args.host,
                "--port",
                str(args.port),
            ],
        )
        run_step(
            "logprob_validation",
            [
                sys.executable,
                str(REPO_ROOT / "scripts" / "validate_logprob.py"),
                "--host",
                args.host,
                "--port",
                str(args.port),
            ],
        )

    print("ALL CHECKS PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
