#!/usr/bin/env python3
"""
对本地 SGLang 服务做最小自然语言生成验收。

用途：
1. 升级 SGLang 后快速检查插件是否仍能返回可读文本
2. 避免手工重复敲 curl
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.request


DEFAULT_CASES = [
    {
        "name": "zh_intro",
        "prompt": "你好，请用一句话介绍你自己。",
        "max_new_tokens": 16,
    },
    {
        "name": "en_math",
        "prompt": "What is 2 + 3? Answer briefly.",
        "max_new_tokens": 16,
    },
]


def request_json(url: str, payload: dict) -> dict:
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=120) as resp:
        return json.loads(resp.read().decode("utf-8"))


def is_readable_text(text: str) -> bool:
    stripped = text.strip()
    if not stripped:
        return False
    bad_markers = ["<unk><unk>", "\x00"]
    return not any(marker in stripped for marker in bad_markers)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=30110)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-new-tokens", type=int, default=None)
    args = parser.parse_args()

    url = f"http://{args.host}:{args.port}/generate"
    failed = False

    for case in DEFAULT_CASES:
        max_new_tokens = args.max_new_tokens or case["max_new_tokens"]
        payload = {
            "text": case["prompt"],
            "sampling_params": {
                "temperature": args.temperature,
                "max_new_tokens": max_new_tokens,
            },
        }
        try:
            result = request_json(url, payload)
        except urllib.error.URLError as exc:
            print(f"[FAIL] {case['name']}: request error: {exc}")
            failed = True
            continue

        text = result.get("text", "")
        output_ids = result.get("output_ids", [])
        ok = is_readable_text(text)
        status = "PASS" if ok else "FAIL"
        print(f"[{status}] {case['name']}")
        print(f"prompt: {case['prompt']}")
        print(f"text: {text!r}")
        print(f"output_ids_head: {output_ids[:12]}")
        print()
        failed = failed or (not ok)

    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
