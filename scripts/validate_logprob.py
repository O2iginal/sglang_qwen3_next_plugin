#!/usr/bin/env python3
"""
对本地 SGLang 服务做最小 logprob / 近似 lm loss 验收。

说明：
- 这里使用服务端返回的 prompt token logprob
- 以平均负对数似然（avg_nll）作为近似指标
- 默认阈值偏宽松，目的是排除“模型路径明显坏掉”的情况
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import urllib.error
import urllib.request


DEFAULT_CASES = [
    {
        "name": "zh_definition",
        "prompt": "请用两句话解释什么是机器学习。",
    },
    {
        "name": "en_math",
        "prompt": "What is 2 + 3? Answer briefly.",
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


def compute_avg_nll(input_token_logprobs: list) -> float:
    values = []
    for item in input_token_logprobs:
        if not item:
            continue
        logprob = item[0]
        if logprob is None:
            continue
        values.append(-float(logprob))
    if not values:
        raise ValueError("no valid input token logprobs found")
    return sum(values) / len(values)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=30110)
    parser.add_argument(
        "--max-avg-nll",
        type=float,
        default=8.0,
        help="平均负对数似然阈值；默认值用于筛除明显坏掉的运行路径。",
    )
    args = parser.parse_args()

    url = f"http://{args.host}:{args.port}/generate"
    failed = False

    for case in DEFAULT_CASES:
        payload = {
            "text": case["prompt"],
            "sampling_params": {
                "temperature": 0.0,
                "max_new_tokens": 1,
            },
            "return_logprob": True,
            "logprob_start_len": 0,
        }
        try:
            result = request_json(url, payload)
        except urllib.error.URLError as exc:
            print(f"[FAIL] {case['name']}: request error: {exc}")
            failed = True
            continue

        meta_info = result.get("meta_info", {})
        input_token_logprobs = meta_info.get("input_token_logprobs", [])
        avg_nll = compute_avg_nll(input_token_logprobs)
        ppl = math.exp(min(avg_nll, 20.0))
        ok = avg_nll <= args.max_avg_nll
        status = "PASS" if ok else "FAIL"
        print(f"[{status}] {case['name']}")
        print(f"prompt: {case['prompt']}")
        print(f"avg_nll: {avg_nll:.4f}")
        print(f"approx_ppl: {ppl:.4f}")
        print(f"tokens_scored: {len(input_token_logprobs)}")
        print()
        failed = failed or (not ok)

    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
