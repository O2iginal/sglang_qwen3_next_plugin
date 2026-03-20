from __future__ import annotations

import pathlib
import re


SUPPORTED_VERSION_KEYS = {
    "0.5.2": "sglang_0_5_2",
    "0.5.9": "sglang_0_5_9",
}


def normalize_version_key(version: str) -> str:
    match = re.match(r"^(\d+)\.(\d+)\.(\d+)", version)
    if match is None:
        raise ValueError(f"unsupported sglang version format: {version}")
    major, minor, patch = match.groups()
    return f"sglang_{major}_{minor}_{patch}"


def registry_supports_external_model_package(registry_path: pathlib.Path) -> bool:
    text = registry_path.read_text()
    return "SGLANG_EXTERNAL_MODEL_PACKAGE" in text


def get_upstream_variant_paths(repo_root: pathlib.Path) -> dict[str, pathlib.Path]:
    return {
        "sglang_0_5_2": repo_root
        / "sglang_qwen3_next_plugin"
        / "upstream"
        / "sglang_0_5_2"
        / "qwen3_next.py",
        "sglang_0_5_9": repo_root
        / "sglang_qwen3_next_plugin"
        / "upstream"
        / "sglang_0_5_9"
        / "qwen3_next.py",
    }


def get_supported_version_key(version: str) -> str:
    try:
        return SUPPORTED_VERSION_KEYS[version]
    except KeyError as exc:
        raise ValueError(f"unsupported sglang version: {version}") from exc


def get_compat_module_paths(repo_root: pathlib.Path) -> dict[str, pathlib.Path]:
    compat_root = repo_root / "sglang_qwen3_next_plugin" / "compat"
    return {
        "sglang_0_5_2": compat_root / "sglang_0_5_2.py",
        "sglang_0_5_9": compat_root / "sglang_0_5_9.py",
    }


def get_variant_module_names() -> dict[str, str]:
    return {
        "sglang_0_5_2": "sglang_qwen3_next_plugin.variants.sglang_0_5_2",
        "sglang_0_5_9": "sglang_qwen3_next_plugin.variants.sglang_0_5_9",
    }


def get_active_variant_module_name(version: str) -> str:
    version_key = get_supported_version_key(version)
    return get_variant_module_names()[version_key]
