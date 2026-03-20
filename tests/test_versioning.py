import pathlib

from sglang_qwen3_next_plugin import versioning


def test_version_key_is_normalized() -> None:
    assert versioning.normalize_version_key("0.5.2") == "sglang_0_5_2"
    assert versioning.normalize_version_key("0.5.9") == "sglang_0_5_9"


def test_version_key_uses_numeric_prefix_only() -> None:
    assert versioning.normalize_version_key("0.5.9.post1") == "sglang_0_5_9"
    assert versioning.normalize_version_key("0.6.0rc2") == "sglang_0_6_0"


def test_detect_external_model_package_support_from_registry_source(tmp_path) -> None:
    registry_path = tmp_path / "registry.py"
    registry_path.write_text(
        "ModelRegistry.register('sglang.srt.models')\n"
        "if external_pkg := envs.SGLANG_EXTERNAL_MODEL_PACKAGE.get():\n"
        "    ModelRegistry.register(external_pkg, overwrite=True)\n"
    )

    assert versioning.registry_supports_external_model_package(registry_path) is True


def test_detect_missing_external_model_package_support_from_registry_source(
    tmp_path,
) -> None:
    registry_path = tmp_path / "registry.py"
    registry_path.write_text("ModelRegistry.register('sglang.srt.models')\n")

    assert versioning.registry_supports_external_model_package(registry_path) is False


def test_upstream_variant_paths_are_declared() -> None:
    mapping = versioning.get_upstream_variant_paths(pathlib.Path.cwd())

    assert mapping["sglang_0_5_2"].name == "qwen3_next.py"
    assert mapping["sglang_0_5_9"].name == "qwen3_next.py"


def test_new_upstream_baseline_files_exist() -> None:
    base = pathlib.Path.cwd() / "sglang_qwen3_next_plugin" / "upstream" / "sglang_0_5_9"

    assert (base / "qwen3_next.py").exists()
    assert (base / "configs_qwen3_next.py").exists()


def test_old_upstream_baseline_files_exist() -> None:
    base = pathlib.Path.cwd() / "sglang_qwen3_next_plugin" / "upstream" / "sglang_0_5_2"

    assert (base / "qwen3_next.py").exists()


def test_supported_version_key_must_be_declared() -> None:
    assert versioning.get_supported_version_key("0.5.2") == "sglang_0_5_2"
    assert versioning.get_supported_version_key("0.5.9") == "sglang_0_5_9"


def test_compat_module_paths_are_declared() -> None:
    mapping = versioning.get_compat_module_paths(pathlib.Path.cwd())

    assert mapping["sglang_0_5_2"].name == "sglang_0_5_2.py"
    assert mapping["sglang_0_5_9"].name == "sglang_0_5_9.py"


def test_variant_module_names_are_declared() -> None:
    mapping = versioning.get_variant_module_names()

    assert (
        mapping["sglang_0_5_2"]
        == "sglang_qwen3_next_plugin.variants.sglang_0_5_2"
    )
    assert (
        mapping["sglang_0_5_9"]
        == "sglang_qwen3_next_plugin.variants.sglang_0_5_9"
    )


def test_active_variant_module_name_follows_supported_version() -> None:
    assert (
        versioning.get_active_variant_module_name("0.5.2")
        == "sglang_qwen3_next_plugin.variants.sglang_0_5_2"
    )
    assert (
        versioning.get_active_variant_module_name("0.5.9")
        == "sglang_qwen3_next_plugin.variants.sglang_0_5_9"
    )
