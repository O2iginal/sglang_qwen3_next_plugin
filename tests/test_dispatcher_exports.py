import importlib


def test_dispatcher_reexports_variant_symbols() -> None:
    dispatcher = importlib.import_module("sglang_qwen3_next_plugin.qwen3_next")
    variant = importlib.import_module(
        "sglang_qwen3_next_plugin.variants.sglang_0_5_2"
    )

    assert dispatcher.Qwen3NextForCausalLM is variant.Qwen3NextForCausalLM
    assert dispatcher.Qwen3NextModel is variant.Qwen3NextModel

    if hasattr(variant, "gdn_with_output"):
        assert dispatcher.gdn_with_output is variant.gdn_with_output
