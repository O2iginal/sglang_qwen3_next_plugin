from pathlib import Path


def test_codex_skill_exists_and_mentions_runtime_lessons() -> None:
    skill_path = Path(".codex/skills/sglang-plugin-version-adaptation/SKILL.md")
    assert skill_path.exists()

    text = skill_path.read_text()
    assert "vllm._C" in text
    assert "python scripts/run_acceptance.py --host 127.0.0.1 --port 30110" in text
    assert "卸载" in text or "移除" in text
    assert "0.5.2" in text and "0.5.9" in text
    assert "dtype" in text or "backend" in text


def test_skill_mentions_service_level_acceptance() -> None:
    text = Path(".codex/skills/sglang-plugin-version-adaptation/SKILL.md").read_text()

    assert "scripts/check_plugin_import.py" in text
    assert "validate_generation.py" in text
    assert "validate_logprob.py" in text
