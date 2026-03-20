from pathlib import Path


def test_chat_smoke_script_exists_and_targets_chat_completions() -> None:
    script = Path("scripts/chat_smoke.sh")
    assert script.exists()

    text = script.read_text()
    assert "/v1/chat/completions" in text
    assert 'CHAT_PROMPT=${CHAT_PROMPT:-简短介绍一下你自己，用中文回答}' in text
    assert 'CHAT_MODEL=${CHAT_MODEL:-default}' in text
    assert 'CHAT_PORT=${CHAT_PORT:-30110}' in text
    assert '"stream": false' in text


def test_readme_mentions_chat_smoke_script() -> None:
    text = Path("README.md").read_text()

    assert "scripts/chat_smoke.sh" in text
