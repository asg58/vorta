"""Unit tests for the Claude conversation brain (no network: the client is faked)."""
import asyncio
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from shared.ai.claude_brain import BrainError, ClaudeBrain  # noqa: E402


class FakeClient:
    def __init__(self, text="Hallo!", stop_reason="end_turn"):
        self.calls = []
        self._text = text
        self._stop_reason = stop_reason
        self.beta = SimpleNamespace(messages=SimpleNamespace(create=self._create))

    async def _create(self, **kwargs):
        self.calls.append(kwargs)
        content = [SimpleNamespace(type="thinking", thinking="")]
        if self._text:
            content.append(SimpleNamespace(type="text", text=self._text))
        return SimpleNamespace(
            content=content,
            model=kwargs["model"],
            stop_reason=self._stop_reason,
            usage=SimpleNamespace(input_tokens=12, output_tokens=5),
        )


def run(coro):
    return asyncio.run(coro)


def test_reply_sends_history_and_returns_text():
    client = FakeClient("Het is drie uur.")
    brain = ClaudeBrain(client=client)
    reply = run(brain.reply([
        {"role": "assistant", "content": "stray leading turn"},
        {"role": "user", "content": "Hoi"},
        {"role": "assistant", "content": "Hallo!"},
        {"role": "user", "content": "  "},
        {"role": "user", "content": "Hoe laat is het?"},
    ]))

    assert reply.text == "Het is drie uur."
    assert not reply.refused
    call = client.calls[0]
    assert call["model"] == "claude-fable-5-2"
    assert call["fallbacks"] == "default"
    assert call["output_config"] == {"effort": "low"}
    assert "thinking" not in call and "temperature" not in call
    assert call["messages"] == [
        {"role": "user", "content": "Hoi"},
        {"role": "assistant", "content": "Hallo!"},
        {"role": "user", "content": "Hoe laat is het?"},
    ]


def test_model_and_effort_come_from_environment(monkeypatch):
    monkeypatch.setenv("VORTA_LLM_MODEL", "claude-opus-5-5")
    monkeypatch.setenv("VORTA_LLM_EFFORT", "high")
    client = FakeClient()
    run(ClaudeBrain(client=client).reply([{"role": "user", "content": "Hoi"}]))
    assert client.calls[0]["model"] == "claude-opus-5-5"
    assert client.calls[0]["output_config"] == {"effort": "high"}


def test_refusal_gives_a_speakable_reply():
    brain = ClaudeBrain(client=FakeClient(text="", stop_reason="refusal"))
    reply = run(brain.reply([{"role": "user", "content": "..."}]))
    assert reply.refused and reply.text


def test_empty_reply_and_bad_history_raise():
    with pytest.raises(BrainError):
        run(ClaudeBrain(client=FakeClient(text="")).reply([{"role": "user", "content": "Hoi"}]))
    with pytest.raises(BrainError):
        run(ClaudeBrain(client=FakeClient()).reply([]))


def test_missing_credentials_raise_brain_error(monkeypatch):
    for name in ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("HOME", "/nonexistent")
    with pytest.raises(BrainError):
        run(ClaudeBrain().reply([{"role": "user", "content": "Hoi"}]))
