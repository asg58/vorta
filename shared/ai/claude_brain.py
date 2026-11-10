"""
VORTA conversation brain, backed by Claude.

One small module shared by the inference engine (live voice pipeline) and the
dashboard (chat + voice agent), so every spoken or typed answer comes from a
real language model instead of a canned sentence.

Configuration (environment variables):
    ANTHROPIC_API_KEY   credentials for the Claude API (required)
    VORTA_LLM_MODEL     model id, default "claude-fable-5-2"
    VORTA_LLM_EFFORT    low | medium | high | xhigh | max, default "low"
                        (voice turns want a fast answer; raise it for harder work)
    VORTA_LLM_TIMEOUT   seconds before a turn is abandoned, default 60
    VORTA_SYSTEM_PROMPT replaces the default voice-assistant instructions
"""

import logging
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import anthropic

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "claude-fable-5-2"
DEFAULT_EFFORT = "low"
MAX_TOKENS = 16000  # includes the model's thinking; reply length is set by the prompt
FALLBACK_BETA = "server-side-fallback-2026-07-01"

# Kept free of timestamps and per-request values so the prompt stays identical
# between turns of a conversation.
DEFAULT_SYSTEM_PROMPT = """\
You are VORTA, a voice assistant. People talk to you out loud and your reply is \
read to them by a text-to-speech engine, so you are having a spoken conversation, \
not writing a document.

How to answer:
- Reply in the language the person is speaking. When that is unclear, use Dutch.
- Sound like a knowledgeable person talking: plain sentences, no markdown, no \
bullet points, no headings, no emoji, no URLs read out character by character. \
Write numbers, dates and abbreviations the way they should be pronounced.
- Lead with the answer. One to three sentences is right for most questions; go \
longer only when the person asks for an explanation or the topic needs it, and \
then offer to continue rather than delivering a monologue.
- The person's words reach you through speech recognition, which mishears names, \
numbers and short words. Read for the meaning they most likely intended. When a \
mishearing would change the answer, ask one short question to check.
- Use what was said earlier in the conversation. Do not repeat the question back \
or re-introduce yourself.
- Think the problem through properly before answering: give real reasoning, real \
advice and concrete next steps, not generic filler.
- Say so plainly when you do not know something or cannot do it. You have no live \
access to the internet, the clock, or the person's devices unless a message gives \
you that information.
"""


class BrainError(RuntimeError):
    """The model could not produce a reply (missing key, API failure, ...)."""


@dataclass
class BrainReply:
    text: str
    model: str
    stop_reason: Optional[str] = None
    refused: bool = False
    usage: Dict[str, int] = field(default_factory=dict)

    def metadata(self) -> Dict[str, Any]:
        return {
            "model": self.model,
            "stop_reason": self.stop_reason,
            "refused": self.refused,
            "usage": self.usage,
        }


def _clean_history(messages: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    """Keep non-empty user/assistant text turns, starting at the first user turn."""
    cleaned: List[Dict[str, str]] = []
    for message in messages:
        role = message.get("role")
        content = message.get("content")
        if role not in ("user", "assistant") or not isinstance(content, str):
            continue
        content = content.strip()
        if not content:
            continue
        if not cleaned and role != "user":
            continue
        cleaned.append({"role": role, "content": content})
    return cleaned


class ClaudeBrain:
    """Turns a conversation history into the assistant's next spoken reply."""

    def __init__(
        self,
        model: Optional[str] = None,
        effort: Optional[str] = None,
        system_prompt: Optional[str] = None,
        client: Optional[Any] = None,
    ):
        self.model = model or os.getenv("VORTA_LLM_MODEL") or DEFAULT_MODEL
        self.effort = effort or os.getenv("VORTA_LLM_EFFORT") or DEFAULT_EFFORT
        self.system_prompt = (
            system_prompt or os.getenv("VORTA_SYSTEM_PROMPT") or DEFAULT_SYSTEM_PROMPT
        )
        self._client = client

    @property
    def client(self) -> Any:
        if self._client is None:
            timeout = float(os.getenv("VORTA_LLM_TIMEOUT", "60"))
            try:
                self._client = anthropic.AsyncAnthropic(timeout=timeout)
            except anthropic.AnthropicError as e:
                raise BrainError(
                    "No Claude API credentials found. Set ANTHROPIC_API_KEY."
                ) from e
        return self._client

    async def reply(self, messages: List[Dict[str, Any]]) -> BrainReply:
        """
        Generate the next assistant turn.

        `messages` is the conversation so far as {"role", "content"} dicts,
        ending with the user's latest turn.
        """
        history = _clean_history(messages)
        if not history or history[-1]["role"] != "user":
            raise BrainError("The conversation must end with a user message.")

        try:
            # If the model's safety classifiers decline a turn, the API reruns
            # it on a fallback model inside the same call (fallbacks="default").
            response = await self.client.beta.messages.create(
                model=self.model,
                max_tokens=MAX_TOKENS,
                system=self.system_prompt,
                messages=history,
                output_config={"effort": self.effort},
                betas=[FALLBACK_BETA],
                fallbacks="default",
            )
        except anthropic.AuthenticationError as e:
            raise BrainError("The Claude API key was rejected. Check ANTHROPIC_API_KEY.") from e
        except anthropic.RateLimitError as e:
            raise BrainError("The Claude API rate limit was reached. Try again shortly.") from e
        except anthropic.APIStatusError as e:
            raise BrainError(f"Claude API error {e.status_code}: {e.message}") from e
        except anthropic.APIConnectionError as e:
            raise BrainError(f"Could not reach the Claude API: {e}") from e
        except anthropic.AnthropicError as e:
            raise BrainError(f"Claude client error: {e}") from e
        except TypeError as e:
            # The SDK raises TypeError when it finds no credentials to send.
            raise BrainError(
                f"Claude client is not configured (is ANTHROPIC_API_KEY set?): {e}"
            ) from e

        usage = getattr(response, "usage", None)
        reply = BrainReply(
            text="".join(
                block.text for block in response.content if block.type == "text"
            ).strip(),
            model=response.model,
            stop_reason=response.stop_reason,
            usage={
                "input_tokens": getattr(usage, "input_tokens", 0) or 0,
                "output_tokens": getattr(usage, "output_tokens", 0) or 0,
            },
        )

        if response.stop_reason == "refusal":
            reply.refused = True
            reply.text = reply.text or "Daar kan ik je helaas niet mee helpen."
        elif not reply.text:
            raise BrainError(
                f"The model returned no text (stop_reason={response.stop_reason})."
            )
        elif response.stop_reason == "max_tokens":
            logger.warning("Reply was cut off at max_tokens")

        return reply


_brain: Optional[ClaudeBrain] = None


def get_brain() -> ClaudeBrain:
    """Process-wide brain instance."""
    global _brain
    if _brain is None:
        _brain = ClaudeBrain()
    return _brain
