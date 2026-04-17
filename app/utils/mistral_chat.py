from __future__ import annotations

from collections.abc import Sequence

from mistralai.client import Mistral


def _assistant_content_to_text(content: object) -> str:
    """
    Mistral SDK may return message.content as a str or as a list of content chunks
    (e.g. text + thinking). Normalize to a single string for JSON / plain-text callers.
    """
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(_assistant_content_to_text(item) for item in content)
    text = getattr(content, "text", None)
    if isinstance(text, str):
        return text
    thinking = getattr(content, "thinking", None)
    if thinking is not None:
        return _assistant_content_to_text(list(thinking))
    raw = getattr(content, "raw", None)
    if raw is not None:
        return _assistant_content_to_text(raw)
    if isinstance(content, dict):
        if "text" in content:
            return str(content["text"])
        if content.get("type") == "thinking" and isinstance(content.get("thinking"), list):
            return _assistant_content_to_text(content["thinking"])
    return ""


def mistral_chat_messages(
    api_key: str,
    model: str,
    messages: Sequence[dict[str, str]],
) -> str:
    """Run a Mistral chat completion and return the assistant message text."""
    client = Mistral(api_key=api_key)
    response = client.chat.complete(model=model, messages=list(messages))
    content = response.choices[0].message.content
    return _assistant_content_to_text(content).strip()
