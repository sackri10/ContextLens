"""Classify a message into a source category.

Order of precedence:
1. user-registered classifier hook (returns a source string or None)
2. structural heuristics (content block types)
3. compaction markers (registered prefixes)
4. role fallback
"""

from __future__ import annotations
from typing import Callable, Optional

Classifier = Callable[[dict, int], Optional[str]]

DEFAULT_COMPACTION_MARKERS = (
    "[conversation summary]", "summary of prior conversation",
    "<compacted>", "previous context (summarized)",
)


def classify(message: dict, index: int, *,
             hooks: list[Classifier] = (),
             compaction_markers: tuple[str, ...] = DEFAULT_COMPACTION_MARKERS,
             ) -> tuple[str, Optional[str]]:
    """Returns (source, tool_name)."""
    for hook in hooks:
        result = hook(message, index)
        if result:
            return result, None

    role = message.get("role", "other")
    content = message.get("content")

    if role == "system":
        return "system", None

    # Anthropic-style content block lists
    if isinstance(content, list):
        types = {b.get("type") for b in content if isinstance(b, dict)}
        if "tool_result" in types:
            return "tool_result", _first_tool_id(content)
        if "tool_use" in types:
            return "assistant_action", _first_tool_name(content)

    text = content if isinstance(content, str) else ""
    lowered = text[:200].lower()
    if any(m in lowered for m in compaction_markers):
        return "compaction_summary", None

    if role in ("user", "assistant"):
        return role, None
    return "other", None


def _first_tool_name(blocks) -> Optional[str]:
    for b in blocks:
        if isinstance(b, dict) and b.get("type") == "tool_use":
            return b.get("name")
    return None


def _first_tool_id(blocks) -> Optional[str]:
    for b in blocks:
        if isinstance(b, dict) and b.get("type") == "tool_result":
            return b.get("tool_use_id")
    return None
