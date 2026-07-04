"""Data models for the CtxScope ledger."""

from __future__ import annotations

import hashlib
import json
import time
import uuid
from dataclasses import dataclass, field, asdict
from typing import Any, Optional

SCHEMA_VERSION = 1

SOURCES = (
    "system", "user", "assistant", "assistant_action",
    "tool_result", "compaction_summary", "retrieval", "other",
)


def _normalize_content(content: Any) -> str:
    """Flatten message content (str or Anthropic-style block list) to text."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict):
                btype = block.get("type", "")
                if btype == "text":
                    parts.append(block.get("text", ""))
                elif btype == "tool_use":
                    parts.append(
                        f"[tool_use:{block.get('name','?')}] "
                        + json.dumps(block.get("input", {}), sort_keys=True, default=str)
                    )
                elif btype == "tool_result":
                    parts.append(
                        f"[tool_result:{block.get('tool_use_id','?')}] "
                        + _normalize_content(block.get("content"))
                    )
                else:
                    parts.append(json.dumps(block, sort_keys=True, default=str))
            else:
                parts.append(str(block))
        return "\n".join(parts)
    return str(content)


def block_id(role: str, content: Any) -> str:
    """Stable identity for a message across turns: hash(role + content)."""
    text = _normalize_content(content)
    return hashlib.sha1(f"{role}\x00{text}".encode("utf-8", "replace")).hexdigest()[:12]


@dataclass
class ContextBlock:
    id: str
    index: int
    role: str
    source: str
    tokens: int
    chars: int
    first_seen_turn: int
    age: int                      # turns since entry; 0 = entered this turn
    preview: str                  # first ~140 chars for the report UI
    tool_name: Optional[str] = None

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class Event:
    type: str                     # "entered" | "evicted" | "compaction"
    block_ids: list[str] = field(default_factory=list)
    tokens: int = 0
    detail: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class TurnRecord:
    kind: str
    turn: int
    ts: float
    model: Optional[str]
    blocks: list[ContextBlock]
    events: list[Event]
    totals: dict
    usage: Optional[dict]
    meta: dict

    def to_dict(self) -> dict:
        d = asdict(self)
        return d


@dataclass
class SessionHeader:
    kind: str
    schema: int
    session_id: str
    started_at: float
    label: str
    tokenizer: str

    @staticmethod
    def new(label: str, tokenizer: str) -> "SessionHeader":
        return SessionHeader(
            kind="session", schema=SCHEMA_VERSION,
            session_id=uuid.uuid4().hex[:10],
            started_at=time.time(), label=label, tokenizer=tokenizer,
        )

    def to_dict(self) -> dict:
        return asdict(self)
