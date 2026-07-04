"""ContextLens: a context window profiler for LLM agents.

Public API:
    ContextProfiler   - core recorder, turns message lists into a JSONL ledger
    wrap_anthropic     - one-line adoption for the Anthropic SDK
"""

from __future__ import annotations

from .profiler import ContextProfiler
from .models import ContextBlock, Event, TurnRecord, SessionHeader
from .integrations.anthropic_client import wrap_anthropic, wrap_anthropic_beta

__all__ = [
    "ContextProfiler",
    "ContextBlock",
    "Event",
    "TurnRecord",
    "SessionHeader",
    "wrap_anthropic",
    "wrap_anthropic_beta",
]

__version__ = "0.1.0"
