"""Pluggable token counting. Exactness ladder:
heuristic (free, ~±15%) -> tiktoken (fast, GPT-family exact) ->
anthropic count_tokens API (exact, network call - off the hot path only).
"""

from __future__ import annotations
from typing import Callable

TokenCounter = Callable[[str], int]


def heuristic_counter(text: str) -> int:
    """chars/4 blended with words*1.3 - good enough for composition ratios."""
    if not text:
        return 0
    by_chars = len(text) / 4.0
    by_words = len(text.split()) * 1.3
    return max(1, int((by_chars + by_words) / 2))


def tiktoken_counter(encoding: str = "cl100k_base") -> TokenCounter:
    import tiktoken
    enc = tiktoken.get_encoding(encoding)
    return lambda text: len(enc.encode(text)) if text else 0


def anthropic_counter(client, model: str) -> TokenCounter:
    """Exact via the count_tokens endpoint. Use for offline re-scoring of a
    ledger (`contextlens rescore`), never inline in the agent loop."""
    def count(text: str) -> int:
        if not text:
            return 0
        r = client.messages.count_tokens(
            model=model, messages=[{"role": "user", "content": text}])
        return r.input_tokens
    return count
