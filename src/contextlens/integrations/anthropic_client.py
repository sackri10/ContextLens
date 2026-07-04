"""wrap_anthropic(client, profiler): profile every messages.create call."""

from __future__ import annotations
import functools


def wrap_anthropic(client, profiler, *, meta_fn=None):
    """Monkeypatch client.messages.create to record each call.

    Records the *request* context (what the model actually saw) plus the
    response usage (exact input/output token counts).
    """
    original = client.messages.create

    @functools.wraps(original)
    def create(*args, **kwargs):
        response = original(*args, **kwargs)
        usage = getattr(response, "usage", None)
        profiler.record_turn(
            kwargs.get("messages", []),
            system=kwargs.get("system"),
            model=kwargs.get("model"),
            usage={"input_tokens": usage.input_tokens,
                   "output_tokens": usage.output_tokens} if usage else None,
            meta=meta_fn(kwargs) if meta_fn else None,
        )
        return response

    client.messages.create = create
    return client


def wrap_anthropic_beta(client, profiler, *, meta_fn=None):
    """Wrap client.beta.messages.create; captures server-side context edits.

    Anthropic's context editing (`clear_tool_uses_20250919`) and compaction
    (`compact_20260112`) run server-side -- the client `messages` list never
    changes, so `wrap_anthropic`'s hash-diffing sees nothing. This wrapper
    reads `response.context_management.applied_edits` and
    `response.stop_reason` instead, and passes them through as `meta` flags
    that `ContextProfiler._record` promotes into `server_edit` /
    `server_compaction` events.
    """
    original = client.beta.messages.create

    @functools.wraps(original)
    def create(*args, **kwargs):
        response = original(*args, **kwargs)
        usage = getattr(response, "usage", None)
        meta = dict(meta_fn(kwargs) if meta_fn else {})

        cm = getattr(response, "context_management", None)
        applied = list(getattr(cm, "applied_edits", []) or [])
        if applied:
            meta["server_edits"] = [
                {"type": getattr(e, "type", "?"),
                 "cleared_tool_uses": getattr(e, "cleared_tool_uses", None),
                 "cleared_input_tokens": getattr(e, "cleared_input_tokens", None)}
                for e in applied
            ]
        if getattr(response, "stop_reason", None) == "compaction":
            meta["server_compaction"] = True

        profiler.record_turn(
            kwargs.get("messages", []),
            system=kwargs.get("system"),
            model=kwargs.get("model"),
            usage={"input_tokens": usage.input_tokens,
                   "output_tokens": usage.output_tokens} if usage else None,
            meta=meta,
        )
        return response

    client.beta.messages.create = create
    return client
