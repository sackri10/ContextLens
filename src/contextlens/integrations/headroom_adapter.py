"""Wrap headroom.compress() so every call is profiled before the compressed
messages go to the model. Headroom reports tokens_before/after directly --
no need to infer compression from a hash diff, unlike every other integration.

Headroom (https://github.com/chopratejas/headroom) is a *reversible*
compression layer: it neither deletes nor summarizes tool outputs/logs/
files/RAG chunks, it compresses them and caches the original so the model
can retrieve it later. That makes it a third failure mode, distinct from
lossy `compaction` and plain `evicted` -- nothing is actually lost, so it
gets its own event type: `reversible_evict`.
"""

from __future__ import annotations

from contextlens.models import Event


def wrap_headroom_compress(compress_fn, profiler):
    """compress_fn: headroom.compress. Returns a wrapped callable with the
    same signature that also queues a `reversible_evict` event, which is
    flushed into the next record_turn() call on this profiler.
    """

    def compress(messages, **kwargs):
        result = compress_fn(messages, **kwargs)
        saved = result.tokens_before - result.tokens_after
        if saved > 0:
            ratio = saved / max(result.tokens_before, 1)
            profiler._pending_events = getattr(profiler, "_pending_events", [])
            profiler._pending_events.append(Event(
                type="reversible_evict",
                tokens=saved,
                detail=(f"headroom compressed {result.tokens_before}->"
                        f"{result.tokens_after} tok ({ratio:.0%} reduction); "
                        f"retrievable via headroom_retrieve"),
            ))
        return result

    return compress
