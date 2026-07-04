from dataclasses import dataclass

from ctxscope import ContextProfiler
from ctxscope.integrations.headroom_adapter import wrap_headroom_compress


@dataclass
class FakeCompressResult:
    tokens_before: int
    tokens_after: int
    messages: list


def fake_compress(messages, **kwargs):
    return FakeCompressResult(tokens_before=3000, tokens_after=400, messages=messages)


def test_reversible_evict_event(tmp_path):
    p = ContextProfiler(tmp_path / "l.jsonl", strict=True)
    compress = wrap_headroom_compress(fake_compress, p)

    compress([{"role": "user", "content": "big tool result"}])
    rec = p.record_turn([{"role": "user", "content": "compressed stand-in"}])

    evts = [e for e in rec.events if e.type == "reversible_evict"]
    assert len(evts) == 1
    assert evts[0].tokens == 2600
    assert "retrievable" in evts[0].detail


def test_no_event_when_nothing_saved(tmp_path):
    p = ContextProfiler(tmp_path / "l.jsonl", strict=True)
    compress = wrap_headroom_compress(
        lambda messages, **kw: FakeCompressResult(100, 100, messages), p)

    compress([{"role": "user", "content": "small"}])
    rec = p.record_turn([{"role": "user", "content": "small"}])

    assert not any(e.type == "reversible_evict" for e in rec.events)


def test_reversible_evict_never_classified_as_compaction(tmp_path):
    """A reversible_evict must never be mistaken for a lossy compaction --
    that would misrepresent retrievable data as gone for good."""
    p = ContextProfiler(tmp_path / "l.jsonl", strict=True)
    compress = wrap_headroom_compress(fake_compress, p)

    compress([{"role": "user", "content": "big tool result"}])
    rec = p.record_turn([{"role": "user", "content": "compressed stand-in"}])

    assert not any(e.type == "compaction" for e in rec.events)


def test_pending_events_do_not_leak_into_next_turn_without_compress(tmp_path):
    """Once flushed, a queued event must not reappear on a later turn that
    had no compress() call."""
    p = ContextProfiler(tmp_path / "l.jsonl", strict=True)
    compress = wrap_headroom_compress(fake_compress, p)

    compress([{"role": "user", "content": "big tool result"}])
    p.record_turn([{"role": "user", "content": "turn with compression"}])
    rec2 = p.record_turn([{"role": "user", "content": "turn without compression"}])

    assert not any(e.type == "reversible_evict" for e in rec2.events)
