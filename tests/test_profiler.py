from contextwatch import ContextProfiler


def _turns(profiler):
    return profiler.records


def test_entry_and_eviction_detection(tmp_path):
    p = ContextProfiler(tmp_path / "l.jsonl", strict=True)
    m1 = [{"role": "user", "content": "hello"}]
    m2 = m1 + [{"role": "assistant", "content": "hi"},
               {"role": "user", "content": "run the query"}]
    p.record_turn(m1); p.record_turn(m2)
    # drop the first user message -> eviction
    p.record_turn(m2[1:])
    ev = [e for e in _turns(p)[2].events if e.type == "evicted"]
    assert ev and len(ev[0].block_ids) == 1


def test_compaction_detected(tmp_path):
    p = ContextProfiler(tmp_path / "l.jsonl", strict=True)
    old = [{"role": "user", "content": f"msg {i} " + "x" * 200} for i in range(5)]
    p.record_turn(old)
    compacted = [{"role": "user",
                  "content": "[conversation summary] user asked about msgs 0-4"}]
    rec = p.record_turn(compacted)
    assert any(e.type == "compaction" for e in rec.events)


def test_age_tracking(tmp_path):
    p = ContextProfiler(tmp_path / "l.jsonl", strict=True)
    msgs = [{"role": "system", "content": "you are helpful"}]
    for _ in range(3):
        p.record_turn(msgs)
    assert _turns(p)[-1].blocks[0].age == 2


def test_profiler_never_breaks_agent(tmp_path):
    p = ContextProfiler(tmp_path / "l.jsonl", strict=False)
    assert p.record_turn(None) is None        # garbage in, no exception out


def test_ledger_written_to_disk(tmp_path):
    path = tmp_path / "l.jsonl"
    p = ContextProfiler(path, strict=True, label="disk-test")
    p.record_turn([{"role": "user", "content": "hi"}])
    p.close()
    lines = path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2  # session header + 1 turn


def test_block_id_stable_across_turns():
    from contextwatch.models import block_id
    a = block_id("user", "hello")
    b = block_id("user", "hello")
    c = block_id("user", "hello world")
    assert a == b
    assert a != c


def test_source_classification():
    from contextwatch.classify import classify
    assert classify({"role": "system", "content": "sys"}, 0) == ("system", None)
    assert classify({"role": "user", "content": "hi"}, 0) == ("user", None)
    tool_use = {"role": "assistant", "content": [
        {"type": "tool_use", "name": "run_sql", "input": {}}]}
    assert classify(tool_use, 0) == ("assistant_action", "run_sql")
    tool_result = {"role": "user", "content": [
        {"type": "tool_result", "tool_use_id": "abc", "content": "rows"}]}
    assert classify(tool_result, 0) == ("tool_result", "abc")


def test_negative_control_trim_produces_no_compaction(tmp_path):
    """Pure deletion (e.g. trim_messages) must yield evictions but never a
    false-positive compaction event -- there's no summary block entering."""
    p = ContextProfiler(tmp_path / "l.jsonl", strict=True)
    old = [{"role": "user", "content": f"Fact #{i} about Kafka, " + "x" * 200}
           for i in range(6)]
    p.record_turn(old)
    trimmed = old[-2:]  # trim_messages: keep-last, no summary
    rec = p.record_turn(trimmed)
    assert any(e.type == "evicted" for e in rec.events)
    assert not any(e.type == "compaction" for e in rec.events)


def test_manual_marker_needs_hook_to_be_detected(tmp_path):
    """Docs-style manual RemoveMessage pattern: a DIY summary with a custom
    marker ("MEMO:") is invisible to the default classifier until a hook is
    registered -- reproduces the before/after from the manual-compaction
    validation example."""
    old = [{"role": "user", "content": f"msg {i} " + "x" * 200} for i in range(5)]
    compacted = [{"role": "user", "content": "MEMO: summary of the above"}]

    without_hook = ContextProfiler(tmp_path / "no_hook.jsonl", strict=True)
    without_hook.record_turn(old)
    rec = without_hook.record_turn(compacted)
    assert not any(e.type == "compaction" for e in rec.events)

    with_hook = ContextProfiler(tmp_path / "with_hook.jsonl", strict=True)
    with_hook.add_classifier(
        lambda msg, i: "compaction_summary"
        if isinstance(msg.get("content"), str) and msg["content"].startswith("MEMO:")
        else None)
    with_hook.record_turn(old)
    rec = with_hook.record_turn(compacted)
    assert any(e.type == "compaction" for e in rec.events)


def test_server_edit_event_promoted_from_meta(tmp_path):
    """Anthropic's server-side clear_tool_uses_20250919 never touches the
    client message list, so it's surfaced purely through `meta` (see
    wrap_anthropic_beta) and promoted to a first-class event here."""
    p = ContextProfiler(tmp_path / "l.jsonl", strict=True)
    rec = p.record_turn(
        [{"role": "user", "content": "hi"}],
        meta={"server_edits": [{"type": "clear_tool_uses_20250919",
                                 "cleared_tool_uses": 6,
                                 "cleared_input_tokens": 4000}]})
    edits = [e for e in rec.events if e.type == "server_edit"]
    assert len(edits) == 1
    assert edits[0].tokens == 4000
    assert "cleared 6 tool uses" in edits[0].detail


def test_server_compaction_event_promoted_from_meta(tmp_path):
    p = ContextProfiler(tmp_path / "l.jsonl", strict=True)
    rec = p.record_turn(
        [{"role": "user", "content": "hi"}],
        meta={"server_compaction": True})
    assert any(e.type == "server_compaction" for e in rec.events)


def test_wrap_anthropic_beta_captures_applied_edits(tmp_path):
    """No network call: a fake client stands in for anthropic.Anthropic()
    to prove wrap_anthropic_beta reads context_management.applied_edits and
    stop_reason off the response, not the request."""
    from types import SimpleNamespace
    from contextwatch.integrations.anthropic_client import wrap_anthropic_beta

    edit = SimpleNamespace(type="clear_tool_uses_20250919",
                            cleared_tool_uses=3, cleared_input_tokens=1200)
    response = SimpleNamespace(
        usage=SimpleNamespace(input_tokens=500, output_tokens=20),
        context_management=SimpleNamespace(applied_edits=[edit]),
        stop_reason="end_turn",
    )

    class FakeBeta:
        def create(self, **kwargs):
            return response

    client = SimpleNamespace(beta=SimpleNamespace(messages=FakeBeta()))
    p = ContextProfiler(tmp_path / "l.jsonl", strict=True)
    wrap_anthropic_beta(client, p)

    client.beta.messages.create(
        model="claude-sonnet-4-6",
        messages=[{"role": "user", "content": "hi"}])

    rec = p.records[-1]
    assert rec.usage == {"input_tokens": 500, "output_tokens": 20}
    server_edit = [e for e in rec.events if e.type == "server_edit"][0]
    assert server_edit.tokens == 1200
    assert "clear_tool_uses_20250919" in server_edit.detail


def test_wrap_anthropic_beta_captures_server_compaction(tmp_path):
    from types import SimpleNamespace
    from contextwatch.integrations.anthropic_client import wrap_anthropic_beta

    response = SimpleNamespace(
        usage=SimpleNamespace(input_tokens=300, output_tokens=10),
        context_management=None,
        stop_reason="compaction",
    )

    class FakeBeta:
        def create(self, **kwargs):
            return response

    client = SimpleNamespace(beta=SimpleNamespace(messages=FakeBeta()))
    p = ContextProfiler(tmp_path / "l.jsonl", strict=True)
    wrap_anthropic_beta(client, p)

    client.beta.messages.create(
        model="claude-sonnet-4-6",
        messages=[{"role": "user", "content": "hi"}])

    rec = p.records[-1]
    assert any(e.type == "server_compaction" for e in rec.events)
