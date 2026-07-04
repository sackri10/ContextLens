"""trim_messages -- the negative control.

Pure deletion, no summarization. The detector must emit evictions and
*zero* compaction events. Run this after ex1-ex3 to prove the compaction
heuristic doesn't fire on eviction alone -- it needs a summary-classified
block entering in the same turn.
"""

from __future__ import annotations

from langchain.chat_models import init_chat_model
from langchain_core.messages.utils import trim_messages, count_tokens_approximately
from langgraph.graph import StateGraph, START, MessagesState
from langgraph.checkpoint.memory import InMemorySaver

from harness import make_profiler, finish

LABEL = "ex4-trim"


def main():
    profiler, handler = make_profiler(LABEL)

    model = init_chat_model("anthropic:claude-sonnet-4-6")

    def call_model(state: MessagesState):
        trimmed = trim_messages(
            state["messages"],
            strategy="last",
            token_counter=count_tokens_approximately,
            max_tokens=1500,
            start_on="human",
            allow_partial=False,
        )
        return {"messages": [model.invoke(trimmed)]}

    builder = StateGraph(MessagesState)
    builder.add_node("llm", call_model)
    builder.add_edge(START, "llm")
    graph = builder.compile(checkpointer=InMemorySaver())

    config = {"configurable": {"thread_id": "t1"}, "callbacks": [handler]}
    for i in range(10):
        graph.invoke({"messages": [("user", f"Fact #{i} about Kafka, ~120 words.")]}, config)

    finish(profiler, LABEL)
    _assert_negative_control(LABEL)


def _assert_negative_control(label: str):
    """The assertion from the validation spec: eviction yes, compaction no."""
    import json

    turns = [json.loads(l) for l in open(f"{label}.jsonl") if '"turn"' in l]
    assert any(e["type"] == "evicted" for t in turns for e in t["events"]), \
        "expected at least one eviction from trim_messages"
    assert not any(e["type"] == "compaction" for t in turns for e in t["events"]), \
        "trim_messages must never produce a false-positive compaction event"
    print(f"{label}: negative control passed -- evictions yes, compaction no")


if __name__ == "__main__":
    main()
