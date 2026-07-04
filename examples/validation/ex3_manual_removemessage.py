"""Manual RemoveMessage pattern (docs-style DIY compaction node).

The canonical hand-rolled pattern: an LLM writes a summary, RemoveMessage
deletes all but the last two messages. This is the test case for
*classifier hooks*, because DIY summaries carry team-specific markers that
the default compaction-marker heuristics won't match.

Run once with HOOK_ENABLED = False: evictions are detected but there is no
`compaction` event -- the summary gets misclassified as plain `user` text.
Run again with HOOK_ENABLED = True: the `compaction` event fires. Comparing
the two ledgers/reports side by side IS the hook-API documentation.
"""

from __future__ import annotations

import os

from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, RemoveMessage
from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.checkpoint.memory import InMemorySaver

from harness import make_profiler, finish

MARKER = "MEMO:"
HOOK_ENABLED = os.environ.get("EX3_HOOK", "1") != "0"
LABEL = "ex3-manual" + ("-with-hook" if HOOK_ENABLED else "-no-hook")


def main():
    profiler, handler = make_profiler(LABEL)

    if HOOK_ENABLED:
        # Register the hook BEFORE running -- this is what flips detection on.
        profiler.add_classifier(
            lambda msg, i: "compaction_summary"
            if isinstance(msg.get("content"), str) and msg["content"].startswith(MARKER)
            else None
        )

    model = init_chat_model("anthropic:claude-sonnet-4-6")

    class State(MessagesState):
        summary: str

    def call_model(state: State):
        return {"messages": [model.invoke(state["messages"])]}

    def should_summarize(state: State):
        return "summarize" if len(state["messages"]) > 6 else END

    def summarize(state: State):
        prompt = state["messages"] + [
            HumanMessage(content="Summarize the conversation above in 5 bullet points.")]
        summary = model.invoke(prompt).content
        deletes = [RemoveMessage(id=m.id) for m in state["messages"][:-2]]
        return {"summary": summary,
                "messages": deletes + [HumanMessage(content=f"{MARKER} {summary}")]}

    builder = StateGraph(State)
    builder.add_node("llm", call_model)
    builder.add_node("summarize", summarize)
    builder.add_edge(START, "llm")
    builder.add_conditional_edges("llm", should_summarize, {"summarize": "summarize", END: END})
    builder.add_edge("summarize", END)
    graph = builder.compile(checkpointer=InMemorySaver())

    config = {"configurable": {"thread_id": "t1"}, "callbacks": [handler]}
    for i in range(8):
        graph.invoke({"messages": [("user", f"Explain CAP theorem aspect #{i} in 100 words.")]},
                      config)

    finish(profiler, LABEL)


if __name__ == "__main__":
    main()
