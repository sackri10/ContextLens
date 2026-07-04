"""langmem SummarizationNode in a raw StateGraph.

SummarizationNode summarizes once cumulative tokens reach
`max_tokens_before_summary`, replacing older messages with a summary
message. Key subtlety: it keeps full history in `messages` and writes the
compacted view to `summarized_messages` -- but ContextWatch observes *what
the model was actually invoked with* (call_model reads `summarized_messages`),
so it sees the compacted list. This example validates exactly that
boundary-level claim.

Expected ledger signature: identical shape to ex1 (evictions + summary
entry), proving detection is mechanism-independent. Register a hook for
langmem's summary prefix if the default markers miss it.

Bonus validation -- reproduce langmem bug #111: wire SummarizationNode as
`pre_model_hook` on `create_react_agent` with a tool, trigger summarization
right after a tool invocation, and watch the `user` band vanish from the
composition stream while only `system` remains -- the bug caught in one
chart.
"""

from __future__ import annotations

from typing import Any, TypedDict

from langchain.chat_models import init_chat_model
from langchain_core.messages import AnyMessage
from langchain_core.messages.utils import count_tokens_approximately
from langgraph.graph import StateGraph, START, MessagesState
from langgraph.checkpoint.memory import InMemorySaver
from langmem.short_term import SummarizationNode

from harness import make_profiler, finish

LABEL = "ex2-langmem"


class State(MessagesState):
    context: dict[str, Any]


class LLMInput(TypedDict):
    summarized_messages: list[AnyMessage]
    context: dict[str, Any]


def main():
    profiler, handler = make_profiler(LABEL)

    # Uncomment if the default compaction markers miss langmem's summary text:
    # profiler.add_classifier(
    #     lambda msg, i: "compaction_summary"
    #     if isinstance(msg.get("content"), str)
    #     and msg["content"][:200].lower().startswith("summary")
    #     else None
    # )

    model = init_chat_model("anthropic:claude-sonnet-4-6")
    summarizer = init_chat_model("anthropic:claude-haiku-4-5").bind(max_tokens=256)

    summarization_node = SummarizationNode(
        token_counter=count_tokens_approximately,
        model=summarizer,
        max_tokens=2000,
        max_tokens_before_summary=2000,
        max_summary_tokens=256,
    )

    def call_model(state: LLMInput):
        response = model.invoke(state["summarized_messages"])   # compacted view
        return {"messages": [response]}

    builder = StateGraph(State)
    builder.add_node("summarize", summarization_node)
    builder.add_node("llm", call_model)
    builder.add_edge(START, "summarize")
    builder.add_edge("summarize", "llm")
    graph = builder.compile(checkpointer=InMemorySaver())

    config = {"configurable": {"thread_id": "t1"}, "callbacks": [handler]}
    for i in range(10):
        graph.invoke(
            {"messages": [("user", f"Tell me fact #{i} about distributed systems, "
                                    f"in ~150 words, and recall fact #{max(i - 3, 0)}.")]},
            config,
        )

    finish(profiler, LABEL)


if __name__ == "__main__":
    main()
