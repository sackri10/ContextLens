"""LangChain SummarizationMiddleware (auto-compaction) -- primary target.

The middleware monitors token counts and summarizes older messages when the
trigger threshold is hit, keeping recent messages and keeping AI/tool pairs
together. Low thresholds below force compaction within a few turns.

Expected ledger signature:
  - Early turns: `tool_result` band grows steeply in the composition stream
  - Compaction turn: `evicted` event (many block ids) AND a new block
    classified `compaction_summary` entering in the same turn -> a
    `compaction` event with a `ratio` in its detail
  - The final probe question ("what did section A say") is a manual quality
    check -- compare the agent's answer against the pre-compaction ledger

Classifier note: LangChain's summary message text may not match the default
markers. Inspect the summary block's `preview` in the ledger and register a
hook if needed (see the commented example below).
"""

from __future__ import annotations

from dotenv import load_dotenv

from langchain.agents import create_agent
from langchain.agents.middleware import SummarizationMiddleware

from harness import make_profiler, finish
from tools import fetch_report, run_sql

LABEL = "ex1-middleware"
load_dotenv()

QUESTIONS = [
    "Fetch report section A and summarize revenue.",
    "Now fetch section B and compare with A.",
    "Run a SQL query for top customers and relate to section B.",
    "Fetch section C. Which section had highest revenue overall?",
    "What did section A say about revenue, specifically?",   # post-compaction recall probe
]


def main():
    profiler, handler = make_profiler(LABEL)

    # Uncomment if the default compaction markers miss LangChain's summary text:
    # profiler.add_classifier(
    #     lambda msg, i: "compaction_summary"
    #     if isinstance(msg.get("content"), str)
    #     and "summary of the conversation" in msg["content"][:200].lower()
    #     else None
    # )

    agent = create_agent(
        model="anthropic:claude-sonnet-4-6",
        tools=[fetch_report, run_sql],
        middleware=[
            SummarizationMiddleware(
                model="anthropic:claude-haiku-4-5",   # cheap summarizer
                trigger=("tokens", 4000),             # compact past 4K tokens
                keep=("messages", 4),                 # keep last 4 messages verbatim
            ),
        ],
    )

    state = {"messages": []}
    for q in QUESTIONS:
        state["messages"].append({"role": "user", "content": q})
        state = agent.invoke(state, config={"callbacks": [handler]})

    finish(profiler, LABEL)


if __name__ == "__main__":
    main()
