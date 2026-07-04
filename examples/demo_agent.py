"""Simulated 24-turn Text-to-SQL agent session -- no API key required.

Demonstrates: a growing context, two large `run_sql` tool results, staleness
as those results age untouched, a mid-session compaction, and post-compaction
growth. Produces a JSONL ledger and renders it to demo-report.html.
"""

from __future__ import annotations

from pathlib import Path

from contextlens import ContextProfiler

SYSTEM_PROMPT = (
    "You are a SQL analyst agent. You have access to a `run_sql` tool "
    "against a sales database. Always cite the query you ran."
)

HERE = Path(__file__).parent


def user(content):
    return {"role": "user", "content": content}


def assistant(content):
    return {"role": "assistant", "content": content}


def tool_use(call_id, name, sql):
    return {"role": "assistant", "content": [
        {"type": "tool_use", "id": call_id, "name": name, "input": {"query": sql}},
    ]}


def tool_result(call_id, rows_text):
    return {"role": "user", "content": [
        {"type": "tool_result", "tool_use_id": call_id, "content": rows_text},
    ]}


def fake_rows(n_rows, seed):
    """A chunky, realistic-looking result blob."""
    lines = [f"row {seed}-{i}: region=US-{i % 4} amount={100 + i * 7} qty={i % 9}"
              for i in range(n_rows)]
    return "[" + ", ".join(lines) + "]"


def _record(profiler, history, node, output_tokens):
    profiler.record_turn(
        history, system=SYSTEM_PROMPT, model="claude-sonnet-4-6",
        usage={"input_tokens": sum(len(str(m)) for m in history) // 4,
               "output_tokens": output_tokens},
        meta={"node": node},
    )


def main():
    ledger_path = HERE / "demo.jsonl"
    ledger_path.unlink(missing_ok=True)
    profiler = ContextProfiler(ledger_path, label="text2sql-run-42")
    history: list[dict] = []

    questions = [
        "What were total sales last quarter?",
        "Break that down by region.",
        "Which product line grew fastest?",
        "Show me the top 10 customers by revenue.",
        "What's the average order size this year?",
        "Compare Q1 vs Q2 for the Northeast region.",
        "Which reps closed the most deals in June?",
        "What's the return rate on the Orders table?",
    ]

    # Turns 1-8: conversation grows; two large run_sql results land (turns 3, 6)
    for i, q in enumerate(questions, start=1):
        call_id = f"call_{i}"
        sql = f"SELECT * FROM sales WHERE quarter = {i};"
        big = i in (3, 6)
        rows = fake_rows(140 if big else 12, seed=i)

        history.append(user(q))
        history.append(tool_use(call_id, "run_sql", sql))
        history.append(tool_result(call_id, rows))
        history.append(assistant(f"Answer to Q{i}: based on `{sql}`, here's the summary."))

        _record(profiler, history, "sql_writer", 120)

    # Turns 9-14: no new material enters context -- turns 1-8 results go stale.
    for _ in range(9, 15):
        _record(profiler, history, "sql_writer", 40)

    # Turn 15: compaction -- fold the first six exchanges (24 blocks) into one
    # `[conversation summary]` block, keep the last two exchanges live.
    kept_exchanges = history[24:]
    summary = user(
        "[conversation summary] User asked about Q1-Q6 sales totals, regional "
        "breakdowns, fastest-growing product line, and top 10 customers. Answers "
        "were derived from `sales` and `orders` tables filtered by quarter."
    )
    history = [summary] + kept_exchanges
    _record(profiler, history, "compactor", 60)

    # Turns 16-24: growth resumes on top of the compacted base.
    more_questions = [
        "Now show returns by product category.",
        "Which region has the highest churn?",
        "What's the forecast for next quarter?",
        "Break down revenue by sales channel.",
        "Which SKUs are most frequently returned?",
        "What's our average discount rate?",
        "Show deal velocity by rep.",
        "Summarize everything for the exec review.",
        "One more: what's the YoY growth rate?",
    ]
    for j, q in enumerate(more_questions, start=16):
        call_id = f"call_{j}"
        sql = f"SELECT * FROM orders WHERE turn = {j};"
        rows = fake_rows(15, seed=j)

        history.append(user(q))
        history.append(tool_use(call_id, "run_sql", sql))
        history.append(tool_result(call_id, rows))
        history.append(assistant(f"Answer to turn {j}: based on `{sql}`."))

        _record(profiler, history, "sql_writer", 100)

    profiler.close()

    from contextlens.report import render
    out = render(ledger_path, HERE / "demo-report.html")
    print(f"Ledger: {ledger_path}")
    print(f"Report: {out}")


if __name__ == "__main__":
    main()
