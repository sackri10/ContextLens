"""Shared "context bloater" tools -- deterministic large tool results so
compaction triggers reliably at low thresholds."""

from __future__ import annotations

from langchain_core.tools import tool


@tool
def fetch_report(section: str) -> str:
    """Fetch a section of the quarterly report (returns a large payload)."""
    return f"REPORT SECTION {section}: " + ("revenue detail row; " * 400)


@tool
def run_sql(query: str) -> str:
    """Run a SQL query against the warehouse (returns many rows)."""
    return "rows: " + "; ".join(f"(id={i}, amt={i * 7})" for i in range(500))
