"""ctxscope CLI."""

from __future__ import annotations

import argparse
import json
import sys


def main():
    p = argparse.ArgumentParser(prog="ctxscope")
    sub = p.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("report", help="render ledger to standalone HTML")
    r.add_argument("ledger")
    r.add_argument("-o", "--out", default="ctxscope-report.html")

    s = sub.add_parser("stats", help="print summary stats to stdout")
    s.add_argument("ledger")

    q = sub.add_parser("quality", help="score compaction events in a ledger")
    q.add_argument("ledger")
    q.add_argument("--model", default="claude-haiku-4-5")
    q.add_argument("--n-probes", type=int, default=5)

    args = p.parse_args()
    if args.cmd == "report":
        from .report import render
        print(render(args.ledger, args.out))
    elif args.cmd == "stats":
        _stats(args.ledger)
    elif args.cmd == "quality":
        _quality(args.ledger, args.model, args.n_probes)


def _stats(path):
    from .report import load_ledger
    header, turns, _ = load_ledger(path)
    if header is None:
        print(f"error: no session header found in {path}", file=sys.stderr)
        raise SystemExit(1)
    peak = max((t["totals"]["tokens"] for t in turns), default=0)
    comps = sum(1 for t in turns for e in t["events"] if e["type"] == "compaction")
    evictions = sum(1 for t in turns for e in t["events"] if e["type"] == "evicted")
    reversible = sum(
        e["tokens"] for t in turns for e in t["events"] if e["type"] == "reversible_evict")
    print(json.dumps({"session": header["label"], "turns": len(turns),
                       "peak_tokens": peak, "compactions": comps,
                       "evictions": evictions,
                       "reversible_saved_tokens": reversible}, indent=2))


def _quality(path, model, n_probes):
    from .quality import anthropic_llm, score_ledger
    llm = anthropic_llm(model)
    reports = score_ledger(path, llm, n_probes=n_probes)
    if not reports:
        print("no compaction events found in ledger")
        return
    for r in reports:
        print(json.dumps(r))
