"""CompactionQualityChecker: does the summary still answer what the
original context could? Probe-based scoring with any LLM callable."""

from __future__ import annotations
import json
from dataclasses import dataclass, field
from typing import Callable

LLM = Callable[[str], str]   # prompt in, text out -- bring any model

PROBE_PROMPT = """From the following conversation context, write {n} short
factual questions that this context answers, with their answers.
Respond ONLY with JSON: [{{"q": "...", "a": "..."}}].

CONTEXT:
{context}"""

ANSWER_PROMPT = """Answer the question using ONLY the summary below.
If the summary does not contain the answer, reply exactly: UNKNOWN.

SUMMARY:
{summary}

QUESTION: {question}
Answer in one sentence."""

GRADE_PROMPT = """Reference answer: {ref}
Candidate answer: {cand}
Is the candidate factually consistent with the reference? Reply ONLY "yes" or "no"."""


@dataclass
class QualityReport:
    score: float                 # 0..1 -- fraction of probes survived
    n_probes: int
    failures: list[dict] = field(default_factory=list)

    def to_dict(self):
        return {"score": self.score, "n_probes": self.n_probes,
                "failures": self.failures}


def _parse_json(text: str):
    text = text.strip().removeprefix("```json").removeprefix("```").removesuffix("```")
    return json.loads(text.strip())


class CompactionQualityChecker:
    def __init__(self, llm: LLM, n_probes: int = 5):
        self.llm, self.n = llm, n_probes

    def check(self, original_text: str, summary_text: str) -> QualityReport:
        probes = _parse_json(self.llm(
            PROBE_PROMPT.format(n=self.n, context=original_text[:20000])))
        failures, passed = [], 0
        for p in probes[: self.n]:
            cand = self.llm(ANSWER_PROMPT.format(
                summary=summary_text, question=p["q"])).strip()
            if cand.upper().startswith("UNKNOWN"):
                ok = False
            else:
                ok = self.llm(GRADE_PROMPT.format(
                    ref=p["a"], cand=cand)).strip().lower().startswith("yes")
            if ok:
                passed += 1
            else:
                failures.append({"q": p["q"], "expected": p["a"], "got": cand})
        n = max(len(probes[: self.n]), 1)
        return QualityReport(score=passed / n, n_probes=n, failures=failures)


def anthropic_llm(model: str = "claude-haiku-4-5") -> LLM:
    """Convenience LLM callable backed by the Anthropic SDK."""
    from anthropic import Anthropic
    client = Anthropic()

    def call(prompt: str) -> str:
        resp = client.messages.create(
            model=model, max_tokens=1024,
            messages=[{"role": "user", "content": prompt}],
        )
        return "".join(b.text for b in resp.content if getattr(b, "type", "") == "text")

    return call


def score_ledger(ledger_path, llm: LLM, n_probes: int = 5):
    """For each compaction event in the ledger, reconstruct the evicted text
    from the last turn those blocks were present, grade the summary that
    replaced them, and append a `{"kind": "quality", ...}` line to the file.
    """
    from pathlib import Path
    from .report import load_ledger

    header, turns, _ = load_ledger(ledger_path)
    checker = CompactionQualityChecker(llm, n_probes=n_probes)

    # index: block id -> preview/text last seen, by scanning all turns
    block_text: dict[str, str] = {}
    for t in turns:
        for b in t["blocks"]:
            block_text[b["id"]] = b["preview"]

    reports = []
    for t in turns:
        for e in t.get("events", []):
            if e["type"] != "compaction":
                continue
            evicted_ids = []
            # the evicted ids are the ones in the same turn's "evicted" event
            for e2 in t["events"]:
                if e2["type"] == "evicted":
                    evicted_ids = e2["block_ids"]
            original_text = "\n".join(block_text.get(bid, "") for bid in evicted_ids)
            summary_ids = e["block_ids"]
            summary_text = "\n".join(block_text.get(bid, "") for bid in summary_ids)
            report = checker.check(original_text, summary_text)
            line = {"kind": "quality", "turn": t["turn"], **report.to_dict()}
            reports.append(line)

    if reports:
        with open(Path(ledger_path), "a", encoding="utf-8") as fh:
            for line in reports:
                fh.write(json.dumps(line, ensure_ascii=False) + "\n")
    return reports
