"""report.py -- render a ledger to a standalone HTML file."""

from __future__ import annotations

import json
from pathlib import Path

TEMPLATE = Path(__file__).parent / "report_template.html"


def load_ledger(path):
    header, turns, quality = None, [], []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        obj = json.loads(line)
        kind = obj.get("kind")
        if kind == "session":
            header = obj
        elif kind == "turn":
            turns.append(obj)
        elif kind == "quality":
            quality.append(obj)
    return header, turns, quality


def render(ledger_path, out_path="ctxscope-report.html"):
    header, turns, quality = load_ledger(ledger_path)
    html = TEMPLATE.read_text(encoding="utf-8").replace(
        "/*__DATA__*/",
        f"const SESSION={json.dumps(header)};"
        f"const TURNS={json.dumps(turns)};"
        f"const QUALITY={json.dumps(quality)};",
    )
    out_path = Path(out_path)
    out_path.write_text(html, encoding="utf-8")
    return str(out_path)
