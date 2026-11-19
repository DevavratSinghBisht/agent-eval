"""Static HTML trajectory viewer: step timeline, reasoning, tool I/O, raw-vs-delivered diff with
the gold answer highlighted (so truncation losses are visible at a glance)."""

from __future__ import annotations

import html
import json
import re
from pathlib import Path

from .evidence import answer_variants

CSS = """
:root{--bg:#fff;--fg:#1d1d1f;--muted:#6b6b70;--card:#f5f5f7;--line:#d9d9de;--hit:#ffe08a;--err:#b42318;--ok:#067647;--warn:#b54708}
@media (prefers-color-scheme: dark){:root{--bg:#141416;--fg:#ececf0;--muted:#a0a0a8;--card:#1f1f23;--line:#33333a;--hit:#7a5b00;--err:#f97066;--ok:#47cd89;--warn:#fdb022}}
body{background:var(--bg);color:var(--fg);font:14px/1.5 system-ui,sans-serif;margin:0;padding:16px;max-width:1100px;margin:auto}
h1{font-size:20px} .meta{color:var(--muted)} .step{border:1px solid var(--line);border-radius:8px;margin:12px 0;padding:10px;background:var(--card)}
.phase{font-size:12px;color:var(--muted)} pre{white-space:pre-wrap;word-break:break-word;max-height:360px;overflow:auto;background:var(--bg);border:1px solid var(--line);border-radius:6px;padding:8px}
mark{background:var(--hit);color:inherit} .err{color:var(--err)} .ok{color:var(--ok)} .warn{color:var(--warn)} details{margin:6px 0}
"""


def _hl(text: str, pattern: re.Pattern | None) -> str:
    esc = html.escape(text or "")
    return pattern.sub(lambda m: f"<mark>{m.group(0)}</mark>", esc) if pattern else esc


def render(traj: dict, out: Path) -> Path:
    ans = str(traj.get("reference") or "")
    variants = sorted({v for s in traj.get("evidence_strings") or [] for v in answer_variants(str(s)) if len(v) >= 3})
    pat = re.compile("|".join(re.escape(html.escape(v)) for v in sorted(variants, key=len, reverse=True)), re.I) if variants else None
    closing = next((s for s in traj["steps"] if s.get("closing")), {})
    res = closing.get("passed")
    parts = [
        f"<!doctype html><html><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'>"
        f"<title>Trajectory viewer</title><style>{CSS}</style></head><body>",
        f"<h1>{html.escape(traj['run_id'])} / {html.escape(traj['task_id'])}</h1>",
        f"<p class='meta'>Result: <b class='{'ok' if res else 'err'}'>{'PASS' if res else 'FAIL'}</b> · termination "
        f"{html.escape(str(closing.get('termination_reason')))} · reference <mark>{html.escape(ans)}</mark></p>",
        f"<details open><summary>Task</summary><pre>{html.escape(traj.get('prompt', ''))}</pre></details>",
    ]
    for s in traj["steps"]:
        if s.get("closing"):
            parts.append(f"<div class='step'><b>Closing</b><pre>{html.escape(json.dumps(s, indent=1, default=str))}</pre></div>")
            continue
        exc = f" <span class='err'>{html.escape(str(s['exception_type']))}</span>" if s.get("exception_type") else ""
        parts.append(
            f"<div class='step'><div class='phase'>step {s['step']} · {s['harness_phase']} · in {s['prompt_tokens']}"
            f"+{s['cached_tokens']} cached · out {s['completion_tokens']} · {s['latency_seconds']}s{exc}</div>"
        )
        if s.get("reasoning"):
            parts.append(f"<details><summary>Reasoning</summary><pre>{_hl(s['reasoning'], pat)}</pre></details>")
        if s.get("text"):
            parts.append(f"<pre>{_hl(s['text'], pat)}</pre>")
        for tc in s.get("tool_calls", []):
            m = tc.get("delivery_meta") or {}
            badge = ""
            if m.get("truncated"):
                badge = f" <span class='warn'>[{html.escape(str(m.get('policy')))}: {m.get('delivered_tokens')}/{m.get('raw_tokens')} tok]</span>"
            if tc.get("is_error"):
                badge += " <span class='err'>[error]</span>"
            parts.append(f"<div><b>{html.escape(tc['tool'])}</b> <code>{html.escape(json.dumps(tc['tool_input'], ensure_ascii=False))}</code>{badge}</div>")
            parts.append(f"<details><summary>Delivered to model</summary><pre>{_hl(tc.get('tool_output_delivered', ''), pat)}</pre></details>")
            if tc.get("tool_output_raw") != tc.get("tool_output_delivered"):
                parts.append(f"<details><summary>Raw tool output (not all shown to model)</summary><pre>{_hl(tc.get('tool_output_raw', ''), pat)}</pre></details>")
        parts.append("</div>")
    parts.append("</body></html>")
    out.write_text("".join(parts), encoding="utf-8")
    return out
