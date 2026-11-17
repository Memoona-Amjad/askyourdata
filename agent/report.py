"""Auto report: the agent explores the data on its own.

    1. LOOK      profile the data
    2. PROPOSE   LLM suggests 6 candidate questions, ranked by usefulness
    3. ANALYSE   each candidate goes through the normal plan -> code -> run -> fix loop,
                 until 5 have worked (a candidate that fails 3 times is skipped)
    4. EXPLAIN   one LLM call writes an insight per finding + an executive summary
    5. EXPORT    everything is rendered into one self-contained HTML file
"""

from __future__ import annotations

import html
import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import Callable

import plotly.io as pio

from agent import prompts
from agent.analyst import AgentRun, Analyst, Step
from agent.llm import LLMError

N_INSIGHTS = 5
N_CANDIDATES = 6

ProgressCallback = Callable[[str], None]


@dataclass
class Finding:
    question: str
    run: AgentRun
    insight: str = ""


@dataclass
class Report:
    dataset_name: str
    created_at: datetime = field(default_factory=datetime.now)
    candidates: list[str] = field(default_factory=list)
    findings: list[Finding] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)
    summary: str = ""


def parse_questions(reply: str, n: int) -> list[str]:
    """One question per line. Strips numbering, bullets, quotes and list syntax the model may add anyway."""
    questions = []
    for line in reply.splitlines():
        line = re.sub(r"^\s*(?:\d+[.)]|[-*•])\s*", "", line).strip().rstrip(",").strip().strip("\"'`").strip()
        if len(line) > 10 and "=" not in line and not line.startswith(("```", "[", "]")):
            questions.append(line)
    return questions[:n]


def parse_insights(reply: str, n: int) -> tuple[list[str], str]:
    """Pull out the numbered insights and the SUMMARY line."""
    insights = []
    for i in range(1, n + 1):
        match = re.search(rf"^\s*{i}[.)]\s*(.+?)(?=^\s*{i + 1}[.)]|^\s*SUMMARY:|\Z)", reply, flags=re.S | re.M)
        insights.append(match.group(1).strip() if match else "")
    summary = re.search(r"SUMMARY:\s*(.+)", reply, flags=re.S)
    return insights, (summary.group(1).strip() if summary else "")


def build_report(analyst: Analyst, dataset_name: str, on_progress: ProgressCallback | None = None) -> Report:
    """Run the whole auto-report agent. Raises llm.LLMError if the AI service is unavailable."""
    say = on_progress or (lambda msg: None)
    report = Report(dataset_name=dataset_name)

    say(":material/search: Looked at the data")
    reply = analyst.llm(prompts.AUTO_REPORT_QUESTIONS.format(profile=analyst.profile, n=N_CANDIDATES),
                        prompts.SYSTEM)
    report.candidates = parse_questions(reply, N_CANDIDATES)
    say(f":material/checklist: Picked {len(report.candidates)} candidate questions to explore")

    for question in report.candidates:
        if len(report.findings) == N_INSIGHTS:
            break
        say(f":material/code: Analysing: *{question}*")

        def on_step(step: Step) -> None:
            if step.kind == "run_error":
                say(f":red[:material/error: Code failed (try {step.attempt}) — the agent is fixing it]")

        run = analyst.answer(question, on_step=on_step, explain=False)
        if run.success:
            report.findings.append(Finding(question, run))
        elif run.service_error:
            raise LLMError(run.error_message)
        else:
            report.skipped.append(question)
            say(f":orange[:material/skip_next: Skipped after 3 tries: *{question}*]")

    if report.findings:
        say(":material/lightbulb: Writing the insights and executive summary")
        blocks = [f"{i}. Question: {f.question}\nResult:\n{f.run.result_as_text(max_rows=20)[:1500]}"
                  for i, f in enumerate(report.findings, start=1)]
        format_lines = "\n".join(f"{i}. <insight>" for i in range(1, len(report.findings) + 1))
        reply = analyst.llm(prompts.REPORT_SUMMARY.format(findings="\n\n".join(blocks), format_lines=format_lines),
                            prompts.SYSTEM)
        insights, report.summary = parse_insights(reply, len(report.findings))
        for finding, insight in zip(report.findings, insights):
            finding.insight = insight
    return report


# ---------- HTML export ----------

CSS = """
:root { --ink:#0b0b0b; --muted:#52514e; --line:#e4e3df; --accent:#2a78d6; --bg:#fcfcfb; }
* { box-sizing: border-box; }
body { margin:0; background:var(--bg); color:var(--ink);
       font-family: Inter, "Segoe UI", Arial, sans-serif; line-height:1.55; }
main { max-width: 980px; margin: 0 auto; padding: 40px 20px 64px; }
h1 { margin:0 0 4px; font-size: 28px; }
h2 { font-size: 19px; margin: 0 0 8px; }
.meta { color:var(--muted); margin:0 0 28px; font-size:14px; }
.summary { background:#fff; border:1px solid var(--line); border-left:4px solid var(--accent);
           border-radius:8px; padding:16px 20px; margin-bottom:32px; }
section { background:#fff; border:1px solid var(--line); border-radius:10px; padding:22px 22px 12px;
          margin-bottom:24px; }
.num { color:var(--accent); margin-right:6px; }
table { border-collapse: collapse; width:100%; font-size:13px; margin:8px 0 12px; }
th, td { text-align:left; padding:6px 10px; border-bottom:1px solid var(--line); }
th { color:var(--muted); font-weight:600; }
details { margin: 8px 0 12px; color:var(--muted); font-size:14px; }
pre { background:#f5f5f3; padding:12px; border-radius:6px; overflow-x:auto; font-size:12.5px; color:var(--ink); }
.err { color:#b3261e; }
footer { color:var(--muted); font-size:13px; margin-top:32px; }
"""


def report_to_html(report: Report) -> str:
    """Render the report as one HTML file (charts load plotly.js from a CDN)."""
    esc = html.escape
    parts = [
        "<!doctype html><html lang='en'><head><meta charset='utf-8'>",
        "<meta name='viewport' content='width=device-width, initial-scale=1'>",
        f"<title>AskYourData report — {esc(report.dataset_name)}</title><style>{CSS}</style></head><body><main>",
        "<h1>AskYourData — auto report</h1>",
        f"<p class='meta'>Dataset: <b>{esc(report.dataset_name)}</b> · "
        f"Generated {report.created_at:%d %b %Y, %H:%M} · {len(report.findings)} insights</p>",
    ]
    if report.summary:
        parts.append(f"<div class='summary'><b>Executive summary.</b> {esc(report.summary)}</div>")

    first_chart = True
    for i, f in enumerate(report.findings, start=1):
        parts.append(f"<section><h2><span class='num'>{i}.</span>{esc(f.question)}</h2>")
        if f.insight:
            parts.append(f"<p>{esc(f.insight)}</p>")
        if f.run.fig_json:
            fig = pio.from_json(f.run.fig_json)
            parts.append(fig.to_html(full_html=False, include_plotlyjs="cdn" if first_chart else False,
                                     config={"displaylogo": False}))
            first_chart = False
        frame = f.run.result_frame()
        if frame is not None:
            parts.append("<details><summary>Data table</summary>"
                         + frame.head(25).to_html(index=False, border=0, float_format=lambda x: f"{x:,.2f}")
                         + "</details>")
        parts.append(_agent_trace_html(f.run))
        parts.append("</section>")

    if report.skipped:
        items = "".join(f"<li>{esc(q)}</li>" for q in report.skipped)
        parts.append(f"<p class='meta'>Questions the agent tried but skipped after 3 failed attempts:</p><ul>{items}</ul>")
    parts.append("<footer>Made by the AskYourData agent: it planned, wrote and ran its own Python code, "
                 "and fixed its own errors. Numbers come from the code, the wording from an LLM — "
                 "check key figures before making decisions.</footer></main></body></html>")
    return "\n".join(parts)


def _agent_trace_html(run: AgentRun) -> str:
    esc = html.escape
    tries = f"{run.attempts} tr{'y' if run.attempts == 1 else 'ies'}"
    rows = []
    for step in run.steps:
        if step.kind == "look":
            continue
        css = " class='err'" if step.kind == "run_error" else ""
        rows.append(f"<p{css}><b>{esc(step.title)}</b><br>{esc(step.content)}</p>" if step.content
                    else f"<p><b>{esc(step.title)}</b></p>")
        if step.code:
            rows.append(f"<pre>{esc(step.code)}</pre>")
    return f"<details><summary>How the agent did it ({tries})</summary>{''.join(rows)}</details>"
