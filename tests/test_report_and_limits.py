"""Offline tests for the auto report and the rate limiter (fake LLM, no API key needed)."""

import pandas as pd

from agent.analyst import Analyst
from agent.report import build_report, parse_insights, parse_questions, report_to_html
from rate_limit import DailyRateLimiter


def test_rate_limiter() -> None:
    limiter = DailyRateLimiter(limit_per_day=3)
    assert limiter.try_consume("1.2.3.4", 2)
    assert not limiter.try_consume("1.2.3.4", 2)      # would go over the limit
    assert limiter.remaining("1.2.3.4") == 1
    assert limiter.try_consume("5.6.7.8", 3)          # other visitors have their own budget
    limiter.refund("1.2.3.4", 2)
    assert limiter.remaining("1.2.3.4") == 3


def test_parsers() -> None:
    assert parse_questions("1. Which city sells most?\n- What is the monthly trend?\n\n", 6) == [
        "Which city sells most?", "What is the monthly trend?"]
    # The model sometimes answers with a Python list anyway
    assert parse_questions('```python\nquestions = [\n  "Which city sells most?",\n]\n```', 6) == [
        "Which city sells most?"]
    insights, summary = parse_insights("1. First one.\n2. Second\nline.\nSUMMARY: All good.", 2)
    assert insights == ["First one.", "Second\nline."] and summary == "All good."


class ScriptedLLM:
    """Answers by prompt type, so the order of calls does not matter."""

    def __init__(self) -> None:
        self.fix_calls = 0

    def __call__(self, prompt: str, system: str | None = None) -> str:
        if "Propose" in prompt:
            return "\n".join(f"Question number {i} about revenue?" for i in range(1, 7))
        if "failed" in prompt:
            self.fix_calls += 1
            return "DIAGNOSIS: wrong column.\nCODE:\n```python\nresult = df.groupby('city')['revenue'].sum()\nfig = None\n```"
        if "Question number 2" in prompt and "PLAN" in prompt:
            # Question 2 starts with a bug so the report shows a self-fix
            return "PLAN:\n1. Sum\nCODE:\n```python\nresult = df['nope'].sum()\nfig = None\n```"
        if "PLAN" in prompt:
            return ("PLAN:\n1. Sum by city\nCODE:\n```python\nresult = df.groupby('city')['revenue'].sum()\n"
                    "fig = px.bar(result.reset_index(), x='city', y='revenue')\n```")
        return "\n".join(f"{i}. Insight {i}." for i in range(1, 6)) + "\nSUMMARY: Karachi leads."


def test_auto_report_end_to_end() -> None:
    df = pd.DataFrame({"city": ["Karachi", "Lahore"], "revenue": [100, 80]})
    fake = ScriptedLLM()
    report = build_report(Analyst(df, llm_func=fake), "Test data")

    assert len(report.findings) == 5                 # stops once it has 5 insights
    assert fake.fix_calls == 1                       # question 2 was fixed by the agent
    assert report.findings[1].run.attempts == 2
    assert report.findings[0].insight == "Insight 1." and report.summary == "Karachi leads."

    page = report_to_html(report)
    assert page.startswith("<!doctype html>")
    assert page.count("<section>") == 5
    assert "cdn.plot.ly" in page and "Karachi leads." in page
