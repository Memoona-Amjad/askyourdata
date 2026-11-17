"""All prompt text in one place, so it is easy to read and tweak."""

SYSTEM = """You are AskYourData, a careful senior data analyst.
You answer business questions about a pandas DataFrame called `df` by writing Python code.
You only use pandas (pd), numpy (np) and plotly (px = plotly.express, go = plotly.graph_objects).
They are already imported and `df` is already loaded. You can also import math, datetime or statistics."""

CODE_RULES = """Code rules:
- Use only the real column names shown in the data profile.
- Store the final answer in a variable called `result`. Prefer a small DataFrame that holds the
  numbers behind the answer (e.g. every category with its average), not just a single label.
  Name columns clearly and keep units from the source column (e.g. avg_first_response_hours).
  Keep tables small (at most ~25 rows): aggregate, sort and pick the top rows.
- Store ONE plotly chart in a variable called `fig` (use px or go) with a clear title and axis labels.
  If a chart makes no sense, set fig = None.
- Do NOT read or write files, do NOT call fig.show(), do NOT use print for the answer.
- Do NOT import os, sys, subprocess or any library other than pandas, numpy, plotly, math, datetime, statistics.
- Columns with dates stored as text must be converted with pd.to_datetime before date logic.
- Handle missing values sensibly (e.g. dropna on the column you average)."""

PLAN_AND_CODE = """Question: {question}

Data profile:
{profile}

{rules}

Reply in exactly this format:
PLAN:
1. <short step>
2. <short step>
(2 to 5 steps)

CODE:
```python
<your code>
```"""

FIX = """Your code for the question below failed. Read the error, find the cause and fix it.

Question: {question}

Data profile:
{profile}

Code that failed:
```python
{code}
```

Error:
{error}

{rules}

Reply in exactly this format:
DIAGNOSIS: <one sentence: what went wrong>

CODE:
```python
<the full corrected code>
```"""

EXPLAIN = """Question: {question}

The analysis code produced this result:
{result}

Write a 2-4 sentence insight for a busy business executive.
Lead with the direct answer, use the real numbers, mention one notable pattern or
recommended action. Only use numbers that appear in the result, with the units implied by the
column names (do not convert units or invent numbers). Round large numbers for readability (e.g. 62,397,977 -> 62.4M,
0.2783 -> 27.8%). Plain business language, no code, no jargon, no markdown headings."""

AUTO_REPORT_QUESTIONS = """You are exploring a dataset to write a short executive report.

Data profile:
{profile}

Propose {n} DIFFERENT business questions to answer from this data, ranked from most to least
useful for a business leader. Favour questions that reveal something a leader would act on:
change over time, growth or decline of a segment, seasonal dips or spikes, unusual segments,
and drivers of a key outcome. Avoid more than one plain "total by X" question.
Each must be answerable with pandas and shown as one chart. Keep each question short and specific.

Reply with exactly {n} lines of plain text, one question per line, no numbering, no quotes, no code."""

REPORT_SUMMARY = """You analysed a dataset and found these results:

{findings}

For EACH numbered finding, write a 2-3 sentence insight for a busy executive. Only use numbers that
appear in that finding's result, with the units implied by the column names, rounded for readability
(e.g. 62.4M, 27.8%). Then write a 2-sentence overall summary with the single most important takeaway.

Reply in exactly this format:
{format_lines}
SUMMARY: <overall summary>"""
