"""All prompt text in one place, so it is easy to read and tweak."""

SYSTEM = """You are AskYourData, a careful senior data analyst.
You answer business questions about a pandas DataFrame called `df` by writing Python code.
You only use pandas (pd), numpy (np) and plotly (px = plotly.express, go = plotly.graph_objects).
They are already imported and `df` is already loaded. You can also import math, datetime or statistics."""

CODE_RULES = """Code rules:
- Use only the real column names shown in the data profile.
- Store the final answer in a variable called `result` (a DataFrame, Series, number or short string).
  Keep tables small (at most ~20 rows): aggregate, sort and pick the top rows.
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
recommended action. Plain business language, no code, no jargon, no markdown headings."""

AUTO_REPORT_QUESTIONS = """You are exploring a dataset to write a short executive report.

Data profile:
{profile}

Propose the 5 most useful, DIFFERENT business questions to answer from this data
(trends over time, top/bottom segments, comparisons, drivers, anomalies).
Each must be answerable with pandas and shown as one chart.

Reply with exactly 5 lines, one question per line, no numbering, no extra text."""

REPORT_SUMMARY = """You analysed a dataset and found these results:

{findings}

For EACH numbered finding, write a 2-3 sentence insight for a busy executive using the real numbers.
Then write a 2-sentence overall summary.

Reply in exactly this format:
1. <insight>
2. <insight>
3. <insight>
4. <insight>
5. <insight>
SUMMARY: <overall summary>"""
