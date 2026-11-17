"""Offline tests: safe runner + agent loop with a scripted fake LLM (no API key needed).

Run:  .venv\\Scripts\\python -m pytest -q
"""

import pandas as pd
import pytest

from agent.analyst import Analyst, extract_code
from agent.safe_runner import check_code, run_code


@pytest.fixture
def df() -> pd.DataFrame:
    return pd.DataFrame({"city": ["Karachi", "Lahore", "Karachi"], "revenue": [100, 80, 50]})


# ---------- safety checks ----------

@pytest.mark.parametrize("code", [
    "import os",
    "import subprocess",
    "from shutil import rmtree",
    "open('x.txt', 'w')",
    "eval('1+1')",
    "exec('x=1')",
    "__import__('os')",
    "x = df.__class__",
    "import requests",
    "df.to_csv('out.csv')",
])
def test_blocked_code_is_rejected(code: str) -> None:
    assert check_code(code), f"should be blocked: {code}"


def test_allowed_code_passes() -> None:
    assert check_code("import math\nimport plotly.express as px\nresult = math.sqrt(4)") == []


def test_run_returns_table_and_chart(df: pd.DataFrame) -> None:
    code = (
        "result = df.groupby('city')['revenue'].sum().sort_values(ascending=False)\n"
        "fig = px.bar(result.reset_index(), x='city', y='revenue', title='Revenue')"
    )
    out = run_code(code, df)
    assert out.ok, out.error
    assert out.result_kind == "table"
    assert out.result["data"][0] == ["Karachi", 150]
    assert out.fig_json


def test_plain_index_is_dropped_but_named_index_kept(df: pd.DataFrame) -> None:
    sorted_out = run_code("result = df.sort_values('revenue')", df)
    assert sorted_out.result["columns"] == ["city", "revenue"]
    grouped = run_code("result = df.groupby('city')['revenue'].mean()", df)
    assert grouped.result["columns"] == ["city", "revenue"]


def test_runtime_error_is_reported(df: pd.DataFrame) -> None:
    out = run_code("result = df['sales'].sum()", df)
    assert not out.ok
    assert "KeyError" in out.error


def test_timeout(df: pd.DataFrame) -> None:
    out = run_code("while True:\n    pass", df, timeout=2)
    assert not out.ok and "Timeout" in out.error


def test_unsafe_code_never_runs(df: pd.DataFrame) -> None:
    out = run_code("import os\nresult = os.listdir('.')", df)
    assert not out.ok and out.error.startswith("Safety check failed")


# ---------- agent loop with a fake LLM ----------

class FakeLLM:
    """Returns scripted replies in order and remembers the prompts it got."""

    def __init__(self, replies: list[str]):
        self.replies = list(replies)
        self.prompts: list[str] = []

    def __call__(self, prompt: str, system: str | None = None) -> str:
        self.prompts.append(prompt)
        return self.replies.pop(0)


BROKEN = """PLAN:
1. Sum sales by city
CODE:
```python
result = df.groupby('city')['sales'].sum()
fig = None
```"""

FIXED = """DIAGNOSIS: The column is called 'revenue', not 'sales'.
CODE:
```python
result = df.groupby('city')['revenue'].sum()
fig = px.bar(result.reset_index(), x='city', y='revenue')
```"""


def test_agent_fixes_its_own_error(df: pd.DataFrame) -> None:
    fake = FakeLLM([BROKEN, FIXED, "Karachi leads with 150 in revenue."])
    run = Analyst(df, llm_func=fake).answer("Revenue by city?")

    kinds = [s.kind for s in run.steps]
    assert kinds == ["look", "plan", "code", "run_error", "fix", "run_ok", "explain"]
    assert run.success and run.attempts == 2
    assert "KeyError" in fake.prompts[1]          # the error was given back to the LLM
    assert run.insight.startswith("Karachi")


def test_agent_gives_up_after_three_tries(df: pd.DataFrame) -> None:
    bad_fix = FIXED.replace("'revenue'", "'nope'")
    fake = FakeLLM([BROKEN, bad_fix, bad_fix])
    run = Analyst(df, llm_func=fake).answer("Revenue by city?")
    assert not run.success
    assert run.attempts == 3
    assert run.steps[-1].kind == "give_up"
    assert run.error_message


def test_extract_code_without_fence() -> None:
    assert extract_code("CODE:\nresult = 1") == "result = 1"
