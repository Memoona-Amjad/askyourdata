"""The agent loop.

    1. LOOK     profile the data (no LLM)
    2. PLAN     LLM writes a short plan ...
    3. CODE     ... and pandas/plotly code that sets `result` and `fig`
    4. RUN      the safe runner executes the code
    5. FIX      on error, the LLM reads the error and rewrites the code (max 3 tries total)
    6. EXPLAIN  LLM turns the result into a 2-4 sentence business insight

Every step is recorded as a `Step`, so the UI can show exactly what the agent did.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Callable

import pandas as pd

from agent import llm, prompts
from agent.profile import profile_dataframe
from agent.safe_runner import RunResult, run_code

MAX_ATTEMPTS = 3

# Type of the LLM function: (prompt, system) -> reply text. Swappable for tests.
LLMFunc = Callable[[str, str | None], str]
StepCallback = Callable[["Step"], None]


@dataclass
class Step:
    """One visible step of the agent's work."""

    kind: str          # "look" | "plan" | "code" | "run_ok" | "run_error" | "fix" | "explain" | "give_up"
    title: str
    content: str = ""
    code: str = ""
    attempt: int = 0


@dataclass
class AgentRun:
    """Everything produced while answering one question."""

    question: str
    steps: list[Step] = field(default_factory=list)
    success: bool = False
    attempts: int = 0
    code: str = ""
    result: Any = None
    result_kind: str = "none"
    fig_json: str | None = None
    insight: str = ""
    error_message: str = ""   # friendly message if the run failed
    service_error: bool = False  # True if the AI service failed (not the agent's code)

    def result_frame(self) -> pd.DataFrame | None:
        """The result as a DataFrame if it was a table."""
        if self.result_kind == "table" and self.result:
            return pd.DataFrame(self.result["data"], columns=self.result["columns"])
        return None

    def result_as_text(self, max_rows: int = 25) -> str:
        frame = self.result_frame()
        if frame is not None:
            return frame.head(max_rows).to_string(index=False)
        return str(self.result) if self.result is not None else "(no result value)"


# ---------- parsing helpers ----------

def extract_code(reply: str) -> str:
    """Pull the Python code out of an LLM reply."""
    blocks = re.findall(r"```(?:python|py)?\s*\n(.*?)```", reply, flags=re.S)
    if blocks:
        return max(blocks, key=len).strip()
    # Fallback: everything after "CODE:"
    if "CODE:" in reply:
        return reply.split("CODE:", 1)[1].strip().strip("`")
    return reply.strip()


def extract_section(reply: str, name: str) -> str:
    """Text after 'NAME:' up to the next 'CODE:' marker or code fence."""
    match = re.search(rf"{name}:\s*(.*?)(?:\n\s*CODE:|```|$)", reply, flags=re.S | re.I)
    return match.group(1).strip() if match else ""


# ---------- the agent ----------

class Analyst:
    def __init__(self, df: pd.DataFrame, llm_func: LLMFunc | None = None, max_attempts: int = MAX_ATTEMPTS):
        self.df = df
        self.llm: LLMFunc = llm_func or (lambda prompt, system=None: llm.generate(prompt, system=system))
        self.max_attempts = max_attempts
        self.profile = profile_dataframe(df)

    def _ask(self, prompt: str) -> str:
        return self.llm(prompt, prompts.SYSTEM)

    def _record(self, run: AgentRun, step: Step, on_step: StepCallback | None) -> None:
        run.steps.append(step)
        if on_step:
            on_step(step)

    def answer(self, question: str, on_step: StepCallback | None = None, explain: bool = True) -> AgentRun:
        """Run the full agent loop for one question."""
        run = AgentRun(question=question)
        try:
            # 1. LOOK
            self._record(run, Step("look", "Looked at the data", self.profile), on_step)

            # 2 + 3. PLAN and CODE (one LLM call to save free-tier quota)
            reply = self._ask(prompts.PLAN_AND_CODE.format(
                question=question, profile=self.profile, rules=prompts.CODE_RULES))
            plan = extract_section(reply, "PLAN") or "(no plan given)"
            code = extract_code(reply)
            self._record(run, Step("plan", "Made a plan", plan), on_step)
            self._record(run, Step("code", "Wrote Python code (try 1)", code=code, attempt=1), on_step)

            # 4 + 5. RUN, and FIX on failure
            outcome = self._run_with_fixes(run, question, code, on_step)
            if outcome is None:
                return run

            # 6. EXPLAIN
            run.success = True
            if explain:
                run.insight = self._ask(prompts.EXPLAIN.format(question=question, result=self._result_summary(run)))
                self._record(run, Step("explain", "Explained the result", run.insight), on_step)
        except llm.LLMError as exc:
            run.success = False
            run.service_error = True
            run.error_message = str(exc)
        return run

    def _run_with_fixes(self, run: AgentRun, question: str, code: str,
                        on_step: StepCallback | None) -> RunResult | None:
        """Run code; on error ask the LLM to fix it. Returns the good RunResult or None."""
        for attempt in range(1, self.max_attempts + 1):
            run.attempts = attempt
            outcome = run_code(code, self.df)
            if outcome.ok:
                self._record(run, Step("run_ok", f"Ran the code successfully (try {attempt})",
                                       outcome.stdout, attempt=attempt), on_step)
                run.code = code
                run.result, run.result_kind, run.fig_json = outcome.result, outcome.result_kind, outcome.fig_json
                return outcome

            self._record(run, Step("run_error", f"Code failed (try {attempt})", outcome.error,
                                   attempt=attempt), on_step)
            if attempt == self.max_attempts:
                break

            # Give the error back to the LLM and ask for a fix
            reply = self._ask(prompts.FIX.format(question=question, profile=self.profile, code=code,
                                                 error=outcome.error, rules=prompts.CODE_RULES))
            diagnosis = extract_section(reply, "DIAGNOSIS") or "Rewrote the code."
            code = extract_code(reply)
            self._record(run, Step("fix", f"Read the error and fixed the code (try {attempt + 1})",
                                   diagnosis, code=code, attempt=attempt + 1), on_step)

        run.code = code
        run.error_message = (f"The agent could not get working code after {self.max_attempts} tries. "
                             "Try rephrasing the question or making it more specific.")
        self._record(run, Step("give_up", "Stopped after 3 tries", run.error_message), on_step)
        return None

    @staticmethod
    def _result_summary(run: AgentRun) -> str:
        text = run.result_as_text()
        if run.fig_json and run.result is None:
            text = "(Only a chart was produced.)"
        return text[:4000]
