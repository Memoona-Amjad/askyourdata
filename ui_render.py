"""Streamlit helpers that draw the agent's answer and its step-by-step work."""

from __future__ import annotations

import plotly.io as pio
import streamlit as st

from agent.analyst import AgentRun, Step

# Icon and colour for each kind of step, so the agent's work is easy to scan.
STEP_STYLE: dict[str, tuple[str, str]] = {
    "look": (":material/search:", "blue"),
    "plan": (":material/checklist:", "violet"),
    "code": (":material/code:", "gray"),
    "run_ok": (":material/check_circle:", "green"),
    "run_error": (":material/error:", "red"),
    "fix": (":material/build:", "orange"),
    "explain": (":material/lightbulb:", "blue"),
    "give_up": (":material/block:", "red"),
}


def step_headline(step: Step) -> str:
    icon, color = STEP_STYLE.get(step.kind, (":material/circle:", "gray"))
    return f"{icon} :{color}[**{step.title}**]"


def render_step(step: Step, number: int) -> None:
    """Draw one step in full (used in the 'How the agent did it' panel)."""
    st.markdown(f"**{number}.** {step_headline(step)}")
    if step.kind == "look":
        st.code(step.content, language="text")
    elif step.kind == "plan":
        st.markdown(step.content)
    elif step.kind == "run_error":
        st.error(step.content, icon=":material/bug_report:")
    elif step.kind == "fix":
        st.warning(f"**Diagnosis:** {step.content}", icon=":material/psychology:")
    elif step.kind == "give_up":
        st.error(step.content)
    elif step.kind == "run_ok" and step.content.strip():
        st.caption("Printed output:")
        st.code(step.content, language="text")
    elif step.kind == "explain":
        st.markdown(step.content)
    if step.code:
        st.code(step.code, language="python")


def render_answer(run: AgentRun, key: str) -> None:
    """Draw the final answer: insight, chart, result table and the agent-steps panel."""
    if run.success:
        tries = f"{run.attempts} tr{'y' if run.attempts == 1 else 'ies'}"
        fixed = " — the agent fixed its own error" if run.attempts > 1 else ""
        st.caption(f":material/smart_toy: Answered in {tries}{fixed}")
        if run.insight:
            st.markdown(run.insight)
        if run.fig_json:
            # theme=None keeps the agent's own chart style, so the app matches the HTML report
            st.plotly_chart(pio.from_json(run.fig_json), key=f"chart_{key}", theme=None)
        frame = run.result_frame()
        if frame is not None:
            st.dataframe(frame, hide_index=True, key=f"table_{key}")
        elif run.result is not None and not run.insight:
            st.markdown(f"**Result:** {run.result}")
    else:
        st.error(run.error_message or "Something went wrong. Please try again.", icon=":material/warning:")

    if run.steps:
        with st.expander("How the agent did it", icon=":material/account_tree:"):
            st.caption("Look → plan → write code → run → fix errors (up to 3 tries) → explain")
            for i, step in enumerate(run.steps, start=1):
                render_step(step, i)
