"""AskYourData — Streamlit app.

Run locally:  .venv\\Scripts\\streamlit run app.py
"""

from __future__ import annotations

import io
import re
import uuid

import pandas as pd
import streamlit as st

from agent import llm
from agent.analyst import Analyst, AgentRun, Step
from agent.profile import load_table
from agent.report import Report, build_report, report_to_html
from agent.samples import SAMPLES
from rate_limit import DailyRateLimiter
from ui_render import render_answer, step_headline

st.set_page_config(page_title="AskYourData — AI data analyst agent", page_icon=":material/query_stats:",
                   layout="wide")

REPORT_COST = 3  # an auto report uses ~4x the AI calls of a single question
GITHUB_URL = "https://github.com/Memoona-Amjad/askyourdata"


# ---------- rate limit ----------

@st.cache_resource
def get_limiter() -> DailyRateLimiter:
    """One limiter shared by every visitor of this running app."""
    return DailyRateLimiter(int(llm.setting("RUNS_PER_IP_PER_DAY", "10") or 10))


def visitor_id() -> str:
    """The visitor's IP address (first X-Forwarded-For hop on hosted apps), else a per-session id."""
    forwarded = st.context.headers.get("X-Forwarded-For")
    forwarded = forwarded.split(",")[0].strip() if isinstance(forwarded, str) else ""
    ip = forwarded or st.context.ip_address
    if isinstance(ip, str) and ip:
        return ip
    return "session:" + st.session_state.setdefault("session_uid", uuid.uuid4().hex)


def use_runs(cost: int) -> bool:
    """Spend `cost` runs from today's budget, or show a friendly message and return False."""
    if get_limiter().try_consume(visitor_id(), cost):
        return True
    st.warning(
        f"You've used today's {get_limiter().limit} free demo runs — thanks for trying AskYourData! "
        f"The limit protects the free AI quota. Come back tomorrow, or run it yourself with your own "
        f"free Gemini key: [{GITHUB_URL}]({GITHUB_URL}).",
        icon=":material/hourglass_top:",
    )
    return False


# ---------- data loading ----------

@st.cache_data(show_spinner=False)
def load_sample(key: str) -> pd.DataFrame:
    return SAMPLES[key].load()


@st.cache_data(show_spinner=False)
def load_upload(data: bytes, name: str) -> pd.DataFrame:
    return load_table(io.BytesIO(data), name)


def sidebar() -> tuple[str, str, pd.DataFrame | None, tuple[str, ...]]:
    """Dataset picker + preview. Returns (dataset id, display name, DataFrame, example questions)."""
    with st.sidebar:
        st.header("1. Pick your data")
        source = st.radio("Data source", ["Try with sample data", "Upload my own file"], label_visibility="collapsed")

        if source == "Try with sample data":
            key = st.selectbox("Sample dataset", list(SAMPLES), format_func=lambda k: SAMPLES[k].label)
            sample = SAMPLES[key]
            st.caption(sample.description)
            df, dataset_id, name, questions = load_sample(key), f"sample:{key}", sample.label, sample.questions
        else:
            upload = st.file_uploader("CSV or Excel file (max 20 MB)", type=["csv", "xlsx", "xls"])
            if upload is None:
                st.info("Upload a CSV or XLSX file, or switch back to the sample data.")
                return "none", "", None, ()
            try:
                df = load_upload(upload.getvalue(), upload.name)
            except Exception:
                st.error("Sorry, that file could not be read. Please upload a valid CSV or Excel file.")
                return "none", "", None, ()
            if df.empty:
                st.error("That file has no rows.")
                return "none", "", None, ()
            dataset_id, name = f"upload:{upload.name}:{upload.size}", upload.name
            questions = (
                "Give me a quick summary of the most important numbers in this data.",
                "What are the top 5 rows by the most important numeric column?",
            )

        st.subheader("Preview")
        st.caption(f"{len(df):,} rows × {df.shape[1]} columns")
        st.dataframe(df.head(50), hide_index=True, height=260)

        st.divider()
    return dataset_id, name, df, questions


def show_runs_left(slot) -> None:
    """Filled in at the end of the page, so it counts the run that just happened."""
    slot.caption(f":material/bolt: {get_limiter().remaining(visitor_id())} free demo runs left today "
                 f"(a question uses 1, an auto report {REPORT_COST}).")


# ---------- tab 1: ask a question ----------

def ask(question: str, df: pd.DataFrame) -> AgentRun:
    """Run the agent and show its steps live while it works."""
    with st.status("The agent is working…", expanded=True) as status:
        def on_step(step: Step) -> None:
            status.markdown(step_headline(step))
            if step.kind == "run_error":
                status.caption(step.content[:300])

        run = Analyst(df).answer(question, on_step=on_step)
        if run.success:
            status.update(label=f"Done in {run.attempts} tr{'y' if run.attempts == 1 else 'ies'}",
                          state="complete", expanded=False)
        else:
            status.update(label="The agent could not answer this one", state="error", expanded=False)
    return run


def chat_tab(df: pd.DataFrame, questions: tuple[str, ...]) -> None:
    st.markdown("**Try an example question:**")
    cols = st.columns(2)
    for i, q in enumerate(questions):
        if cols[i % 2].button(q, key=f"example_{i}", width="stretch"):
            st.session_state.pending = q

    # Previous questions and answers
    for i, (q, run) in enumerate(st.session_state.history):
        with st.chat_message("user"):
            st.markdown(q)
        with st.chat_message("assistant"):
            render_answer(run, key=str(i))

    typed = st.chat_input("Ask a question about this data…", max_chars=500)
    question = typed or st.session_state.pending
    st.session_state.pending = None
    if not question:
        return

    with st.chat_message("user"):
        st.markdown(question)
    with st.chat_message("assistant"):
        if not use_runs(1):
            return
        try:
            run = ask(question, df)
        except Exception:
            # Never show a raw stack trace to the user
            run = AgentRun(question=question, service_error=True,
                           error_message="Something unexpected went wrong. Please try again.")
        if run.service_error:
            get_limiter().refund(visitor_id(), 1)  # not the visitor's fault: give the run back
        render_answer(run, key=str(len(st.session_state.history)))
    st.session_state.history.append((question, run))


# ---------- tab 2: auto report ----------

def report_tab(df: pd.DataFrame, dataset_name: str) -> None:
    st.markdown("The agent explores the data **on its own**: it proposes questions, writes and runs code for "
                "each one (fixing its own errors), keeps the **5 most useful insights**, and builds a report "
                "you can download as an HTML file. Takes 1–2 minutes.")

    if st.button("Generate auto report", type="primary", icon=":material/auto_awesome:"):
        if use_runs(REPORT_COST):
            with st.status("The agent is exploring the data…", expanded=True) as status:
                try:
                    st.session_state.report = build_report(Analyst(df), dataset_name, on_progress=status.markdown)
                    status.update(label="Report ready", state="complete", expanded=False)
                except llm.LLMError as exc:
                    get_limiter().refund(visitor_id(), REPORT_COST)
                    status.update(label="The report could not be finished", state="error", expanded=False)
                    st.error(str(exc), icon=":material/warning:")
                except Exception:
                    get_limiter().refund(visitor_id(), REPORT_COST)
                    status.update(label="The report could not be finished", state="error", expanded=False)
                    st.error("Something unexpected went wrong while building the report. Please try again.")

    report: Report | None = st.session_state.get("report")
    if report is None:
        return
    if not report.findings:
        st.warning("The agent could not produce any working analysis for this data. Try asking a question instead.")
        return

    file_name = "askyourdata_report_" + re.sub(r"[^a-z0-9]+", "_", dataset_name.lower()).strip("_") + ".html"
    st.download_button("Download report (HTML)", data=report_to_html(report), file_name=file_name,
                       mime="text/html", icon=":material/download:", type="primary")
    if report.summary:
        st.info(f"**Executive summary.** {report.summary}", icon=":material/summarize:")
    for i, f in enumerate(report.findings, start=1):
        st.subheader(f"{i}. {f.question}")
        f.run.insight = f.insight
        render_answer(f.run, key=f"report_{i}")
    if report.skipped:
        st.caption("Skipped after 3 failed tries: " + "; ".join(report.skipped))


# ---------- page ----------

def main() -> None:
    dataset_id, dataset_name, df, questions = sidebar()
    runs_left_slot = st.sidebar.empty()

    st.title("AskYourData")
    st.markdown("Ask a question about your spreadsheet in plain English. The AI agent **plans**, "
                "**writes Python**, **runs it**, **fixes its own errors**, and answers with a chart and an insight.")

    # Start fresh when the dataset changes
    if st.session_state.get("dataset_id") != dataset_id:
        st.session_state.dataset_id = dataset_id
        st.session_state.history = []  # list of (question, AgentRun)
        st.session_state.pending = None
        st.session_state.report = None

    if df is None:
        st.info("Pick a sample dataset or upload a file in the sidebar to begin.")
        show_runs_left(runs_left_slot)
        return

    ask_tab, auto_tab = st.tabs([":material/chat: Ask a question", ":material/auto_awesome: Auto report"])
    with ask_tab:
        chat_tab(df, questions)
    with auto_tab:
        report_tab(df, dataset_name)
    show_runs_left(runs_left_slot)


main()
