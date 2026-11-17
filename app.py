"""AskYourData — Streamlit app.

Run locally:  .venv\\Scripts\\streamlit run app.py
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from agent.analyst import Analyst, AgentRun, Step
from agent.profile import load_table
from agent.samples import SAMPLES
from ui_render import render_answer, step_headline

st.set_page_config(page_title="AskYourData — AI data analyst agent", page_icon=":material/query_stats:",
                   layout="wide")


# ---------- data loading ----------

@st.cache_data(show_spinner=False)
def load_sample(key: str) -> pd.DataFrame:
    return SAMPLES[key].load()


@st.cache_data(show_spinner=False)
def load_upload(data: bytes, name: str) -> pd.DataFrame:
    import io

    return load_table(io.BytesIO(data), name)


def sidebar() -> tuple[str, pd.DataFrame | None, tuple[str, ...]]:
    """Dataset picker + preview. Returns (dataset id, DataFrame, example questions)."""
    with st.sidebar:
        st.header("1. Pick your data")
        source = st.radio("Data source", ["Try with sample data", "Upload my own file"], label_visibility="collapsed")

        if source == "Try with sample data":
            key = st.selectbox("Sample dataset", list(SAMPLES), format_func=lambda k: SAMPLES[k].label)
            sample = SAMPLES[key]
            st.caption(sample.description)
            df, dataset_id, questions = load_sample(key), f"sample:{key}", sample.questions
        else:
            upload = st.file_uploader("CSV or Excel file", type=["csv", "xlsx", "xls"])
            if upload is None:
                st.info("Upload a CSV or XLSX file, or switch back to the sample data.")
                return "none", None, ()
            try:
                df = load_upload(upload.getvalue(), upload.name)
            except Exception:
                st.error("Sorry, that file could not be read. Please upload a valid CSV or Excel file.")
                return "none", None, ()
            if df.empty:
                st.error("That file has no rows.")
                return "none", None, ()
            dataset_id = f"upload:{upload.name}:{upload.size}"
            questions = (
                "Give me a quick summary of the most important numbers in this data.",
                "What are the top 5 rows by the most important numeric column?",
            )

        st.subheader("Preview")
        st.caption(f"{len(df):,} rows × {df.shape[1]} columns")
        st.dataframe(df.head(50), hide_index=True, height=260)
    return dataset_id, df, questions


# ---------- chat ----------

def ask(question: str, df: pd.DataFrame) -> AgentRun:
    """Run the agent and show its steps live while it works."""
    with st.status("The agent is working…", expanded=True) as status:
        def on_step(step: Step) -> None:
            status.markdown(step_headline(step))
            if step.kind == "run_error":
                status.caption(step.content[:300])

        run = Analyst(df).answer(question, on_step=on_step)
        if run.success:
            label = f"Done in {run.attempts} tr{'y' if run.attempts == 1 else 'ies'}"
            status.update(label=label, state="complete", expanded=False)
        else:
            status.update(label="The agent could not answer this one", state="error", expanded=False)
    return run


def main() -> None:
    dataset_id, df, questions = sidebar()

    st.title("AskYourData")
    st.markdown("Ask a question about your spreadsheet in plain English. The AI agent **plans**, "
                "**writes Python**, **runs it**, **fixes its own errors**, and answers with a chart and an insight.")

    # Start a fresh conversation when the dataset changes
    if st.session_state.get("dataset_id") != dataset_id:
        st.session_state.dataset_id = dataset_id
        st.session_state.history = []  # list of (question, AgentRun)
        st.session_state.pending = None

    if df is None:
        st.info("Pick a sample dataset or upload a file in the sidebar to begin.")
        return

    # Clickable example questions
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

    if question:
        with st.chat_message("user"):
            st.markdown(question)
        with st.chat_message("assistant"):
            try:
                run = ask(question, df)
            except Exception:
                # Never show a raw stack trace to the user
                run = AgentRun(question=question, error_message="Something unexpected went wrong. Please try again.")
            render_answer(run, key=str(len(st.session_state.history)))
        st.session_state.history.append((question, run))


main()
