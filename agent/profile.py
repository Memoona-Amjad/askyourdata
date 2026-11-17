"""Step 1 of the agent: LOOK at the data. Pure pandas, no LLM needed.

Builds a short text description of the table (columns, types, sample rows,
basic stats) that is given to the LLM so it writes code for the real columns.
"""

from __future__ import annotations

import pandas as pd


def load_table(file, filename: str) -> pd.DataFrame:
    """Read an uploaded CSV or Excel file into a DataFrame."""
    name = filename.lower()
    if name.endswith((".xlsx", ".xls")):
        return pd.read_excel(file)
    try:
        return pd.read_csv(file)
    except UnicodeDecodeError:
        file.seek(0)
        return pd.read_csv(file, encoding="latin-1")


def profile_dataframe(df: pd.DataFrame, max_cols: int = 40) -> str:
    """Return a compact, LLM-friendly description of `df`."""
    lines = [f"Rows: {len(df):,}   Columns: {df.shape[1]}", "", "Columns (name | type | missing | details):"]
    for col in list(df.columns)[:max_cols]:
        s = df[col]
        missing = int(s.isna().sum())
        if pd.api.types.is_numeric_dtype(s):
            details = f"min={s.min():,.4g}, mean={s.mean():,.4g}, max={s.max():,.4g}" if s.notna().any() else "all empty"
        else:
            n_unique = s.nunique(dropna=True)
            top = s.dropna().astype(str).value_counts().head(8).index.tolist()
            details = f"{n_unique} unique; e.g. {top}"
            if _looks_like_dates(s):
                parsed = pd.to_datetime(s, errors="coerce", format="mixed")
                details += f"; looks like dates from {parsed.min():%Y-%m-%d} to {parsed.max():%Y-%m-%d} (stored as text)"
        lines.append(f"- {col} | {s.dtype} | {missing} | {details}")
    if df.shape[1] > max_cols:
        lines.append(f"... and {df.shape[1] - max_cols} more columns")

    lines += ["", "First 5 rows:", df.head(5).to_string(max_colwidth=30)]
    return "\n".join(lines)


def _looks_like_dates(s: pd.Series) -> bool:
    """True if most sampled text values parse as dates."""
    if not (pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s)):
        return False
    sample = s.dropna().astype(str).head(50)
    if sample.empty or sample.str.len().min() < 6:
        return False
    parsed = pd.to_datetime(sample, errors="coerce", format="mixed")
    return parsed.notna().mean() > 0.8
