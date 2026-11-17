"""Runs inside the child process started by safe_runner.run_code.

Usage: python -I _harness.py <work_dir>
  <work_dir>/code.py   the generated code
  <work_dir>/data.pkl  the DataFrame (available to the code as `df`)
Writes <work_dir>/output.json with the result, the figure and any error.
"""

import builtins
import contextlib
import io
import json
import sys
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

ALLOWED_MODULES = {"pandas", "numpy", "plotly", "math", "datetime", "statistics"}
MAX_ROWS = 200
MAX_STDOUT = 4000

_real_import = builtins.__import__


def _safe_import(name, globals=None, locals=None, fromlist=(), level=0):
    """Only allow imports of approved libraries from the generated code."""
    if level != 0 or name.split(".")[0] not in ALLOWED_MODULES:
        raise ImportError(f"Import of '{name}' is not allowed.")
    return _real_import(name, globals, locals, fromlist, level)


def _safe_builtins() -> dict:
    blocked = {"open", "eval", "exec", "compile", "input", "breakpoint", "exit", "quit", "help"}
    safe = {k: v for k, v in vars(builtins).items() if k not in blocked}
    safe["__import__"] = _safe_import
    return safe


def _to_jsonable(value):
    """Turn `result` into something JSON can store (tables are cut to MAX_ROWS)."""
    if value is None:
        return None, "none"
    if isinstance(value, pd.Series):
        value = value.to_frame(name=value.name if value.name is not None else "value")
    if isinstance(value, pd.DataFrame):
        table = value.head(MAX_ROWS)
        # Keep a meaningful index (e.g. after groupby) as normal columns
        if not isinstance(table.index, pd.RangeIndex):
            table = table.reset_index()
        table.columns = [str(c) for c in table.columns]
        return json.loads(table.to_json(orient="split", date_format="iso", index=False)), "table"
    if isinstance(value, (np.generic,)):
        value = value.item()
    if isinstance(value, float):
        return f"{value:,.4g}" if abs(value) < 1e4 else f"{value:,.2f}", "text"
    if isinstance(value, dict):
        return json.dumps(value, default=str, indent=2)[:MAX_STDOUT], "text"
    return str(value)[:MAX_STDOUT], "text"


def main() -> None:
    work = Path(sys.argv[1])
    code = (work / "code.py").read_text(encoding="utf-8")
    df = pd.read_pickle(work / "data.pkl")

    namespace = {
        "__builtins__": _safe_builtins(),
        "df": df, "pd": pd, "np": np, "px": px, "go": go,
        "result": None, "fig": None,
    }
    output = {"ok": True, "error": "", "stdout": ""}
    buffer = io.StringIO()
    try:
        with contextlib.redirect_stdout(buffer):
            exec(compile(code, "generated_code.py", "exec"), namespace)
    except Exception as exc:
        # Keep only the frames from the generated code so the LLM sees a short, useful error.
        tb = traceback.extract_tb(exc.__traceback__)
        lines = [f"line {f.lineno}: {f.line}" for f in tb if f.filename == "generated_code.py"]
        where = f" (at {lines[-1]})" if lines else ""
        output.update(ok=False, error=f"{type(exc).__name__}: {exc}{where}")

    output["stdout"] = buffer.getvalue()[:MAX_STDOUT]

    if output["ok"]:
        try:
            output["result"], output["result_kind"] = _to_jsonable(namespace.get("result"))
            fig = namespace.get("fig")
            if fig is not None:
                if not isinstance(fig, go.Figure):
                    raise TypeError("`fig` must be a plotly Figure (use plotly.express or graph_objects).")
                output["fig_json"] = fig.to_json()
            if output["result"] is None and fig is None:
                raise ValueError("The code did not set `result` or `fig`.")
        except Exception as exc:
            output.update(ok=False, error=f"{type(exc).__name__}: {exc}")

    (work / "output.json").write_text(json.dumps(output, default=str), encoding="utf-8")


if __name__ == "__main__":
    main()
