"""Safe code runner (demo-level safety, NOT a full sandbox).

Two layers:
  1. Static check with Python's `ast`: only a few libraries may be imported,
     and dangerous built-ins / dunder access are rejected before anything runs.
  2. The code runs in a separate Python subprocess, inside a temp folder,
     with a timeout. The data is handed over as a pickle file and the
     results come back as JSON.

A production system would run this inside a locked-down container sandbox
(no network, read-only file system, CPU/memory limits).
"""

from __future__ import annotations

import ast
import json
import os
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

TIMEOUT_SECONDS = 20

# Top-level modules the generated code may import.
ALLOWED_MODULES = {"pandas", "numpy", "plotly", "math", "datetime", "statistics"}

# Names that must never appear in generated code.
BLOCKED_NAMES = {
    "os", "sys", "subprocess", "socket", "shutil", "requests", "urllib", "pathlib",
    "open", "eval", "exec", "compile", "__import__", "input", "breakpoint",
    "globals", "locals", "vars", "getattr", "setattr", "delattr", "help", "exit", "quit",
    "importlib", "builtins", "pickle", "ctypes",
}

# Attribute calls that could read/write arbitrary files or run code.
BLOCKED_ATTRIBUTES = {
    "system", "popen", "read_pickle", "to_pickle", "read_csv", "read_excel",
    "read_json", "read_html", "read_sql", "read_parquet", "to_csv", "to_excel",
    "to_json", "to_html", "to_parquet", "write_html", "write_image", "show",
}

HARNESS = Path(__file__).with_name("_harness.py")


@dataclass
class RunResult:
    """What came back from running one piece of generated code."""

    ok: bool
    error: str = ""                       # short error message the LLM can read
    stdout: str = ""                      # anything the code printed
    result: Any = None                    # JSON-friendly version of `result`
    result_kind: str = "none"             # "table", "text" or "none"
    fig_json: str | None = None           # plotly figure as JSON (or None)
    extra: dict[str, Any] = field(default_factory=dict)


def check_code(code: str) -> list[str]:
    """Return a list of safety problems found in `code` (empty list = looks safe)."""
    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        return [f"SyntaxError: {exc.msg} (line {exc.lineno})"]

    problems: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                root = alias.name.split(".")[0]
                if root not in ALLOWED_MODULES:
                    problems.append(f"Import of '{alias.name}' is not allowed.")
        elif isinstance(node, ast.ImportFrom):
            root = (node.module or "").split(".")[0]
            if node.level or root not in ALLOWED_MODULES:
                problems.append(f"Import from '{node.module}' is not allowed.")
        elif isinstance(node, ast.Name):
            if node.id in BLOCKED_NAMES:
                problems.append(f"Use of '{node.id}' is not allowed.")
            elif node.id.startswith("__"):
                problems.append(f"Dunder name '{node.id}' is not allowed.")
        elif isinstance(node, ast.Attribute):
            if node.attr.startswith("__"):
                problems.append(f"Dunder attribute '.{node.attr}' is not allowed.")
            elif node.attr in BLOCKED_ATTRIBUTES:
                problems.append(f"'.{node.attr}' is not allowed (no file, network or display access).")
        elif isinstance(node, (ast.Global, ast.Nonlocal)):
            problems.append("global/nonlocal statements are not allowed.")

    # Remove duplicates but keep order
    return list(dict.fromkeys(problems))


def run_code(code: str, df: pd.DataFrame, timeout: int = TIMEOUT_SECONDS) -> RunResult:
    """Check and run `code` against `df` in a subprocess. Never raises."""
    problems = check_code(code)
    if problems:
        return RunResult(ok=False, error="Safety check failed: " + " ".join(problems))

    with tempfile.TemporaryDirectory(prefix="askyourdata_") as tmp:
        tmp_path = Path(tmp)
        (tmp_path / "code.py").write_text(code, encoding="utf-8")
        df.to_pickle(tmp_path / "data.pkl")

        # Minimal environment: no API keys or other secrets are passed to the child.
        env = {k: os.environ[k] for k in ("PATH", "SYSTEMROOT", "TEMP", "TMP") if k in os.environ}
        env["PYTHONIOENCODING"] = "utf-8"

        try:
            proc = subprocess.run(
                [sys.executable, "-I", str(HARNESS), str(tmp_path)],
                cwd=tmp_path,
                env=env,
                capture_output=True,
                text=True,
                encoding="utf-8",
                timeout=timeout,
            )
        except subprocess.TimeoutExpired:
            return RunResult(ok=False, error=f"TimeoutError: the code took longer than {timeout} seconds.")

        out_file = tmp_path / "output.json"
        if not out_file.exists():
            # The harness itself crashed (should be rare).
            tail = (proc.stderr or "").strip().splitlines()[-3:]
            return RunResult(ok=False, error="Runner crashed: " + " ".join(tail))

        data = json.loads(out_file.read_text(encoding="utf-8"))

    return RunResult(
        ok=data["ok"],
        error=data.get("error", ""),
        stdout=data.get("stdout", ""),
        result=data.get("result"),
        result_kind=data.get("result_kind", "none"),
        fig_json=data.get("fig_json"),
    )
