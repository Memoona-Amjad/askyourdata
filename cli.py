"""Command-line version of the agent (useful for testing without the UI).

Examples (Windows):
  .venv\\Scripts\\python cli.py --sample sales "Which city has the most revenue?"
  .venv\\Scripts\\python cli.py --file my_data.csv "What are the top 5 products?"
"""

from __future__ import annotations

import argparse
import sys

from agent.analyst import Analyst, Step
from agent.profile import load_table
from agent.samples import SAMPLES


def print_step(step: Step) -> None:
    print(f"\n=== {step.title} ===")
    if step.content:
        print(step.content if step.kind != "look" else step.content[:600] + " ...")
    if step.code:
        print("--- code ---\n" + step.code)


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="Ask a question about a data file.")
    parser.add_argument("question")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--sample", choices=list(SAMPLES))
    group.add_argument("--file")
    args = parser.parse_args()

    if args.sample:
        df = SAMPLES[args.sample].load()
    else:
        with open(args.file, "rb") as f:
            df = load_table(f, args.file)

    run = Analyst(df).answer(args.question, on_step=print_step)
    print("\n" + "=" * 60)
    if run.success:
        print(f"ANSWER (after {run.attempts} tr{'y' if run.attempts == 1 else 'ies'}):")
        print(run.result_as_text())
        print("\nINSIGHT:", run.insight)
        print("Chart produced:", "yes" if run.fig_json else "no")
        return 0
    print("FAILED:", run.error_message)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
