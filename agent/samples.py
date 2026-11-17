"""Bundled sample datasets and clickable example questions."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


@dataclass(frozen=True)
class Sample:
    key: str
    label: str
    filename: str
    description: str
    questions: tuple[str, ...]

    def load(self) -> pd.DataFrame:
        return pd.read_csv(DATA_DIR / self.filename)


SAMPLES: dict[str, Sample] = {
    "sales": Sample(
        key="sales",
        label="Sales in Pakistan (2024-2025)",
        filename="sales_pakistan.csv",
        description="Monthly sales by city, product category and channel, in PKR.",
        questions=(
            "Which city brings in the most revenue, and how big is the gap to the next one?",
            "How has the Online channel's share of revenue changed month by month?",
            "Is there a city where sales dip in a particular quarter?",
            "Which product category grew the fastest from 2024 to 2025?",
        ),
    ),
    "tickets": Sample(
        key="tickets",
        label="Customer support tickets (2025)",
        filename="customer_support_tickets.csv",
        description="Support tickets with category, priority, channel, response time and satisfaction.",
        questions=(
            "Which ticket category has the lowest customer satisfaction?",
            "Does a slower first response lead to lower satisfaction?",
            "Which support channel answers fastest on average?",
            "How did ticket volume change month by month, and was there any spike?",
        ),
    ),
}
