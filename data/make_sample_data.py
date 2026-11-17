"""Generate the bundled sample datasets with a fixed random seed.

Run:  .venv\\Scripts\\python data\\make_sample_data.py

The data is fake but has a few real-looking patterns baked in so the
agent finds interesting insights:
  sales_pakistan.csv
    - Karachi dips hard in Q3 (Jul-Sep) every year (monsoon / flooding).
    - The Online channel grows steadily month after month.
    - Groceries and Apparel spike in the Ramadan / Eid months.
  customer_support_tickets.csv
    - Delivery tickets are the most common and have the lowest satisfaction.
    - Satisfaction falls as first-response time grows.
    - WhatsApp and Chat are answered fastest; Email is slowest.
    - Billing tickets spike in June 2025 (after a price change).
    - Some satisfaction scores are missing (customer skipped the survey).
"""

from pathlib import Path

import numpy as np
import pandas as pd

SEED = 42
DATA_DIR = Path(__file__).parent


def make_sales(rng: np.random.Generator) -> pd.DataFrame:
    months = pd.date_range("2024-01-01", "2025-12-01", freq="MS")
    # City size multipliers (rough population / market size)
    cities = {
        "Karachi": 1.8, "Lahore": 1.6, "Islamabad": 1.0, "Rawalpindi": 0.8,
        "Faisalabad": 0.9, "Multan": 0.6, "Peshawar": 0.7, "Quetta": 0.4,
    }
    # Base monthly units and average unit price in PKR per category
    categories = {
        "Electronics": (120, 45_000), "Apparel": (900, 3_500),
        "Groceries": (4_000, 650), "Home & Kitchen": (500, 6_000),
        "Beauty": (1_100, 1_800),
    }
    channels = {"Retail Store": 0.55, "Distributor": 0.30, "Online": 0.15}
    # Months containing Ramadan / Eid in 2024-2025 (approximate)
    festive_months = {"2024-03", "2024-04", "2025-03", "2025-06"}

    rows = []
    for i, month in enumerate(months):
        for city, city_mult in cities.items():
            for cat, (base_units, base_price) in categories.items():
                for channel, share in channels.items():
                    mult = city_mult * share
                    # Online grows ~4% per month; other channels are flat-ish
                    if channel == "Online":
                        mult *= 1.04 ** i
                    else:
                        mult *= 1.003 ** i
                    # Karachi Q3 dip
                    if city == "Karachi" and month.month in (7, 8, 9):
                        mult *= 0.6
                    # Festive spike for groceries and apparel
                    if month.strftime("%Y-%m") in festive_months and cat in ("Groceries", "Apparel"):
                        mult *= 1.45
                    # Electronics sell more in November (11.11 / Black Friday sales)
                    if cat == "Electronics" and month.month == 11:
                        mult *= 1.5

                    units = max(1, int(rng.normal(base_units * mult, base_units * mult * 0.08)))
                    price = base_price * rng.uniform(0.95, 1.05)
                    if channel == "Online":
                        price *= 0.95  # online is a bit cheaper
                    rows.append({
                        "month": month.strftime("%Y-%m"),
                        "city": city,
                        "category": cat,
                        "channel": channel,
                        "units": units,
                        "revenue_pkr": round(units * price),
                    })
    return pd.DataFrame(rows)


def make_tickets(rng: np.random.Generator, n_normal: int = 1800, n_june_billing: int = 150) -> pd.DataFrame:
    start = pd.Timestamp("2025-01-01")
    # Normal tickets spread over the year, plus extra Billing tickets in June 2025
    # (customers complaining after a price change) -> a visible spike.
    days = np.concatenate([rng.integers(0, 365, n_normal), rng.integers(151, 181, n_june_billing)])
    n = len(days)
    created = start + pd.to_timedelta(days, unit="D") + pd.to_timedelta(rng.integers(8, 22, n), unit="h")

    categories = np.array(["Delivery", "Billing", "Technical", "Account", "Refund"])
    cat_p = np.array([0.34, 0.20, 0.20, 0.12, 0.14])
    category = np.concatenate([rng.choice(categories, n_normal, p=cat_p), np.full(n_june_billing, "Billing")])

    channel = rng.choice(["Email", "Phone", "Chat", "WhatsApp"], n, p=[0.30, 0.20, 0.25, 0.25])
    priority = rng.choice(["Low", "Medium", "High", "Urgent"], n, p=[0.30, 0.40, 0.22, 0.08])
    city = rng.choice(["Karachi", "Lahore", "Islamabad", "Peshawar", "Quetta", "Multan"], n,
                      p=[0.30, 0.27, 0.15, 0.11, 0.07, 0.10])

    # First response time (hours) depends on channel and priority
    channel_base = {"Email": 14.0, "Phone": 2.0, "Chat": 1.0, "WhatsApp": 0.8}
    priority_mult = {"Low": 1.5, "Medium": 1.0, "High": 0.6, "Urgent": 0.3}
    base = np.array([channel_base[c] * priority_mult[p] for c, p in zip(channel, priority)])
    first_response = np.round(rng.gamma(2.0, base / 2.0), 1)

    # Resolution time: category matters, plus a bit of response time
    cat_res = {"Delivery": 48, "Billing": 20, "Technical": 36, "Account": 12, "Refund": 72}
    resolution = np.round(np.array([rng.gamma(2.0, cat_res[c] / 2.0) for c in category]) + first_response, 1)

    # Satisfaction (1-5): drops with slow responses; Delivery and Refund are worse
    cat_penalty = {"Delivery": -0.8, "Refund": -0.5, "Billing": -0.2, "Technical": 0.0, "Account": 0.3}
    score = 4.4 - 0.06 * first_response - 0.01 * resolution
    score += np.array([cat_penalty[c] for c in category]) + rng.normal(0, 0.6, n)
    satisfaction = np.clip(np.round(score), 1, 5).astype(float)
    # ~15% of customers skip the survey
    satisfaction[rng.random(n) < 0.15] = np.nan

    status = rng.choice(["Resolved", "Open", "Escalated"], n, p=[0.85, 0.09, 0.06])

    df = pd.DataFrame({
        "created_at": created.strftime("%Y-%m-%d %H:%M"),
        "city": city,
        "channel": channel,
        "category": category,
        "priority": priority,
        "first_response_hours": first_response,
        "resolution_hours": resolution,
        "satisfaction": satisfaction,
        "status": status,
    })
    df = df.sort_values("created_at").reset_index(drop=True)
    # Ticket ids in time order
    df.insert(0, "ticket_id", [f"T-{10001 + i}" for i in range(len(df))])
    return df


def main() -> None:
    rng = np.random.default_rng(SEED)
    sales = make_sales(rng)
    tickets = make_tickets(rng)
    sales.to_csv(DATA_DIR / "sales_pakistan.csv", index=False)
    tickets.to_csv(DATA_DIR / "customer_support_tickets.csv", index=False)
    print(f"sales_pakistan.csv: {len(sales)} rows")
    print(f"customer_support_tickets.csv: {len(tickets)} rows")


if __name__ == "__main__":
    main()
