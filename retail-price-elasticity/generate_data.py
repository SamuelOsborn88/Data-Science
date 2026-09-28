"""Generate the synthetic weekly retail dataset used by analyze.py.

The data is fake (parody brand names) but built to behave like real retail
scan data: seasonal demand, holiday spikes, temporary promotions, a mid-year
list-price increase, and random noise. Each brand has a known "true" price
elasticity, saved alongside the data so the analysis can be checked against it.

Run:  python generate_data.py
"""
from pathlib import Path

import numpy as np
import pandas as pd

SEED = 42
WEEKS = pd.date_range("2025-01-06", periods=52, freq="W-MON")
DATA_DIR = Path(__file__).parent / "data"

# brand: (base shelf price, base weekly units, true elasticity)
BRANDS = {
    "Bud Right": (18.99, 1500, -1.3),
    "Budmeiser": (17.99, 1300, -1.6),
    "Coors Blight": (18.49, 1400, -0.8),
    "Michelob Ultra Combo Breaker": (19.99, 1200, -2.1),
}
# Weeks (Mondays) containing US beer-heavy holidays: Memorial Day, July 4th, Labor Day
HOLIDAY_WEEKS = {"2025-05-26": 0.25, "2025-06-30": 0.35, "2025-09-01": 0.20}


def seasonality(weeks):
    """Summer peak / winter dip, plus holiday bumps (multiplier around 1.0)."""
    week_of_year = weeks.isocalendar().week.to_numpy()
    season = 1 + 0.18 * np.sin(2 * np.pi * (week_of_year - 13) / 52)
    holiday = np.array([1 + HOLIDAY_WEEKS.get(w.strftime("%Y-%m-%d"), 0) for w in weeks])
    return season * holiday


def price_path(base, rng):
    """Shelf price with a mid-year list increase and random 1–2 week promotions."""
    price = np.full(len(WEEKS), base)
    increase_week = rng.integers(22, 30)
    price[increase_week:] *= 1 + rng.uniform(0.03, 0.06)
    week = rng.integers(2, 6)
    while week < len(WEEKS):
        length = rng.integers(1, 3)
        price[week:week + length] *= 1 - rng.uniform(0.08, 0.20)
        week += length + rng.integers(4, 8)
    return np.round(price * 4) / 4 - 0.01          # retail-style prices: x.24, x.49, x.74, x.99


def main():
    rng = np.random.default_rng(SEED)
    season = seasonality(WEEKS)
    rows, truth = [], []
    for brand, (base_price, base_units, elasticity) in BRANDS.items():
        price = price_path(base_price, rng)
        noise = rng.lognormal(0, 0.05, len(WEEKS))
        units = base_units * (price / base_price) ** elasticity * season * noise
        rows.append(pd.DataFrame({"week": WEEKS.date, "brand": brand,
                                  "price": price, "units_sold": np.round(units).astype(int)}))
        truth.append({"brand": brand, "true_elasticity": elasticity})

    DATA_DIR.mkdir(exist_ok=True)
    pd.concat(rows).to_csv(DATA_DIR / "retail_price_volume.csv", index=False)
    pd.DataFrame(truth).to_csv(DATA_DIR / "true_elasticities.csv", index=False)
    print(f"Wrote {len(WEEKS) * len(BRANDS)} rows to {DATA_DIR}")


if __name__ == "__main__":
    main()
