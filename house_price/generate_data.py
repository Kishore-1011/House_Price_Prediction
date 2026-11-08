"""Create the made-up housing dataset.

None of this describes real houses. The prices come from the formula below plus random
noise, so the data is only useful for learning and demonstrating the workflow.

Run with: python -m house_price.generate_data
"""

import numpy as np
import pandas as pd

from house_price.config import LOCALITY_RATES, RAW_DATA, SEED

DEFAULT_ROWS = 2000
MISSING_BEDROOM_SHARE = 0.03
MISSING_AGE_SHARE = 0.02


def generate_housing_data(rows=DEFAULT_ROWS, seed=SEED):
    """Return a DataFrame of invented houses. The same seed always gives the same data."""
    rng = np.random.default_rng(seed)

    area = np.clip(rng.lognormal(mean=7.6, sigma=0.4, size=rows), 600, 6000)
    area = (area / 10).round() * 10
    bedroom = np.clip(np.round(area / 800 + rng.normal(0.5, 0.5, rows)), 1, 6).astype(int)
    bathroom = np.clip(bedroom - rng.integers(0, 2, rows), 1, 5).astype(int)
    age = np.clip(np.round(rng.exponential(12, rows)), 0, 40).astype(int)

    names = list(LOCALITY_RATES)
    locality = rng.choice(names, size=rows, p=[0.12, 0.15, 0.2, 0.2, 0.18, 0.15])
    rate = np.array([LOCALITY_RATES[name] for name in locality])

    wear = np.exp(-0.009 * age)
    extras = 1 + 0.015 * bedroom + 0.02 * bathroom
    noise = rng.lognormal(mean=0, sigma=0.08, size=rows)
    price = np.round(area * rate * wear * extras * noise, -3).astype(int)

    data = pd.DataFrame(
        {
            "area": area.astype(int),
            "bedroom": bedroom.astype(float),
            "bathroom": bathroom,
            "age": age.astype(float),
            "locality": locality,
            "price": price,
        }
    )
    data.loc[rng.random(rows) < MISSING_BEDROOM_SHARE, "bedroom"] = np.nan
    data.loc[rng.random(rows) < MISSING_AGE_SHARE, "age"] = np.nan
    return data


def main():
    RAW_DATA.parent.mkdir(parents=True, exist_ok=True)
    data = generate_housing_data()
    data.to_csv(RAW_DATA, index=False)
    print(f"Wrote {len(data)} made-up houses to {RAW_DATA.relative_to(RAW_DATA.parents[2])}")


if __name__ == "__main__":
    main()
