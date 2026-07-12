"""Minimal SmartTab regression example.

Run with:  python examples/quickstart_regression.py
"""

import numpy as np
import pandas as pd

import smarttab


def make_demo_dataframe(n=2000, seed=42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    df = pd.DataFrame(
        {
            "sqft": np.clip(rng.normal(1800, 500, size=n), 300, None),
            "bedrooms": rng.integers(1, 6, size=n),
            "age_years": rng.integers(0, 80, size=n),
            "neighborhood": rng.choice(["downtown", "suburb", "rural"], size=n),
        }
    )
    price = 150 * df["sqft"] + 8000 * df["bedrooms"] - 500 * df["age_years"] + rng.normal(0, 20000, size=n)
    df["price"] = price.clip(lower=20000)
    return df


if __name__ == "__main__":
    df = make_demo_dataframe()

    model = smarttab.fit(df, target="price")

    print("Selected model:", model.model_name)
    print("Metrics:", model.metrics)

    sample = df.drop(columns=["price"]).head(5)
    print("Predictions:", model.predict(sample))

    report_path = model.report()
    print("Report written to:", report_path)
