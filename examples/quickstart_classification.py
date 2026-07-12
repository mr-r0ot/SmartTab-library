"""Minimal SmartTab classification example.

Run with:  python examples/quickstart_classification.py
"""

import numpy as np
import pandas as pd

import smarttab


def make_demo_dataframe(n=2000, seed=42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    df = pd.DataFrame(
        {
            "age": rng.integers(18, 70, size=n),
            "income": rng.normal(50000, 15000, size=n),
            "credit_score": rng.normal(650, 80, size=n),
            "city": rng.choice(["tehran", "isfahan", "shiraz", "mashhad"], size=n),
            "signup_date": pd.date_range("2020-01-01", periods=n, freq="h").astype(str),
        }
    )
    score = 0.00002 * df["income"] + 0.01 * df["credit_score"] - 0.02 * df["age"]
    df["churned"] = (score + rng.normal(0, 1, size=n) < score.median()).astype(int)
    return df


if __name__ == "__main__":
    df = make_demo_dataframe()

    model = smarttab.fit(df, target="churned")

    print("Selected model:", model.model_name)
    print("Metrics:", model.metrics)

    sample = df.drop(columns=["churned"]).head(5)
    print("Predictions:", model.predict(sample))

    model.save("smarttab_reports/churn_model.smarttab")
    reloaded = smarttab.load("smarttab_reports/churn_model.smarttab")
    print("Reloaded predictions match:", (reloaded.predict(sample) == model.predict(sample)).all())
