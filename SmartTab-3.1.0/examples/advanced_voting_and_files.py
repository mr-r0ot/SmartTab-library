"""Advanced SmartTab example: loading data from a CSV file and ensemble="auto".

Demonstrates:
  - passing a file path (not a DataFrame) to fit()
  - ensemble="auto" (OOF-specialist CatBoost/LightGBM candidates, diversity
    pruning, weighted soft voting, and stacking; XGBoost remains optional)
  - the folder-based report: HTML + JSON + chart PNGs, and the JSON also
    returned in-memory as a plain dict

Run with:  python examples/advanced_voting_and_files.py
"""

import numpy as np
import pandas as pd

import smarttab


def make_demo_csv(path: str, n=1500, seed=42) -> None:
    rng = np.random.default_rng(seed)
    df = pd.DataFrame(
        {
            "age": rng.integers(18, 70, size=n),
            "income": rng.normal(50000, 15000, size=n),
            "credit_score": rng.normal(650, 80, size=n),
            "city": rng.choice(["tehran", "isfahan", "shiraz", "mashhad"], size=n),
        }
    )
    score = 0.00002 * df["income"] + 0.01 * df["credit_score"] - 0.02 * df["age"]
    df["churned"] = (score + rng.normal(0, 1, size=n) < score.median()).astype(int)
    df.to_csv(path, index=False)


if __name__ == "__main__":
    csv_path = "smarttab_reports/demo_customers.csv"
    import os

    os.makedirs("smarttab_reports", exist_ok=True)
    make_demo_csv(csv_path)

    # fit() accepts a file path directly (.csv/.tsv/.xlsx/.parquet/.json/.feather/...)
    model = smarttab.fit(
        csv_path,
        target="churned",
        ensemble="auto",
        ensemble_models_limit=5,
        n_trials=8,
        cv=3,
        report=False,
    )

    print("Chosen strategy:", model.model_name, "(used_ensemble:", model.ensemble_info is not None, ")")
    print("Metrics:", model.metrics)

    df = pd.read_csv(csv_path)
    sample = df.drop(columns=["churned"]).head(5)
    print("Predictions:", model.predict(sample))

    report_dict = model.report("smarttab_reports/voting_demo_report", df.drop(columns=["churned"]), df["churned"])
    print("Report folder:", report_dict["_paths"]["folder"])
    print("Report JSON top-level keys:", list(report_dict.keys()))
