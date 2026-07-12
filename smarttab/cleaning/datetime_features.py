"""Datetime feature extraction: replaces each raw datetime column with numeric parts."""

from __future__ import annotations

import pandas as pd


def datetime_feature_names(df: pd.DataFrame, column: str) -> list[str]:
    parsed = pd.to_datetime(df[column], errors="coerce", format="mixed")
    names = [f"{column}_year", f"{column}_month", f"{column}_day", f"{column}_dayofweek"]
    if (parsed.dt.hour.fillna(0) != 0).any() or (parsed.dt.minute.fillna(0) != 0).any():
        names.append(f"{column}_hour")
    return names


def extract_datetime_features(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    df = df.copy()
    for column in columns:
        if column not in df.columns:
            continue
        parsed = pd.to_datetime(df[column], errors="coerce", format="mixed")
        df[f"{column}_year"] = parsed.dt.year.astype("float32")
        df[f"{column}_month"] = parsed.dt.month.astype("float32")
        df[f"{column}_day"] = parsed.dt.day.astype("float32")
        df[f"{column}_dayofweek"] = parsed.dt.dayofweek.astype("float32")
        if f"{column}_hour" in datetime_feature_names(df, column):
            df[f"{column}_hour"] = parsed.dt.hour.astype("float32")
        df = df.drop(columns=[column])
    return df
