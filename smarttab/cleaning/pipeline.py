"""Stage 2 — Smart Cleaning.

``SmartCleaningPipeline`` turns raw feature columns into a model-ready
DataFrame: it drops useless columns, extracts datetime/text features,
imputes missing values, encodes categoricals, optionally scales, and does a
conservative feature-selection pass. All decisions made during ``fit`` are
stored on the instance so ``transform`` can be replayed identically on new
data at predict time (and so the whole object is joblib-serializable for
``save``/``load``).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from smarttab.analysis.dataset_analyzer import DatasetProfile
from smarttab.cleaning.datetime_features import extract_datetime_features
from smarttab.cleaning.encoders import OrdinalCategoricalEncoder
from smarttab.cleaning.feature_selection import select_columns_to_drop
from smarttab.cleaning.imputers import CategoricalConstantImputer, NumericMedianImputer
from smarttab.cleaning.scalers import resolve_scaler
from smarttab.exceptions import ConfigurationError

NEAR_DUPLICATE_OF_TARGET_THRESHOLD = 0.999


@dataclass
class CleaningConfig:
    missing: str = "auto"
    categorical: str = "auto"
    scaling: str = "auto"
    feature_selection: str = "auto"


@dataclass
class CleaningReport:
    dropped_constant_columns: list[str] = field(default_factory=list)
    dropped_id_columns: list[str] = field(default_factory=list)
    dropped_duplicate_columns: list[str] = field(default_factory=list)
    dropped_leakage_columns: list[str] = field(default_factory=list)
    dropped_near_duplicate_numeric_columns: list[str] = field(default_factory=list)
    datetime_columns_expanded: list[str] = field(default_factory=list)
    text_columns_expanded: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


class SmartCleaningPipeline:
    def __init__(
        self,
        missing: str = "auto",
        categorical: str = "auto",
        scaling: str = "auto",
        feature_selection: str = "auto",
    ) -> None:
        if missing not in ("auto",):
            raise ConfigurationError(f"missing must be 'auto', got {missing!r}")
        if categorical not in ("auto",):
            raise ConfigurationError(f"categorical must be 'auto', got {categorical!r}")
        if feature_selection not in ("auto", "none"):
            raise ConfigurationError(f"feature_selection must be 'auto' or 'none', got {feature_selection!r}")

        self.config = CleaningConfig(
            missing=missing, categorical=categorical, scaling=scaling, feature_selection=feature_selection
        )

        self.numeric_imputer_ = NumericMedianImputer()
        self.categorical_imputer_ = CategoricalConstantImputer()
        self.categorical_encoder_ = OrdinalCategoricalEncoder()
        self.scaler_ = resolve_scaler(scaling)

        self.dropped_source_columns_: list[str] = []
        self.datetime_source_columns_: list[str] = []
        self.text_source_columns_: list[str] = []
        self.numeric_columns_: list[str] = []
        self.categorical_columns_: list[str] = []
        self.final_feature_columns_: list[str] = []
        self.report_ = CleaningReport()
        self._fitted = False

    @property
    def final_categorical_columns(self) -> list[str]:
        return [c for c in self.categorical_columns_ if c in self.final_feature_columns_]

    @property
    def final_numeric_columns(self) -> list[str]:
        return [c for c in self.numeric_columns_ if c in self.final_feature_columns_]

    def fit_transform(self, df: pd.DataFrame, y: pd.Series | pd.DataFrame, profile: DatasetProfile) -> pd.DataFrame:
        work = df[profile.feature_columns].copy()

        drop_cols = self._resolve_columns_to_drop(work, y, profile)
        self.dropped_source_columns_ = drop_cols
        work = work.drop(columns=drop_cols, errors="ignore")

        self.datetime_source_columns_ = [c for c in profile.datetime_columns if c in work.columns]
        if self.datetime_source_columns_:
            work = extract_datetime_features(work, self.datetime_source_columns_)
            self.report_.datetime_columns_expanded = list(self.datetime_source_columns_)

        self.text_source_columns_ = [c for c in profile.text_columns if c in work.columns]
        if self.text_source_columns_:
            work = self._expand_text_columns(work, self.text_source_columns_)
            self.report_.text_columns_expanded = list(self.text_source_columns_)

        generated_numeric = [
            c for c in work.columns
            if any(c.startswith(f"{src}_") for src in self.datetime_source_columns_ + self.text_source_columns_)
        ]
        self.numeric_columns_ = [c for c in profile.numeric_columns if c in work.columns] + generated_numeric
        self.categorical_columns_ = [c for c in profile.categorical_columns if c in work.columns]

        self.numeric_imputer_.fit(work, self.numeric_columns_)
        work = self.numeric_imputer_.transform(work)
        self.categorical_imputer_.fit(work, self.categorical_columns_)
        work = self.categorical_imputer_.transform(work)

        self.categorical_encoder_.fit(work, self.categorical_columns_)
        work = self.categorical_encoder_.transform(work)

        if self.scaler_ is not None and self.numeric_columns_:
            work[self.numeric_columns_] = self.scaler_.fit_transform(work[self.numeric_columns_])

        if self.config.feature_selection == "auto":
            fs_drop = select_columns_to_drop(work, self.numeric_columns_)
            if fs_drop:
                work = work.drop(columns=fs_drop)
                self.numeric_columns_ = [c for c in self.numeric_columns_ if c not in fs_drop]
                self.report_.dropped_near_duplicate_numeric_columns = fs_drop

        self.final_feature_columns_ = list(work.columns)
        self._fitted = True
        return work

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self._fitted:
            raise RuntimeError("SmartCleaningPipeline must be fit before transform() can be called")

        work = df.drop(columns=self.dropped_source_columns_, errors="ignore")

        if self.datetime_source_columns_:
            present = [c for c in self.datetime_source_columns_ if c in work.columns]
            work = extract_datetime_features(work, present)

        if self.text_source_columns_:
            present = [c for c in self.text_source_columns_ if c in work.columns]
            work = self._expand_text_columns(work, present)

        work = self.numeric_imputer_.transform(work)
        work = self.categorical_imputer_.transform(work)
        work = self.categorical_encoder_.transform(work)

        if self.scaler_ is not None and self.numeric_columns_:
            cols = [c for c in self.numeric_columns_ if c in work.columns]
            work[cols] = self.scaler_.transform(work[cols])

        return work.reindex(columns=self.final_feature_columns_, fill_value=0)

    def _resolve_columns_to_drop(self, work: pd.DataFrame, y: pd.Series | pd.DataFrame, profile: DatasetProfile) -> list[str]:
        constant = list(profile.constant_columns)
        id_like = list(profile.id_like_columns)

        duplicate_extra: list[str] = []
        for group in profile.duplicate_column_groups:
            duplicate_extra.extend(group[1:])

        y_frame = y.to_frame() if isinstance(y, pd.Series) else y
        leakage_near_dup: list[str] = []
        for c in profile.potential_leakage_columns:
            if c not in profile.numeric_columns or c not in work.columns:
                continue
            try:
                corrs = [abs(work[c].corr(pd.to_numeric(y_frame[col], errors="coerce"))) for col in y_frame.columns]
                max_corr = max((v for v in corrs if pd.notna(v)), default=None)
            except Exception:
                continue
            if max_corr is not None and max_corr >= NEAR_DUPLICATE_OF_TARGET_THRESHOLD:
                leakage_near_dup.append(c)

        self.report_.dropped_constant_columns = constant
        self.report_.dropped_id_columns = id_like
        self.report_.dropped_duplicate_columns = duplicate_extra
        self.report_.dropped_leakage_columns = leakage_near_dup

        return sorted(set(constant) | set(id_like) | set(duplicate_extra) | set(leakage_near_dup))

    @staticmethod
    def _expand_text_columns(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
        df = df.copy()
        for c in columns:
            text = df[c].astype("string").fillna("")
            df[f"{c}_len"] = text.str.len().astype("float32")
            df[f"{c}_word_count"] = text.str.split().str.len().fillna(0).astype("float32")
            df = df.drop(columns=[c])
        return df
