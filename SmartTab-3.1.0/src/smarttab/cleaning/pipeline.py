"""Train-only cleaning pipeline with strict raw-input schema validation."""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from smarttab.analysis.dataset_analyzer import DatasetProfile
from smarttab.cleaning.encoders import NativeCategoricalEncoder, OrdinalCategoricalEncoder
from smarttab.cleaning.feature_selection import select_columns_to_drop, select_top_features
from smarttab.cleaning.imputers import CategoricalConstantImputer
from smarttab.cleaning.robust import (
    ConfigurableNumericImputer,
    MissingIndicatorAdder,
    NumericDistributionTransformer,
    NumericWinsorizer,
    RareCategoryGrouper,
)
from smarttab.datascience.config import DataScienceConfig
from smarttab.cleaning.scalers import resolve_scaler
from smarttab.exceptions import ConfigurationError, DataValidationError
from smarttab.multimodal.config import FeatureSpaceConfig
from smarttab.multimodal.pipeline import MultiModalFeaturePipeline


@dataclass
class CleaningConfig:
    clean: str = "auto"
    missing: str = "auto"
    categorical: str = "auto"
    scaling: str = "auto"
    feature_selection: str = "auto"
    outlier: str = "auto"
    leakage_policy: str = "drop"
    schema_policy: str = "strict"


@dataclass
class CleaningReport:
    dropped_constant_columns: list[str] = field(default_factory=list)
    dropped_id_columns: list[str] = field(default_factory=list)
    dropped_duplicate_columns: list[str] = field(default_factory=list)
    dropped_leakage_columns: list[str] = field(default_factory=list)
    warned_leakage_columns: list[str] = field(default_factory=list)
    dropped_near_duplicate_numeric_columns: list[str] = field(default_factory=list)
    datetime_columns_expanded: list[str] = field(default_factory=list)
    text_columns_expanded: list[str] = field(default_factory=list)
    multimodal_feature_count: int = 0
    multimodal_report: dict = field(default_factory=dict)
    missing_indicator_columns: list[str] = field(default_factory=list)
    numeric_imputation: str = "median"
    imputed_numeric_columns: list[str] = field(default_factory=list)
    rare_categories_grouped: dict[str, int] = field(default_factory=dict)
    winsorized_columns: dict[str, int] = field(default_factory=dict)
    log_transformed_columns: list[str] = field(default_factory=list)
    power_transformed_columns: list[str] = field(default_factory=list)
    model_feature_budget: int | None = None
    dropped_by_feature_budget: list[str] = field(default_factory=list)
    feature_selection_scores: dict[str, float] = field(default_factory=dict)
    before_feature_count: int = 0
    after_feature_count: int = 0
    before_missing_values: int = 0
    after_missing_values: int = 0
    notes: list[str] = field(default_factory=list)


class SmartCleaningPipeline:
    """Replayable transformations learned from training data only.

    ``clean='none'`` disables optional column removal and feature selection, but
    still performs the mandatory model adapter steps: schema validation, missing
    value handling, datetime conversion, and categorical encoding.
    """

    def __init__(
        self,
        clean: str = "auto",
        missing: str = "auto",
        categorical: str = "auto",
        scaling: str = "auto",
        feature_selection: str = "auto",
        outlier: str = "auto",
        leakage_policy: str = "drop",
        schema_policy: str = "strict",
        feature_space_config: FeatureSpaceConfig | None = None,
        data_science_config: DataScienceConfig | None = None,
    ) -> None:
        if clean not in ("auto", "minimal", "none"):
            raise ConfigurationError("clean must be 'auto', 'minimal', or 'none'")
        if missing not in ("auto", "median", "mean", "constant", "knn", "iterative"):
            raise ConfigurationError("missing must be auto, median, mean, constant, knn, or iterative")
        if categorical not in ("auto", "native", "ordinal"):
            raise ConfigurationError("categorical must be 'auto', 'native', or 'ordinal'")
        if feature_selection not in ("auto", "none"):
            raise ConfigurationError("feature_selection must be 'auto' or 'none'")
        if outlier not in ("auto", "keep", "remove", "clip"):
            raise ConfigurationError("outlier must be auto, keep, remove, or clip")
        if leakage_policy not in ("drop", "error", "warn", "ignore"):
            raise ConfigurationError("leakage_policy must be 'drop', 'error', 'warn', or 'ignore'")
        if schema_policy not in ("strict", "coerce"):
            raise ConfigurationError("schema_policy must be 'strict' or 'coerce'")

        self.feature_space_config = feature_space_config
        self.data_science_config = data_science_config or DataScienceConfig()
        self.multimodal_pipeline_: MultiModalFeaturePipeline | None = None
        self.config = CleaningConfig(
            clean=clean,
            missing=missing,
            categorical=categorical,
            scaling=scaling,
            feature_selection=feature_selection,
            outlier=outlier,
            leakage_policy=leakage_policy,
            schema_policy=schema_policy,
        )
        numeric_strategy = self.data_science_config.numeric_imputation if missing == "auto" else missing
        self.numeric_imputer_ = ConfigurableNumericImputer(
            numeric_strategy, self.data_science_config.numeric_fill_value
        )
        self.categorical_imputer_ = CategoricalConstantImputer()
        self.missing_indicator_ = MissingIndicatorAdder()
        self.rare_category_grouper_ = RareCategoryGrouper(
            self.data_science_config.rare_category_min_frequency,
            self.data_science_config.rare_category_max_categories,
        )
        winsor_fraction = self.data_science_config.winsorize
        if outlier == "clip":
            winsor_fraction = 0.01 if winsor_fraction in {"auto", "none"} else winsor_fraction
        elif outlier == "keep":
            winsor_fraction = 0.0
        elif winsor_fraction == "auto":
            winsor_fraction = 0.01 if clean == "auto" else 0.0
        elif winsor_fraction == "none":
            winsor_fraction = 0.0
        self.numeric_winsorizer_ = NumericWinsorizer(float(winsor_fraction or 0.0))
        self.numeric_transformer_ = NumericDistributionTransformer(
            self.data_science_config.numeric_transform if clean == "auto" else "none",
            self.data_science_config.skew_threshold,
        )
        self.categorical_encoder_ = (
            NativeCategoricalEncoder() if categorical in ("auto", "native")
            else OrdinalCategoricalEncoder()
        )
        self.scaler_ = resolve_scaler(scaling)

        self.raw_feature_columns_: list[str] = []
        self.raw_dtypes_: dict[str, str] = {}
        self.raw_numeric_columns_: list[str] = []
        self.raw_categorical_columns_: list[str] = []
        self.raw_datetime_columns_: list[str] = []
        self.raw_text_columns_: list[str] = []
        self.raw_image_columns_: list[str] = []
        self.raw_audio_columns_: list[str] = []
        self.raw_video_columns_: list[str] = []
        self.raw_column_modalities_: dict[str, str] = {}

        self.dropped_source_columns_: list[str] = []
        self.datetime_source_columns_: list[str] = []
        self.datetime_feature_names_: dict[str, list[str]] = {}
        self.text_source_columns_: list[str] = []
        self.numeric_columns_: list[str] = []
        self.categorical_columns_: list[str] = []
        self.final_feature_columns_: list[str] = []
        self.report_ = CleaningReport()
        self._fitted = False

    @property
    def final_categorical_columns(self) -> list[str]:
        return [column for column in self.categorical_columns_ if column in self.final_feature_columns_]

    @property
    def final_numeric_columns(self) -> list[str]:
        return [column for column in self.numeric_columns_ if column in self.final_feature_columns_]

    @property
    def feature_groups(self) -> dict[str, list[str]]:
        """Return stable feature subsets for modality-aware late/hybrid fusion."""
        final = list(self.final_feature_columns_)
        groups: dict[str, list[str]] = {"all": final}
        tabular = [name for name in final if not name.startswith("mm__")]
        if tabular:
            groups["tabular"] = tabular
        modality_groups: dict[str, list[str]] = {}
        if self.multimodal_pipeline_ is not None:
            for source, modality in self.multimodal_pipeline_.column_modalities.items():
                prefix = f"mm__{source}__"
                columns = [name for name in final if name.startswith(prefix)]
                if not columns:
                    continue
                groups[f"source:{source}"] = columns
                modality_groups.setdefault(modality, []).extend(columns)
        for modality, columns in modality_groups.items():
            groups[f"modality:{modality}"] = list(dict.fromkeys(columns))
        return groups

    def fit_transform(
        self,
        df: pd.DataFrame,
        y: pd.Series | pd.DataFrame,
        profile: DatasetProfile,
    ) -> pd.DataFrame:
        self.raw_feature_columns_ = list(profile.feature_columns)
        self.raw_dtypes_ = {column: str(df[column].dtype) for column in self.raw_feature_columns_}
        self.raw_numeric_columns_ = list(profile.numeric_columns)
        self.raw_categorical_columns_ = list(profile.categorical_columns)
        self.raw_datetime_columns_ = list(profile.datetime_columns)
        self.raw_text_columns_ = list(profile.text_columns)
        self.raw_image_columns_ = list(profile.image_columns)
        self.raw_audio_columns_ = list(profile.audio_columns)
        self.raw_video_columns_ = list(profile.video_columns)
        self.raw_column_modalities_ = dict(profile.column_modalities)

        work = self._validate_and_prepare_raw(df[self.raw_feature_columns_], fitting=True)
        self.report_.before_missing_values = int(work.isna().sum().sum())
        drop_columns = self._resolve_columns_to_drop(profile)
        self.dropped_source_columns_ = drop_columns
        work = work.drop(columns=drop_columns, errors="ignore")

        active_modalities = {
            column: modality
            for column, modality in profile.column_modalities.items()
            if column in work.columns
        }
        if active_modalities:
            if self.feature_space_config is None:
                raise ConfigurationError("multimodal columns require a feature-space configuration")
            self.multimodal_pipeline_ = MultiModalFeaturePipeline(
                self.feature_space_config,
                active_modalities,
                profile.task_type,
            )
            multimodal = self.multimodal_pipeline_.fit_transform(work, y)
            work = pd.concat([work.drop(columns=list(active_modalities)), multimodal], axis=1)
            self.report_.multimodal_feature_count = multimodal.shape[1]
            self.report_.multimodal_report = dataclasses.asdict(self.multimodal_pipeline_.report_)
            self.report_.text_columns_expanded = list(profile.text_columns)

        self.datetime_source_columns_ = [
            column for column in profile.datetime_columns if column in work.columns
        ]
        work = self._fit_expand_datetime(work)

        self.text_source_columns_ = []
        generated = set(work.columns) - set(self.raw_feature_columns_)
        self.numeric_columns_ = [
            column for column in profile.numeric_columns if column in work.columns
        ] + sorted(generated)
        self.categorical_columns_ = [
            column for column in profile.categorical_columns
            if column in work.columns and column not in self.numeric_columns_
        ]
        other_columns = [
            column for column in work.columns
            if column not in self.numeric_columns_ and column not in self.categorical_columns_
        ]
        self.categorical_columns_.extend(other_columns)

        self.report_.before_feature_count = int(work.shape[1])
        work = self._fit_mandatory_transforms(work)

        if self.config.clean == "auto" and self.config.feature_selection == "auto":
            feature_selection_drop = select_columns_to_drop(work, self.numeric_columns_)
            if feature_selection_drop:
                work = work.drop(columns=feature_selection_drop)
                self.numeric_columns_ = [
                    column for column in self.numeric_columns_ if column not in feature_selection_drop
                ]
                self.report_.dropped_near_duplicate_numeric_columns = feature_selection_drop

        feature_budget = self._resolve_model_feature_budget(len(work), work.shape[1])
        self.report_.model_feature_budget = feature_budget
        if self.config.feature_selection == "auto" and work.shape[1] > feature_budget:
            selected, scores = select_top_features(
                work, y,
                task_is_classification=profile.task_type.is_classification,
                categorical_columns=self.categorical_columns_,
                max_features=feature_budget,
                random_state=getattr(self.feature_space_config, "random_state", 42),
            )
            dropped = [column for column in work.columns if column not in set(selected)]
            work = work.loc[:, selected]
            self.numeric_columns_ = [column for column in self.numeric_columns_ if column in selected]
            self.categorical_columns_ = [column for column in self.categorical_columns_ if column in selected]
            self.report_.dropped_by_feature_budget = dropped
            self.report_.feature_selection_scores = scores

        if work.shape[1] == 0:
            raise DataValidationError("no usable feature columns remain after cleaning")
        self.report_.after_missing_values = int(work.isna().sum().sum())
        self.report_.after_feature_count = int(work.shape[1])
        self.final_feature_columns_ = list(work.columns)
        self._fitted = True
        return work

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self._fitted:
            raise RuntimeError("SmartCleaningPipeline must be fitted before transform()")
        work = self._validate_and_prepare_raw(df, fitting=False)
        work = work.drop(columns=self.dropped_source_columns_, errors="ignore")
        if self.multimodal_pipeline_ is not None:
            active_columns = self.multimodal_pipeline_.source_columns
            multimodal = self.multimodal_pipeline_.transform(work[active_columns])
            work = pd.concat([work.drop(columns=active_columns), multimodal], axis=1)
        work = self._transform_expand_datetime(work)
        if self.data_science_config.add_missing_indicators:
            work = self.missing_indicator_.transform(work)
        if self.numeric_winsorizer_.fraction > 0:
            work = self.numeric_winsorizer_.transform(work)
        work = self.numeric_imputer_.transform(work)
        work = self.categorical_imputer_.transform(work)
        work = self.rare_category_grouper_.transform(work)
        work = self.categorical_encoder_.transform(work)
        work = self.numeric_transformer_.transform(work)

        if self.scaler_ is not None and self.numeric_columns_:
            work[self.numeric_columns_] = self.scaler_.transform(work[self.numeric_columns_])

        missing_final = [column for column in self.final_feature_columns_ if column not in work.columns]
        if missing_final:
            raise DataValidationError(
                f"internal cleaning error: transformed features are missing {missing_final}"
            )
        return work.loc[:, self.final_feature_columns_]

    def _fit_mandatory_transforms(self, work: pd.DataFrame) -> pd.DataFrame:
        if self.data_science_config.add_missing_indicators:
            self.missing_indicator_.fit(work, self.numeric_columns_ + self.categorical_columns_)
            work = self.missing_indicator_.transform(work)
            self.numeric_columns_.extend(self.missing_indicator_.feature_names_)
            self.report_.missing_indicator_columns = list(self.missing_indicator_.feature_names_)

        if self.numeric_winsorizer_.fraction > 0:
            self.numeric_winsorizer_.fit(work, [c for c in self.numeric_columns_ if c in work])
            work = self.numeric_winsorizer_.transform(work)
            self.report_.winsorized_columns = dict(self.numeric_winsorizer_.clipped_counts_)

        self.numeric_imputer_.fit(work, self.numeric_columns_)
        work = self.numeric_imputer_.transform(work)
        self.report_.numeric_imputation = self.numeric_imputer_.strategy
        self.report_.imputed_numeric_columns = list(self.numeric_imputer_.columns_)

        self.categorical_imputer_.fit(work, self.categorical_columns_)
        work = self.categorical_imputer_.transform(work)
        self.rare_category_grouper_.fit(work, self.categorical_columns_)
        work = self.rare_category_grouper_.transform(work)
        self.report_.rare_categories_grouped = dict(self.rare_category_grouper_.grouped_counts_)
        self.categorical_encoder_.fit(work, self.categorical_columns_)
        work = self.categorical_encoder_.transform(work)

        self.numeric_transformer_.fit(work, self.numeric_columns_)
        work = self.numeric_transformer_.transform(work)
        self.report_.log_transformed_columns = list(self.numeric_transformer_.log_columns_)
        self.report_.power_transformed_columns = list(self.numeric_transformer_.power_columns_)
        if self.scaler_ is not None and self.numeric_columns_:
            work[self.numeric_columns_] = self.scaler_.fit_transform(work[self.numeric_columns_])
        return work

    def _resolve_model_feature_budget(self, n_rows: int, n_features: int) -> int:
        configured = self.data_science_config.max_model_features
        if configured != "auto":
            return min(n_features, int(configured))
        adaptive = int(max(128, min(2048, 48 * np.sqrt(max(n_rows, 1)))))
        return min(n_features, adaptive)

    def _validate_and_prepare_raw(self, df: pd.DataFrame, *, fitting: bool) -> pd.DataFrame:
        if not isinstance(df, pd.DataFrame):
            raise DataValidationError(f"features must be a pandas DataFrame, got {type(df).__name__}")
        if not self.raw_feature_columns_:
            raise RuntimeError("raw feature schema is not initialized")

        incoming = list(df.columns)
        missing = [column for column in self.raw_feature_columns_ if column not in incoming]
        unexpected = [column for column in incoming if column not in self.raw_feature_columns_]
        if self.config.schema_policy == "strict" and (missing or unexpected):
            parts = []
            if missing:
                parts.append(f"missing required columns: {missing}")
            if unexpected:
                parts.append(f"unexpected columns: {unexpected}")
            raise DataValidationError("prediction schema mismatch; " + "; ".join(parts))

        work = df.copy()
        if missing:
            for column in missing:
                work[column] = pd.NA
        work = work.loc[:, self.raw_feature_columns_]

        for column in self.raw_numeric_columns_:
            if column not in work:
                continue
            if self.config.schema_policy == "strict" and not pd.api.types.is_numeric_dtype(work[column]):
                non_missing = work[column].dropna()
                parsed = pd.to_numeric(non_missing, errors="coerce")
                if len(non_missing) and parsed.isna().any():
                    raise DataValidationError(
                        f"dtype mismatch for {column!r}: expected numeric, got {work[column].dtype}"
                    )
            work[column] = pd.to_numeric(work[column], errors="coerce")

        for column in self.raw_datetime_columns_:
            if column not in work:
                continue
            source_non_missing = work[column].notna()
            parsed = pd.to_datetime(work[column], errors="coerce", format="mixed")
            if self.config.schema_policy == "strict" and source_non_missing.any():
                failed = source_non_missing & parsed.isna()
                if failed.any():
                    examples = work.loc[failed, column].astype(str).head(3).tolist()
                    raise DataValidationError(
                        f"dtype mismatch for {column!r}: values are not parseable datetimes; examples={examples}"
                    )
            work[column] = parsed

        for column in set(self.raw_categorical_columns_ + self.raw_text_columns_):
            if column in work:
                work[column] = work[column].astype("string")
        return work

    def _resolve_columns_to_drop(self, profile: DatasetProfile) -> list[str]:
        leakage_columns = list(profile.potential_leakage_columns)
        if leakage_columns and self.config.leakage_policy == "error":
            details = {column: profile.leakage_reasons.get(column, []) for column in leakage_columns}
            raise DataValidationError(f"potential target leakage detected: {details}")
        if self.config.leakage_policy == "drop":
            self.report_.dropped_leakage_columns = leakage_columns
        elif self.config.leakage_policy == "warn":
            self.report_.warned_leakage_columns = leakage_columns
            if leakage_columns:
                self.report_.notes.append(
                    f"potential leakage retained by leakage_policy='warn': {leakage_columns}"
                )

        if self.config.clean == "none":
            return sorted(set(leakage_columns if self.config.leakage_policy == "drop" else []))

        duplicate_extra = [
            column
            for group in profile.duplicate_column_groups
            for column in group[1:]
        ]
        self.report_.dropped_constant_columns = list(profile.constant_columns)
        self.report_.dropped_id_columns = list(profile.id_like_columns)
        self.report_.dropped_duplicate_columns = duplicate_extra
        columns = (
            set(profile.constant_columns)
            | set(duplicate_extra)
            | set(leakage_columns if self.config.leakage_policy == "drop" else [])
        )
        if self.config.clean == "auto":
            columns |= set(profile.id_like_columns)
        return sorted(columns)

    def _fit_expand_datetime(self, df: pd.DataFrame) -> pd.DataFrame:
        result = df.copy()
        self.datetime_feature_names_.clear()
        for column in self.datetime_source_columns_:
            parsed = pd.to_datetime(result[column], errors="coerce", format="mixed")
            names = [
                f"{column}_missing",
                f"{column}_year",
                f"{column}_month",
                f"{column}_day",
                f"{column}_dayofweek",
                f"{column}_dayofyear",
                f"{column}_quarter",
                f"{column}_is_weekend",
                f"{column}_days_since_epoch",
                f"{column}_month_sin",
                f"{column}_month_cos",
                f"{column}_dayofweek_sin",
                f"{column}_dayofweek_cos",
            ]
            if (parsed.dt.hour.fillna(0) != 0).any() or (parsed.dt.minute.fillna(0) != 0).any():
                names.extend([f"{column}_hour", f"{column}_hour_sin", f"{column}_hour_cos"])
            self.datetime_feature_names_[column] = names
            result = self._add_datetime_parts(result, column, parsed, names)
        self.report_.datetime_columns_expanded = list(self.datetime_source_columns_)
        return result

    def _transform_expand_datetime(self, df: pd.DataFrame) -> pd.DataFrame:
        result = df.copy()
        for column in self.datetime_source_columns_:
            parsed = pd.to_datetime(result[column], errors="coerce", format="mixed")
            result = self._add_datetime_parts(
                result,
                column,
                parsed,
                self.datetime_feature_names_[column],
            )
        return result

    @staticmethod
    def _add_datetime_parts(
        df: pd.DataFrame,
        column: str,
        parsed: pd.Series,
        names: list[str],
    ) -> pd.DataFrame:
        result = df.copy()
        month = parsed.dt.month.astype("float64")
        dayofweek = parsed.dt.dayofweek.astype("float64")
        hour = parsed.dt.hour.astype("float64")
        timestamp_ns = parsed.astype("int64", copy=False).astype("float64")
        timestamp_ns = timestamp_ns.where(parsed.notna(), np.nan)
        values = {
            f"{column}_missing": parsed.isna().astype("float32"),
            f"{column}_year": parsed.dt.year,
            f"{column}_month": month,
            f"{column}_day": parsed.dt.day,
            f"{column}_dayofweek": dayofweek,
            f"{column}_dayofyear": parsed.dt.dayofyear,
            f"{column}_quarter": parsed.dt.quarter,
            f"{column}_is_weekend": dayofweek.isin([5.0, 6.0]).astype("float32"),
            f"{column}_days_since_epoch": timestamp_ns / 86_400_000_000_000.0,
            f"{column}_month_sin": np.sin(2.0 * np.pi * (month - 1.0) / 12.0),
            f"{column}_month_cos": np.cos(2.0 * np.pi * (month - 1.0) / 12.0),
            f"{column}_dayofweek_sin": np.sin(2.0 * np.pi * dayofweek / 7.0),
            f"{column}_dayofweek_cos": np.cos(2.0 * np.pi * dayofweek / 7.0),
            f"{column}_hour": hour,
            f"{column}_hour_sin": np.sin(2.0 * np.pi * hour / 24.0),
            f"{column}_hour_cos": np.cos(2.0 * np.pi * hour / 24.0),
        }
        for name in names:
            result[name] = values[name].astype("float32")
        return result.drop(columns=[column])

    @staticmethod
    def _expand_text_columns(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
        result = df.copy()
        for column in columns:
            text = result[column].astype("string").fillna("")
            result[f"{column}_len"] = text.str.len().astype("float32")
            result[f"{column}_word_count"] = text.str.split().str.len().fillna(0).astype("float32")
            result = result.drop(columns=[column])
        return result
