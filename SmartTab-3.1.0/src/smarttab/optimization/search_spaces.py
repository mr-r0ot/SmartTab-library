"""Compact, high-impact Optuna spaces for CatBoost, LightGBM, and conditional XGBoost."""

from __future__ import annotations

from smarttab.hardware.resource_planner import ResourcePlan

LOW_MEMORY_BUDGET_MB = 2000


def catboost_space(trial, resource_plan: ResourcePlan) -> dict:
    max_depth = 8 if resource_plan.memory_budget_mb < LOW_MEMORY_BUDGET_MB else 10
    return {
        "depth": trial.suggest_int("depth", 4, max_depth),
        "learning_rate": trial.suggest_float("learning_rate", 0.025, 0.18, log=True),
        "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1.0, 12.0, log=True),
        "random_strength": trial.suggest_float("random_strength", 0.01, 3.0, log=True),
        "bagging_temperature": trial.suggest_float("bagging_temperature", 0.0, 1.0),
    }


def lightgbm_space(trial, resource_plan: ResourcePlan) -> dict:
    max_leaves = 96 if resource_plan.memory_budget_mb < LOW_MEMORY_BUDGET_MB else 192
    return {
        "num_leaves": trial.suggest_int("num_leaves", 16, max_leaves, log=True),
        "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.18, log=True),
        "min_child_samples": trial.suggest_int("min_child_samples", 10, 100),
        "subsample": trial.suggest_float("subsample", 0.7, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.65, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-6, 5.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-6, 8.0, log=True),
    }


def xgboost_space(trial, resource_plan: ResourcePlan) -> dict:
    max_depth = 7 if resource_plan.memory_budget_mb < LOW_MEMORY_BUDGET_MB else 9
    return {
        "max_depth": trial.suggest_int("max_depth", 3, max_depth),
        "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.18, log=True),
        "min_child_weight": trial.suggest_float("min_child_weight", 1.0, 12.0, log=True),
        "subsample": trial.suggest_float("subsample", 0.7, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.65, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-6, 5.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-6, 8.0, log=True),
    }


SEARCH_SPACES = {
    "catboost": catboost_space,
    "lightgbm": lightgbm_space,
    "xgboost": xgboost_space,
}

DEFAULT_PARAMS = {
    "catboost": {
        "depth": 6,
        "learning_rate": 0.06,
        "l2_leaf_reg": 3.0,
        "random_strength": 0.5,
    },
    "lightgbm": {
        "num_leaves": 31,
        "learning_rate": 0.06,
        "min_child_samples": 20,
        "reg_lambda": 0.1,
    },
    "xgboost": {
        "max_depth": 6,
        "learning_rate": 0.06,
        "min_child_weight": 1.0,
        "reg_lambda": 1.0,
    },
}


def default_params(model_name: str) -> dict:
    return dict(DEFAULT_PARAMS[model_name])
