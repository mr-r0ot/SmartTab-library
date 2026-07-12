"""Per-model Optuna search spaces.

Only hyperparameters that meaningfully move CatBoost/LightGBM/XGBoost
accuracy are tuned. Tree/iteration counts are deliberately excluded — the
optimizer fits each trial with early stopping instead, which implicitly
picks the right number of trees far more efficiently than searching over
it. XGBoost's space is only ever used inside a voting/stacking ensemble
(``training/ensemble.py``), since XGBoost is never a standalone selectable
model.
"""

from __future__ import annotations

from smarttab.hardware.resource_planner import ResourcePlan

LOW_MEMORY_BUDGET_MB = 2000


def catboost_space(trial, resource_plan: ResourcePlan) -> dict:
    max_depth = 8 if resource_plan.memory_budget_mb < LOW_MEMORY_BUDGET_MB else 10
    return {
        "depth": trial.suggest_int("depth", 4, max_depth),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1.0, 10.0, log=True),
        "random_strength": trial.suggest_float("random_strength", 1e-3, 10.0, log=True),
        "bagging_temperature": trial.suggest_float("bagging_temperature", 0.0, 1.0),
        "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 1, 50),
    }


def lightgbm_space(trial, resource_plan: ResourcePlan) -> dict:
    max_leaves = 128 if resource_plan.memory_budget_mb < LOW_MEMORY_BUDGET_MB else 256
    return {
        "num_leaves": trial.suggest_int("num_leaves", 16, max_leaves, log=True),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
    }


def xgboost_space(trial, resource_plan: ResourcePlan) -> dict:
    max_depth = 8 if resource_plan.memory_budget_mb < LOW_MEMORY_BUDGET_MB else 10
    return {
        "max_depth": trial.suggest_int("max_depth", 3, max_depth),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 20),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
    }


SEARCH_SPACES = {"catboost": catboost_space, "lightgbm": lightgbm_space, "xgboost": xgboost_space}

DEFAULT_PARAMS = {
    "catboost": {"depth": 6, "learning_rate": 0.05, "l2_leaf_reg": 3.0},
    "lightgbm": {"num_leaves": 31, "learning_rate": 0.05, "min_child_samples": 20},
    "xgboost": {"max_depth": 6, "learning_rate": 0.05, "reg_lambda": 1.0},
}


def default_params(model_name: str) -> dict:
    """Reasonable fixed hyperparameters used when optimize=False."""
    return dict(DEFAULT_PARAMS[model_name])
