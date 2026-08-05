"""Optional numeric scaling.

CatBoost and LightGBM are tree-based and scale-invariant, so
``scaling="auto"`` deliberately skips scaling entirely. This module only
matters when a user explicitly opts in via ``scaling="standard"``,
``"minmax"``, or ``"robust"``.
"""

from __future__ import annotations

from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler

from smarttab.exceptions import ConfigurationError

_SCALERS = {
    "standard": StandardScaler,
    "minmax": MinMaxScaler,
    "robust": RobustScaler,
}


def resolve_scaler(scaling: str):
    """Return a fitted-later sklearn scaler instance, or None if scaling should be skipped."""
    if scaling in ("auto", "none", None):
        return None
    scaler_cls = _SCALERS.get(scaling)
    if scaler_cls is None:
        raise ConfigurationError(f"scaling must be one of 'auto', 'none', {list(_SCALERS)}, got {scaling!r}")
    return scaler_cls()
