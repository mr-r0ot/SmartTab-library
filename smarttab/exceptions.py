"""Exception hierarchy for SmartTab."""

from __future__ import annotations


class SmartTabError(Exception):
    """Base class for all SmartTab errors."""


class DataValidationError(SmartTabError):
    """Raised when input data fails validation (empty, wrong shape, bad target, ...)."""


class ConfigurationError(SmartTabError):
    """Raised when a fit() parameter is invalid or an unsupported combination is requested."""


class UnsupportedModelError(SmartTabError):
    """Raised when an unknown or not-yet-implemented model/strategy is requested."""


class ModelNotFittedError(SmartTabError):
    """Raised when predict/evaluate/report/save is called before fit() has produced a model."""
