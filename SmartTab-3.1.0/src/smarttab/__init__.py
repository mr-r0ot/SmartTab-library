"""SmartTab: bounded multimodal AutoML with CatBoost and LightGBM."""

__version__ = "3.1.0"

from smarttab.api import audit, fit, fit_audio, fit_folder, fit_images, fit_text, fit_videos, load
from smarttab.model import SmartTabModel
from smarttab.multimodal.config import FeatureSpaceConfig
from smarttab.datascience.config import DataScienceConfig
from smarttab.datascience.quality import DataQualityIssue, DataQualityReport

__all__ = [
    "audit",
    "fit",
    "fit_text",
    "fit_images",
    "fit_audio",
    "fit_videos",
    "fit_folder",
    "load",
    "SmartTabModel",
    "FeatureSpaceConfig",
    "DataScienceConfig",
    "DataQualityIssue",
    "DataQualityReport",
    "__version__",
]
