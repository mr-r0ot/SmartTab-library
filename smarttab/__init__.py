"""SmartTab — a smart decision engine for tabular machine learning.

>>> import smarttab
>>> model = smarttab.fit(df, target="label")
>>> model.predict(new_df)
"""

__version__ = "0.1.0"

from smarttab.api import fit, load
from smarttab.model import SmartTabModel

__all__ = ["fit", "load", "SmartTabModel", "__version__"]
