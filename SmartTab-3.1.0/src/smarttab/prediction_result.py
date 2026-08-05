"""NumPy-compatible prediction result with simple serialization helpers."""

from __future__ import annotations

import json as _json

import numpy as np
import pandas as pd

from smarttab.exceptions import SmartTabError


def _predictions_to_frame(prediction: np.ndarray) -> pd.DataFrame:
    values = np.asarray(prediction)
    if values.ndim <= 1:
        return pd.DataFrame({"prediction": values})
    return pd.DataFrame({f"output_{index}": values[:, index] for index in range(values.shape[1])})


class PredictionArray(np.ndarray):
    """A real ndarray with scalar convenience and CSV/JSON representations."""

    def __new__(cls, values, *, single: bool = False):
        obj = np.asarray(values).view(cls)
        obj._single = single
        return obj

    def __array_finalize__(self, obj) -> None:
        if obj is not None:
            self._single = getattr(obj, "_single", False)

    @property
    def single(self) -> bool:
        return bool(self._single)

    @property
    def prediction(self):
        return self[0] if self._single and len(self) == 1 else np.asarray(self)

    label = prediction

    @property
    def probability(self):
        raise SmartTabError("probabilities are returned explicitly by model.predict_proba(X)")

    @property
    def csv(self) -> str:
        return _predictions_to_frame(np.asarray(self)).to_csv(index=False)

    @property
    def json(self) -> str:
        records = _predictions_to_frame(np.asarray(self)).to_dict(orient="records")
        payload = records[0] if self._single else records
        return _json.dumps(payload, indent=2, default=str)
