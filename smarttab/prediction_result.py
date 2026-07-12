"""Result wrappers returned by ``SmartTabModel.predict()``.

Both wrappers behave exactly like what ``predict()`` has always returned —
a plain array, or a ``(labels, confidence)`` tuple when
``multi_threshold_ensemble=True`` — so no existing code (tuple unpacking,
indexing, iteration, arithmetic, ``isinstance(x, np.ndarray)``) breaks. They
additionally expose named attributes (``.prediction`` / ``.label`` /
``.probability``) and one-line ``.csv`` / ``.json`` export, so a single
prediction (from a dict, a CSV row, whatever came in) doesn't need to be
unpacked from an array just to read it.
"""

from __future__ import annotations

import json as _json

import numpy as np
import pandas as pd

from smarttab.exceptions import SmartTabError


def _predictions_to_frame(prediction: np.ndarray, probability: np.ndarray | None = None) -> pd.DataFrame:
    pred_arr = np.asarray(prediction)
    if pred_arr.ndim <= 1:
        data = {"prediction": pred_arr}
        if probability is not None:
            data["probability"] = np.asarray(probability)
        return pd.DataFrame(data)

    # multi-label / multi-output: one column per target, plus one confidence column per
    # target when a ladder-derived probability matrix is available.
    n_cols = pred_arr.shape[1]
    data = {f"label_{i}": pred_arr[:, i] for i in range(n_cols)}
    if probability is not None:
        prob_arr = np.asarray(probability)
        for i in range(n_cols):
            data[f"label_{i}_confidence"] = prob_arr[:, i]
    return pd.DataFrame(data)


class PredictionArray(np.ndarray):
    """What ``predict()`` returns when ``multi_threshold_ensemble=False`` (the default).

    A real ``np.ndarray`` — indexing, slicing, iteration, arithmetic, and
    ``isinstance(x, np.ndarray)`` all work exactly as before — plus
    ``.prediction`` / ``.label`` aliases and ``.csv`` / ``.json`` export.
    When the input to ``predict()`` was a single sample (a dict, or a 1-D
    array), ``.prediction`` / ``.label`` return a bare scalar instead of a
    length-1 array.
    """

    def __new__(cls, values, *, single: bool = False):
        obj = np.asarray(values).view(cls)
        obj._single = single
        return obj

    def __array_finalize__(self, obj) -> None:
        if obj is None:
            return
        self._single = getattr(obj, "_single", False)

    @property
    def prediction(self):
        return self[0] if self._single and len(self) == 1 else np.asarray(self)

    label = prediction

    @property
    def probability(self):
        raise SmartTabError(
            "probability/confidence is only available when the model was fit with "
            "multi_threshold_ensemble=True — see documents.md section 13."
        )

    @property
    def csv(self) -> str:
        return _predictions_to_frame(np.asarray(self)).to_csv(index=False)

    @property
    def json(self) -> str:
        records = _predictions_to_frame(np.asarray(self)).to_dict(orient="records")
        payload = records[0] if self._single else records
        return _json.dumps(payload, indent=2, default=str)


class PredictionWithConfidence(tuple):
    """What ``predict()`` returns when ``multi_threshold_ensemble=True``.

    A real 2-tuple — ``labels, confidence = model.predict(X)`` keeps working
    exactly as before — plus ``.prediction`` / ``.label`` / ``.probability``
    attributes and ``.csv`` / ``.json`` export. When the input was a single
    sample, the attributes return bare scalars instead of length-1 arrays.
    """

    def __new__(cls, prediction, probability, *, single: bool = False):
        return super().__new__(cls, (prediction, probability))

    def __init__(self, prediction, probability, *, single: bool = False) -> None:
        self._single = single

    @property
    def prediction(self):
        pred = self[0]
        return pred[0] if self._single and len(pred) == 1 else pred

    label = prediction

    @property
    def probability(self):
        prob = self[1]
        return prob[0] if self._single and len(prob) == 1 else prob

    @property
    def csv(self) -> str:
        return _predictions_to_frame(self[0], self[1]).to_csv(index=False)

    @property
    def json(self) -> str:
        records = _predictions_to_frame(self[0], self[1]).to_dict(orient="records")
        payload = records[0] if self._single else records
        return _json.dumps(payload, indent=2, default=str)
