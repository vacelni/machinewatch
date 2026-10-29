"""Remaining-useful-life metrics used for C-MAPSS.

The NASA scoring function (Saxena et al., 2008) penalises late predictions more than early ones: saying an
engine has 20 cycles left when it has 10 risks a failure in service, while the opposite only schedules
maintenance a little early.
"""

from __future__ import annotations

import numpy as np


def rmse(y_true, y_pred) -> float:
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    return float(np.sqrt(np.mean((y_pred - y_true) ** 2)))


def nasa_score(y_true, y_pred) -> float:
    d = np.asarray(y_pred, float) - np.asarray(y_true, float)
    return float(np.sum(np.where(d < 0, np.exp(-d / 13.0) - 1, np.exp(d / 10.0) - 1)))


def evaluate(y_true, y_pred) -> dict[str, float]:
    d = np.asarray(y_pred, float) - np.asarray(y_true, float)
    return {
        "rmse": rmse(y_true, y_pred),
        "nasa_score": nasa_score(y_true, y_pred),
        "mae": float(np.mean(np.abs(d))),
        # Share of engines predicted to last longer than they really do: the risky direction.
        "late_share": float(np.mean(d > 0)),
    }
