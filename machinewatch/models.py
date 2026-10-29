"""RUL models, simplest first. Each maps a feature frame (features.py) to predicted cycles remaining."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .data import RUL_CAP
from .features import SENSORS

# Maintenance thresholds in cycles. A fleet manager acts on these, not on the raw number.
CRITICAL_BELOW = 30
WATCH_BELOW = 80


def health_status(rul: float) -> str:
    if rul < CRITICAL_BELOW:
        return "critical"
    if rul < WATCH_BELOW:
        return "watch"
    return "healthy"


@dataclass
class CurrentReadingsRidge:
    """Floor: a linear model on the current sensor values only, with no history."""

    name: str = "Ridge on current readings"
    model: object = field(default_factory=lambda: make_pipeline(StandardScaler(), Ridge(alpha=1.0)))

    def columns(self, features: pd.DataFrame) -> list[str]:
        return SENSORS

    def fit(self, features: pd.DataFrame, rul) -> "CurrentReadingsRidge":
        self.model.fit(features[self.columns(features)], rul)
        return self

    def predict(self, features: pd.DataFrame) -> np.ndarray:
        return np.clip(self.model.predict(features[self.columns(features)]), 0, RUL_CAP)


@dataclass
class WindowBoosting(CurrentReadingsRidge):
    """Gradient-boosted trees on readings plus their 30-cycle mean, spread and trend."""

    name: str = "Gradient boosting on 30-cycle windows"
    model: object = field(
        default_factory=lambda: HistGradientBoostingRegressor(
            max_iter=400, learning_rate=0.05, max_leaf_nodes=31, min_samples_leaf=40, l2_regularization=1.0, random_state=0
        )
    )

    def columns(self, features: pd.DataFrame) -> list[str]:
        return [c for c in features.columns if c != "cycle"]
