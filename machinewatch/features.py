"""Features computed from an engine's own past only, so the same code serves training, evaluation and the
live API, and no future cycle can leak into a prediction.

For each cycle: the current sensor values, and over the last WINDOW cycles their mean, standard deviation
and slope. The slope carries most of the signal: wear shows as a drift (temperatures rising, the HPC
static pressure s11 rising, the fan speed ratio s15 changing) long before any single reading is unusual.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# Sensors with information in FD001. The others are constant or nearly so (s1, s5, s6, s10, s16, s18, s19),
# and the operational settings barely vary at a single flight condition.
SENSORS = ["s2", "s3", "s4", "s7", "s8", "s9", "s11", "s12", "s13", "s14", "s15", "s17", "s20", "s21"]
WINDOW = 30


def _rolling_slope(values: pd.Series, window: int) -> pd.Series:
    """Least-squares slope over the trailing window (per cycle), defined from the second cycle on."""
    def slope(chunk: np.ndarray) -> float:
        x = np.arange(len(chunk), dtype=float)
        x -= x.mean()
        return float((x * (chunk - chunk.mean())).sum() / (x * x).sum())

    return values.rolling(window, min_periods=2).apply(slope, raw=True).fillna(0.0)


def engine_features(history: pd.DataFrame) -> pd.DataFrame:
    """One feature row per cycle of a single engine, ordered by cycle."""
    history = history.sort_values("cycle")
    out = {"cycle": history["cycle"].to_numpy(dtype=float)}
    for sensor in SENSORS:
        values = history[sensor].astype(float)
        out[sensor] = values.to_numpy()
        roll = values.rolling(WINDOW, min_periods=1)
        out[f"{sensor}_mean"] = roll.mean().to_numpy()
        out[f"{sensor}_std"] = roll.std().fillna(0.0).to_numpy()
        out[f"{sensor}_slope"] = _rolling_slope(values, WINDOW).to_numpy()
    return pd.DataFrame(out, index=history.index)


def fleet_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Features for every cycle of every engine."""
    return pd.concat([engine_features(group) for _, group in frame.groupby("unit", sort=True)]).loc[frame.index]


def last_cycle_features(frame: pd.DataFrame) -> pd.DataFrame:
    """The feature row of each engine's latest cycle, indexed by unit: what a prediction is made from."""
    rows = {unit: engine_features(group).iloc[-1] for unit, group in frame.groupby("unit", sort=True)}
    return pd.DataFrame.from_dict(rows, orient="index").rename_axis("unit")
