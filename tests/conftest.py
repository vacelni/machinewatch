import numpy as np
import pandas as pd
import pytest

from machinewatch.data import SENSORS, SETTINGS


@pytest.fixture
def engine_history():
    """A synthetic engine: sensors drift upward with wear, plus noise. No download needed."""
    rng = np.random.default_rng(0)
    cycles = np.arange(1, 81)
    frame = pd.DataFrame({"unit": 1, "cycle": cycles})
    for s in SETTINGS:
        frame[s] = 0.0
    for i, s in enumerate(SENSORS):
        frame[s] = 100 + i + 0.05 * cycles + rng.normal(0, 0.2, len(cycles))
    return frame
