import numpy as np
import pandas as pd

from machinewatch.features import WINDOW, engine_features, fleet_features, last_cycle_features


def test_features_at_a_cycle_ignore_the_future(engine_history):
    full = engine_features(engine_history)
    for cut in (5, 31, 60):
        truncated = engine_features(engine_history[engine_history.cycle <= cut])
        pd.testing.assert_series_equal(full.iloc[cut - 1], truncated.iloc[-1], check_names=False)


def test_the_last_window_is_all_the_api_needs(engine_history):
    # The API reads back only the last WINDOW cycles; the prediction must equal the offline one.
    offline = engine_features(engine_history).iloc[-1]
    live = engine_features(engine_history.tail(WINDOW)).iloc[-1].drop("cycle")
    pd.testing.assert_series_equal(offline.drop("cycle"), live, check_names=False)


def test_slope_picks_up_the_drift(engine_history):
    last = engine_features(engine_history).iloc[-1]
    assert np.isclose(last["s11_slope"], 0.05, atol=0.02)


def test_fleet_and_last_cycle_features_line_up(engine_history):
    two = pd.concat([engine_history, engine_history.assign(unit=2)], ignore_index=True)
    assert len(fleet_features(two)) == len(two)
    assert list(last_cycle_features(two).index) == [1, 2]
