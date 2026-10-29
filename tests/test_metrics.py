import pytest

from machinewatch.metrics import evaluate, nasa_score, rmse


def test_perfect_prediction_scores_zero():
    assert rmse([10, 20], [10, 20]) == 0
    assert nasa_score([10, 20], [10, 20]) == 0


def test_late_predictions_cost_more_than_early_ones():
    # Predicting 10 cycles too many (engine fails sooner than planned) is worse than 10 too few.
    assert nasa_score([50], [60]) > nasa_score([50], [40])


def test_late_share_counts_optimistic_predictions():
    assert evaluate([10, 10, 10, 10], [12, 8, 15, 10])["late_share"] == pytest.approx(0.5)
