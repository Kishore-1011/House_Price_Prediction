import numpy as np
import pandas as pd

from house_price.config import FEATURES
from house_price.model import LOG_RIDGE


def test_every_real_model_beats_the_baseline(result):
    baseline = result.test_scores["baseline_median"]["mae"]
    for name, score in result.test_scores.items():
        if name != "baseline_median":
            assert score["mae"] < baseline, name


def test_chosen_model_has_the_lowest_cross_validation_error(result):
    assert result.cv_mae[result.best_name] == min(result.cv_mae.values())


def test_test_houses_were_kept_out_of_training(data, result):
    assert len(result.X_test) == round(len(data) * 0.2)
    assert result.X_test.index.is_unique


def test_handles_missing_values(result):
    row = pd.DataFrame([{"area": 1500, "bedroom": np.nan, "bathroom": 2, "age": np.nan, "locality": "Riverview"}])
    assert np.isfinite(result.models[result.best_name].predict(row[FEATURES])).all()


def test_handles_an_unknown_locality(result):
    row = pd.DataFrame([{"area": 1500, "bedroom": 3, "bathroom": 2, "age": 5, "locality": "Nowhere"}])
    assert np.isfinite(result.models[result.best_name].predict(row[FEATURES])).all()


def test_bigger_houses_cost_more(result):
    rows = pd.DataFrame(
        [{"area": area, "bedroom": 3, "bathroom": 2, "age": 5, "locality": "Riverview"} for area in (1000, 2000, 4000)]
    )
    predictions = result.models[LOG_RIDGE].predict(rows[FEATURES])
    assert predictions[0] < predictions[1] < predictions[2]


def test_price_range_brackets_the_prediction(result):
    assert result.price_range["low"] < 1 < result.price_range["high"]


def test_price_range_holds_up_on_unseen_houses(result):
    assert 0.7 <= result.price_range["coverage_on_test"] <= 0.92
