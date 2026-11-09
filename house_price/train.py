"""Compare the candidate models, keep the best one, and write the reports.

Models are chosen by cross-validation on the training houses only. The held-back test
houses are used once, at the end, so the reported error is for houses the model never saw.

Run with: python -m house_price.train
"""

import json
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, cross_val_predict, cross_val_score, train_test_split

from house_price.config import (
    AREA,
    CV_FOLDS,
    FEATURES,
    METRICS_PATH,
    OTHER_NUMERIC,
    RAW_DATA,
    SEED,
    TARGET,
    TEST_SIZE,
)
from house_price.export_web import export_web_model
from house_price.generate_data import generate_housing_data, noise_free_prices
from house_price.model import LOG_RIDGE, build_candidates, score_predictions
from house_price.reports import plot_model_comparison, plot_predicted_vs_actual, plot_price_by_area
from house_price.storage import save_model

RANGE_SHARE = 0.8


@dataclass
class TrainingResult:
    best_name: str
    models: dict
    cv_mae: dict
    test_scores: dict
    price_ranges: dict
    X_test: pd.DataFrame
    y_test: pd.Series


def measure_price_range(candidate, fitted, X_train, y_train, X_test, y_test, folds):
    """How far above and below the prediction real prices usually fall.

    Measured on training houses the model hadn't seen during each fold, then checked on the test houses.
    """
    unseen = cross_val_predict(candidate, X_train, y_train, cv=folds)
    tail = (1 - RANGE_SHARE) / 2
    low, high = np.quantile(y_train / unseen, [tail, 1 - tail])

    predicted = fitted.predict(X_test)
    inside = (y_test >= predicted * low) & (y_test <= predicted * high)
    return {"low": round(float(low), 4), "high": round(float(high), 4), "coverage_on_test": round(float(inside.mean()), 4)}


def train_models(data, include_forest=True):
    """Fit every candidate and score it. Saves nothing, so it's safe to call from a notebook."""
    X_train, X_test, y_train, y_test = train_test_split(
        data[FEATURES], data[TARGET], test_size=TEST_SIZE, random_state=SEED
    )
    folds = KFold(n_splits=CV_FOLDS, shuffle=True, random_state=SEED)

    models, cv_mae, test_scores = {}, {}, {}
    for name, model in build_candidates(include_forest).items():
        scores = cross_val_score(model, X_train, y_train, cv=folds, scoring="neg_mean_absolute_error")
        cv_mae[name] = round(float(-scores.mean()), 2)
        models[name] = model.fit(X_train, y_train)
        test_scores[name] = score_predictions(y_test, model.predict(X_test))

    best_name = min(cv_mae, key=cv_mae.get)
    fresh = build_candidates(include_forest)
    price_ranges = {
        name: measure_price_range(fresh[name], models[name], X_train, y_train, X_test, y_test, folds)
        for name in {best_name, LOG_RIDGE}
    }
    return TrainingResult(best_name, models, cv_mae, test_scores, price_ranges, X_test, y_test)


def input_ranges(data):
    return {name: {"min": int(data[name].min()), "max": int(data[name].max())} for name in [AREA, *OTHER_NUMERIC]}


def best_possible_score(data, result):
    """The score of a model that knew the exact formula. Only the random noise would remain."""
    expected = generate_housing_data(len(data), SEED)
    if not np.array_equal(expected[TARGET].to_numpy(), data[TARGET].to_numpy()):
        return None
    noise_free = pd.Series(noise_free_prices(len(data), SEED), index=data.index)
    return score_predictions(result.y_test, noise_free.loc[result.y_test.index])


def save_outputs(result, data):
    best = result.models[result.best_name]
    save_model(
        best,
        result.best_name,
        result.test_scores[result.best_name],
        extras={"price_range": result.price_ranges[result.best_name], "input_ranges": input_ranges(data)},
    )
    export_web_model(result.models[LOG_RIDGE], result.test_scores[LOG_RIDGE], data, result.price_ranges[LOG_RIDGE])

    METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "data": "Made-up data from house_price/generate_data.py, not real sales.",
        "rows": len(data),
        "test_rows": len(result.y_test),
        "best_model": result.best_name,
        "cross_validation_mae": result.cv_mae,
        "test": result.test_scores,
        "best_possible_on_test": best_possible_score(data, result),
        "price_range": result.price_ranges[result.best_name],
    }
    METRICS_PATH.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    plot_price_by_area(data)
    plot_model_comparison(result.test_scores, result.best_name)
    plot_predicted_vs_actual(
        result.y_test,
        best.predict(result.X_test),
        f"Unseen houses: {result.best_name.replace('_', ' ')}",
    )


def main():
    data = pd.read_csv(RAW_DATA)
    result = train_models(data)
    save_outputs(result, data)

    print(f"{'model':<20}{'CV error':>14}{'test error':>14}{'test R2':>10}")
    for name in sorted(result.cv_mae, key=result.cv_mae.get):
        score = result.test_scores[name]
        print(f"{name:<20}{result.cv_mae[name]:>14,.0f}{score['mae']:>14,.0f}{score['r2']:>10.3f}")
    print(f"\nChosen model: {result.best_name}")


if __name__ == "__main__":
    main()
