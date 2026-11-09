"""The candidate models and how they are scored."""

import numpy as np
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, Ridge, RidgeCV
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, r2_score, root_mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

from house_price.config import AREA, CATEGORICAL, OTHER_NUMERIC, SEED

LOG_RIDGE = "ridge_log_price"
RIDGE_ALPHAS = np.logspace(-3, 3, 25)


def _numeric_steps():
    return [("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]


def build_preprocessor(log_area=False):
    """Fill missing values, scale numbers and one-hot encode the locality."""
    area_steps = _numeric_steps()
    if log_area:
        area_steps.insert(1, ("log", FunctionTransformer(np.log, feature_names_out="one-to-one")))
    return ColumnTransformer(
        [
            ("area", Pipeline(area_steps), [AREA]),
            ("numeric", Pipeline(_numeric_steps()), OTHER_NUMERIC),
            (
                "locality",
                Pipeline(
                    [
                        ("impute", SimpleImputer(strategy="most_frequent")),
                        ("encode", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
                    ]
                ),
                CATEGORICAL,
            ),
        ]
    )


def _regressor(preprocessor, estimator):
    return Pipeline([("prep", preprocessor), ("reg", estimator)])


def _on_log_price(regressor):
    return TransformedTargetRegressor(regressor=regressor, func=np.log, inverse_func=np.exp)


def build_candidates(include_forest=True):
    """Every model to compare. The baseline is the score the others have to beat."""
    candidates = {
        "baseline_median": _regressor(build_preprocessor(), DummyRegressor(strategy="median")),
        "linear_regression": _regressor(build_preprocessor(), LinearRegression()),
        "ridge": _regressor(build_preprocessor(), Ridge(alpha=1.0)),
        LOG_RIDGE: _on_log_price(_regressor(build_preprocessor(log_area=True), RidgeCV(alphas=RIDGE_ALPHAS))),
    }
    if include_forest:
        candidates["random_forest"] = _regressor(
            build_preprocessor(),
            RandomForestRegressor(n_estimators=200, min_samples_leaf=2, random_state=SEED, n_jobs=-1),
        )
        candidates["gradient_boosting"] = _on_log_price(
            _regressor(
                build_preprocessor(),
                HistGradientBoostingRegressor(learning_rate=0.05, max_iter=300, random_state=SEED),
            )
        )
    return candidates


def score_predictions(actual, predicted):
    # Rounded so runs compare equal: multi-threaded models can differ in the last few decimals.
    return {
        "mae": round(float(mean_absolute_error(actual, predicted)), 2),
        "rmse": round(float(root_mean_squared_error(actual, predicted)), 2),
        "r2": round(float(r2_score(actual, predicted)), 6),
        "mape": round(float(mean_absolute_percentage_error(actual, predicted)), 6),
    }
