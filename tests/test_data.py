import pandas as pd

from house_price.config import FEATURES, LOCALITY_RATES, TARGET
from house_price.generate_data import generate_housing_data


def test_same_seed_gives_identical_data():
    pd.testing.assert_frame_equal(generate_housing_data(200, seed=1), generate_housing_data(200, seed=1))


def test_different_seed_gives_different_data():
    assert not generate_housing_data(200, seed=1).equals(generate_housing_data(200, seed=2))


def test_has_expected_columns(data):
    assert list(data.columns) == [*FEATURES, TARGET]


def test_values_are_plausible(data):
    assert data["area"].between(600, 6000).all()
    assert data["bedroom"].dropna().between(1, 6).all()
    assert data["age"].dropna().between(0, 40).all()
    assert (data["price"] > 0).all()
    assert set(data["locality"]) <= set(LOCALITY_RATES)


def test_includes_some_missing_values_to_handle(data):
    assert data["bedroom"].isna().any()
    assert data["age"].isna().any()
    assert not data["area"].isna().any()
