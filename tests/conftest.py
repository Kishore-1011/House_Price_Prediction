import pytest

from house_price.generate_data import generate_housing_data
from house_price.train import train_models


@pytest.fixture(scope="session")
def data():
    return generate_housing_data(rows=600, seed=7)


@pytest.fixture(scope="session")
def result(data):
    return train_models(data, include_forest=False)
