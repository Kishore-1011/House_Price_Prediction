import pandas as pd
import pytest
from fastapi.testclient import TestClient

from house_price.api import app
from house_price.config import FEATURES
from house_price.storage import load_metadata, load_model

HOUSE = {"area": 1800, "bedroom": 3, "bathroom": 2, "age": 10, "locality": "Riverview"}


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as test_client:
        yield test_client


def post(client, **changes):
    return client.post("/predict", json={**HOUSE, **changes})


def test_health(client):
    body = client.get("/health").json()
    assert body["status"] == "ok"
    assert body["model"] == load_metadata()["model"]


def test_prediction_matches_the_saved_model(client):
    body = post(client).json()
    direct = load_model().predict(pd.DataFrame([HOUSE], columns=FEATURES))[0]
    assert body["predicted_price"] == round(direct, -3)


def test_response_has_a_range_and_says_the_data_is_made_up(client):
    body = post(client).json()
    assert body["low"] < body["predicted_price"] < body["high"]
    assert "made-up" in body["note"]
    assert body["warnings"] == []


def test_optional_values_can_be_blank(client):
    response = client.post("/predict", json={"area": 1800, "locality": "Riverview", "bedroom": None})
    assert response.status_code == 200
    assert response.json()["predicted_price"] > 0


def test_outside_the_training_range_warns(client):
    warnings = post(client, area=50000).json()["warnings"]
    assert len(warnings) == 1 and "area" in warnings[0]


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("area", 0),
        ("area", -100),
        ("area", 200000),
        ("area", "big"),
        ("bedroom", -1),
        ("bathroom", 99),
        ("age", 500),
        ("locality", "Atlantis"),
    ],
)
def test_bad_values_are_rejected_with_a_message(client, field, value):
    response = post(client, **{field: value})
    assert response.status_code == 422
    assert list(response.json()["errors"]) == [field]


def test_missing_required_field_is_named(client):
    response = client.post("/predict", json={"area": 1800})
    assert response.status_code == 422
    assert "locality" in response.json()["errors"]


def test_unexpected_fields_are_rejected(client):
    assert post(client, hacker=True).status_code == 422


def test_non_json_body_is_rejected(client):
    response = client.post("/predict", content="not json", headers={"Content-Type": "application/json"})
    assert response.status_code == 422
    assert "form" in response.json()["errors"]


def test_infinite_numbers_are_rejected(client):
    response = client.post("/predict", content='{"area": Infinity, "locality": "Riverview"}',
                           headers={"Content-Type": "application/json"})
    assert response.status_code == 422


def test_serves_the_page(client):
    response = client.get("/")
    assert response.status_code == 200
    assert "House Price Estimator" in response.text


def test_page_is_told_to_use_the_server(client):
    assert '"/predict"' in client.get("/config.js").text


def test_api_reference_is_available(client):
    assert client.get("/docs").status_code == 200
    assert "/predict" in client.get("/openapi.json").json()["paths"]
