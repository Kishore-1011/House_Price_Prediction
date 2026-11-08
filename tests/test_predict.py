import json

import numpy as np
import pandas as pd
import pytest

from house_price.model import LOG_RIDGE
from house_price.predict import InvalidClientFile, predict_prices, validate_client_data
from house_price.storage import ModelIntegrityError, load_model, save_model


def client(**changes):
    row = {"area": 1800, "bedroom": 3, "bathroom": 2, "age": 10, "locality": "Hillcrest"}
    row.update(changes)
    return pd.DataFrame([row])


def test_adds_a_predicted_price_and_keeps_the_input(result):
    output = predict_prices(client(), model=result.models[LOG_RIDGE])
    assert output.loc[0, "predicted_price"] > 0
    assert output.loc[0, "area"] == 1800


def test_missing_column_is_named():
    with pytest.raises(InvalidClientFile, match="locality"):
        validate_client_data(client().drop(columns="locality"))


def test_empty_file_is_rejected():
    with pytest.raises(InvalidClientFile, match="no rows"):
        validate_client_data(client().iloc[0:0])


def test_text_in_a_number_column_names_the_spreadsheet_row():
    frame = pd.concat([client(), client(bedroom="three")], ignore_index=True)
    with pytest.raises(InvalidClientFile, match=r"'bedroom'.*row\(s\): 3"):
        validate_client_data(frame)


@pytest.mark.parametrize("area", [0, np.nan])
def test_area_must_be_present_and_positive(area):
    with pytest.raises(InvalidClientFile, match="Area"):
        validate_client_data(client(area=area))


def test_negative_values_are_rejected():
    with pytest.raises(InvalidClientFile, match="negative"):
        validate_client_data(client(age=-1))


def test_missing_optional_values_are_allowed(result):
    output = predict_prices(client(bedroom=np.nan, age=np.nan), model=result.models[LOG_RIDGE])
    assert output.loc[0, "predicted_price"] > 0


def test_unknown_locality_gives_a_warning():
    with pytest.warns(UserWarning, match="Atlantis"):
        validate_client_data(client(locality="Atlantis"))


@pytest.fixture
def saved(tmp_path, result):
    model_path, metadata_path = tmp_path / "model.joblib", tmp_path / "meta.json"
    save_model(result.models[LOG_RIDGE], LOG_RIDGE, result.test_scores[LOG_RIDGE], model_path, metadata_path)
    return model_path, metadata_path


def test_saved_model_loads_and_predicts(saved, result):
    model = load_model(*saved)
    row = client()
    assert model.predict(row)[0] == pytest.approx(result.models[LOG_RIDGE].predict(row)[0])


def test_changed_model_file_is_refused(saved):
    model_path, metadata_path = saved
    model_path.write_bytes(model_path.read_bytes() + b"tampered")
    with pytest.raises(ModelIntegrityError, match="checksum"):
        load_model(model_path, metadata_path)


def test_missing_model_says_how_to_create_one(tmp_path):
    with pytest.raises(ModelIntegrityError, match="house_price.train"):
        load_model(tmp_path / "none.joblib", tmp_path / "none.json")


def test_version_mismatch_warns(saved):
    model_path, metadata_path = saved
    metadata = json.loads(metadata_path.read_text())
    metadata["versions"]["scikit_learn"] = "0.0.1"
    metadata_path.write_text(json.dumps(metadata))
    with pytest.warns(UserWarning, match="Retrain"):
        load_model(model_path, metadata_path)
