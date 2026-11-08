"""Predict prices for a client's CSV file.

Run with: python -m house_price.predict --input data/client/client_file.csv
"""

import argparse
import sys
import warnings
from pathlib import Path

import pandas as pd

from house_price.config import AREA, CATEGORICAL, CLIENT_DATA, FEATURES, LOCALITY_RATES, OTHER_NUMERIC, PREDICTIONS
from house_price.storage import ModelIntegrityError, load_model


class InvalidClientFile(ValueError):
    """The client's file can't be used, and the message says why."""


def validate_client_data(data):
    """Return a clean copy of the client's rows, or raise an error that names the problem."""
    missing = [column for column in FEATURES if column not in data.columns]
    if missing:
        raise InvalidClientFile(f"Missing column(s): {', '.join(missing)}")
    if data.empty:
        raise InvalidClientFile("The file has no rows.")

    clean = data.copy()
    for column in [AREA, *OTHER_NUMERIC]:
        converted = pd.to_numeric(clean[column], errors="coerce")
        bad_rows = clean.index[converted.isna() & clean[column].notna()]
        if len(bad_rows) > 0:
            raise InvalidClientFile(f"Column '{column}' has non-numeric values in row(s): {_row_numbers(bad_rows)}")
        negative = clean.index[converted < 0]
        if len(negative) > 0:
            raise InvalidClientFile(f"Column '{column}' has negative values in row(s): {_row_numbers(negative)}")
        clean[column] = converted

    unusable_area = clean.index[clean[AREA].isna() | (clean[AREA] <= 0)]
    if len(unusable_area) > 0:
        raise InvalidClientFile(f"Area must be above zero in row(s): {_row_numbers(unusable_area)}")

    for column in CATEGORICAL:
        unknown = sorted(set(clean[column].dropna()) - set(LOCALITY_RATES))
        if unknown:
            warnings.warn(f"Unknown {column} ignored, so these rows are less accurate: {', '.join(map(str, unknown))}",
                          stacklevel=2)
    return clean


def _row_numbers(index):
    # Row numbers as they appear in a spreadsheet, where the header is row 1.
    return ", ".join(str(position + 2) for position in index)


def predict_prices(data, model=None):
    model = load_model() if model is None else model
    clean = validate_client_data(data)
    result = clean.copy()
    result["predicted_price"] = model.predict(clean[FEATURES]).round(-3).astype("int64")
    return result


def main(arguments=None):
    parser = argparse.ArgumentParser(description="Predict house prices for a CSV file.")
    parser.add_argument("--input", type=Path, default=CLIENT_DATA)
    parser.add_argument("--output", type=Path, default=PREDICTIONS)
    options = parser.parse_args(arguments)

    try:
        result = predict_prices(pd.read_csv(options.input))
    except (InvalidClientFile, ModelIntegrityError, FileNotFoundError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1

    options.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(options.output, index=False)
    print(f"Predicted {len(result)} price(s) and wrote {options.output.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
