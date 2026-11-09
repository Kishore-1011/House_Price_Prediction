"""FastAPI service for the house price model.

Run from the repository root with: python -m house_price.api
Then open http://127.0.0.1:8000, or http://127.0.0.1:8000/docs for the API reference.
"""

import argparse
from contextlib import asynccontextmanager
from typing import Literal

import pandas as pd
import uvicorn
from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field

from house_price.config import AREA, FEATURES, LOCALITY_RATES, OTHER_NUMERIC, ROOT
from house_price.predict import predict_prices
from house_price.storage import load_metadata, load_model

MAX_AREA = 100_000
LABELS = {"area": "area", "bedroom": "bedrooms", "bathroom": "bathrooms", "age": "age"}
FIELD_ERRORS = {
    "area": f"Enter an area above 0 and up to {MAX_AREA:,} sq ft.",
    "bedroom": "Enter bedrooms from 0 to 20, or leave it blank.",
    "bathroom": "Enter bathrooms from 0 to 20, or leave it blank.",
    "age": "Enter an age from 0 to 200 years, or leave it blank.",
    "locality": f"Choose one of: {', '.join(LOCALITY_RATES)}.",
}
DATA_NOTE = "This model was trained on made-up data, so the price is not a real valuation."


class House(BaseModel):
    model_config = ConfigDict(extra="forbid")

    area: float = Field(gt=0, le=MAX_AREA, allow_inf_nan=False, examples=[1800])
    bedroom: float | None = Field(default=None, ge=0, le=20, allow_inf_nan=False, examples=[3])
    bathroom: float | None = Field(default=None, ge=0, le=20, allow_inf_nan=False, examples=[2])
    age: float | None = Field(default=None, ge=0, le=200, allow_inf_nan=False, examples=[10])
    locality: Literal[tuple(LOCALITY_RATES)] = Field(examples=["Riverview"])


class Prediction(BaseModel):
    predicted_price: int
    low: int
    high: int
    range_share: str
    model: str
    warnings: list[str]
    note: str


@asynccontextmanager
async def lifespan(app):
    # load_model verifies the checksum first, so a changed file stops the service from starting.
    app.state.model = load_model()
    app.state.metadata = load_metadata()
    yield


app = FastAPI(title="House Price Prediction", lifespan=lifespan)


@app.exception_handler(RequestValidationError)
async def validation_error(request: Request, error: RequestValidationError):
    """Return one readable message per field for the form to show."""
    errors = {}
    for problem in error.errors():
        location = problem["loc"]
        field = location[1] if len(location) > 1 else None
        errors[field if field in FIELD_ERRORS else "form"] = FIELD_ERRORS.get(field, "Send the house details as JSON.")
    return JSONResponse(status_code=422, content={"errors": errors})


def range_warnings(house, ranges):
    outside = []
    for name in [AREA, *OTHER_NUMERIC]:
        value = getattr(house, name)
        if value is not None and not ranges[name]["min"] <= value <= ranges[name]["max"]:
            outside.append(LABELS[name])
    if not outside:
        return []
    return [f"The {', '.join(outside)} you entered is outside what the model learned from, so treat this as a rough guess."]


def round_price(value):
    return int(round(value, -3))


@app.get("/health")
def health(request: Request):
    return {"status": "ok", "model": request.app.state.metadata["model"]}


@app.post("/predict", response_model=Prediction)
def predict(house: House, request: Request):
    metadata = request.app.state.metadata
    frame = pd.DataFrame([house.model_dump()], columns=FEATURES)
    price = predict_prices(frame, request.app.state.model).loc[0, "predicted_price"]
    price_range = metadata["price_range"]
    return Prediction(
        predicted_price=int(price),
        low=round_price(price * price_range["low"]),
        high=round_price(price * price_range["high"]),
        range_share="About 8 in 10 similar houses",
        model=metadata["model"],
        warnings=range_warnings(house, metadata["input_ranges"]),
        note=DATA_NOTE,
    )


@app.get("/config.js", include_in_schema=False)
def page_config():
    """Tells the page to ask this server, instead of calculating in the browser."""
    return Response('window.HOUSE_API = "/predict";\n', media_type="text/javascript")


# Last, so the routes above take priority over the files.
app.mount("/", StaticFiles(directory=ROOT / "docs", html=True), name="site")


def main():
    parser = argparse.ArgumentParser(description="Run the house price API.")
    parser.add_argument("--port", type=int, default=8000)
    options = parser.parse_args()
    uvicorn.run(app, host="127.0.0.1", port=options.port)


if __name__ == "__main__":
    main()
