"""Write the linear model's numbers to a small script the demo page can use."""

import json

from house_price.config import AREA, CATEGORICAL, OTHER_NUMERIC, WEB_MODEL_PATH


def model_to_dict(model, metrics, data, price_range):
    """Describe a fitted log-price Ridge model as plain numbers."""
    pipeline = model.regressor_
    prep = pipeline.named_steps["prep"].named_transformers_
    reg = pipeline.named_steps["reg"]

    area_steps = prep["area"].named_steps
    numeric_steps = prep["numeric"].named_steps
    encoder = prep["locality"].named_steps["encode"]

    numeric_count = len(OTHER_NUMERIC)
    coefficients = [float(value) for value in reg.coef_]
    return {
        "intercept": float(reg.intercept_),
        "area": {
            "mean": float(area_steps["scale"].mean_[0]),
            "scale": float(area_steps["scale"].scale_[0]),
            "coef": coefficients[0],
            "min": int(data[AREA].min()),
            "max": int(data[AREA].max()),
        },
        "numeric": {
            name: {
                "median": float(numeric_steps["impute"].statistics_[index]),
                "mean": float(numeric_steps["scale"].mean_[index]),
                "scale": float(numeric_steps["scale"].scale_[index]),
                "coef": coefficients[1 + index],
                "min": int(data[name].min()),
                "max": int(data[name].max()),
            }
            for index, name in enumerate(OTHER_NUMERIC)
        },
        "localities": {
            str(name): coefficients[1 + numeric_count + index]
            for index, name in enumerate(encoder.categories_[0])
        },
        "metrics": {"mae": metrics["mae"], "mape": metrics["mape"], "r2": metrics["r2"]},
        "priceRange": {"low": price_range["low"], "high": price_range["high"]},
    }


def export_web_model(model, metrics, data, price_range, path=WEB_MODEL_PATH):
    assert CATEGORICAL == ["locality"]
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(model_to_dict(model, metrics, data, price_range), indent=2)
    path.write_text(f"window.HOUSE_MODEL = {payload};\n", encoding="utf-8")
