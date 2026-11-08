"""Save and load the trained model safely.

A joblib file can run code when it is loaded, so the checksum is verified first and only
files produced by this project's own training run are accepted.
"""

import hashlib
import json
import platform
import warnings

import joblib
import numpy
import sklearn

from house_price.config import FEATURES, METADATA_PATH, MODEL_PATH


class ModelIntegrityError(RuntimeError):
    """The model file is missing, changed, or doesn't match its metadata."""


def _sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save_model(model, name, metrics, model_path=MODEL_PATH, metadata_path=METADATA_PATH):
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_path, compress=3)
    metadata = {
        "model": name,
        "features": FEATURES,
        "sha256": _sha256(model_path),
        "versions": {
            "python": platform.python_version(),
            "scikit_learn": sklearn.__version__,
            "numpy": numpy.__version__,
        },
        "test_metrics": metrics,
    }
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")


def load_model(model_path=MODEL_PATH, metadata_path=METADATA_PATH):
    """Return the model after checking it is exactly the file that training produced."""
    if not model_path.exists() or not metadata_path.exists():
        raise ModelIntegrityError("No trained model found. Run: python -m house_price.train")

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if _sha256(model_path) != metadata["sha256"]:
        raise ModelIntegrityError("The model file doesn't match its checksum, so it won't be loaded.")

    trained_with = metadata["versions"]["scikit_learn"]
    if trained_with != sklearn.__version__:
        warnings.warn(
            f"Model was trained with scikit-learn {trained_with} but {sklearn.__version__} is installed. "
            "Retrain it with: python -m house_price.train",
            stacklevel=2,
        )
    return joblib.load(model_path)
