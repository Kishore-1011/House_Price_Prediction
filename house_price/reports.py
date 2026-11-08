"""Charts saved alongside the metrics."""

from matplotlib.figure import Figure

from house_price.config import FIGURES_DIR, LOCALITY_RATES

BLUE = "#2563eb"
GREY = "#94a3b8"
LAKH = 100_000


def _save(fig, name, directory):
    directory.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(directory / name, dpi=130)


def plot_predicted_vs_actual(actual, predicted, title, directory=FIGURES_DIR):
    fig = Figure(figsize=(6, 5))
    axis = fig.add_subplot()
    axis.scatter(actual / LAKH, predicted / LAKH, s=12, alpha=0.5, color=BLUE)
    limit = max(actual.max(), predicted.max()) / LAKH
    axis.plot([0, limit], [0, limit], color=GREY, linewidth=1, label="Perfect prediction")
    axis.set_xlabel("Actual price (lakh)")
    axis.set_ylabel("Predicted price (lakh)")
    axis.set_title(title)
    axis.legend(frameon=False)
    _save(fig, "predicted_vs_actual.png", directory)


def plot_model_comparison(test_scores, best_name, directory=FIGURES_DIR):
    ordered = sorted(test_scores.items(), key=lambda pair: pair[1]["mae"], reverse=True)
    names = [name.replace("_", " ") for name, _ in ordered]
    errors = [score["mae"] / LAKH for _, score in ordered]
    colours = [BLUE if name == best_name else GREY for name, _ in ordered]

    fig = Figure(figsize=(7, 3.6))
    axis = fig.add_subplot()
    bars = axis.barh(names, errors, color=colours)
    axis.bar_label(bars, fmt="%.1f", padding=3)
    axis.set_xlabel("Average error on unseen houses (lakh)")
    axis.set_title("Model comparison (chosen model in blue)")
    axis.spines[["top", "right"]].set_visible(False)
    axis.margins(x=0.12)
    _save(fig, "model_comparison.png", directory)


def plot_price_by_area(data, directory=FIGURES_DIR):
    fig = Figure(figsize=(7, 4.4))
    axis = fig.add_subplot()
    palette = ["#2563eb", "#dc2626", "#16a34a", "#d97706", "#7c3aed", "#0d9488"]
    for colour, locality in zip(palette, LOCALITY_RATES):
        part = data[data["locality"] == locality]
        axis.scatter(part["area"], part["price"] / LAKH, s=8, alpha=0.45, color=colour, label=locality)
    axis.set_xlabel("Area (sq ft)")
    axis.set_ylabel("Price (lakh)")
    axis.set_title("Made-up houses: price by area and locality")
    axis.spines[["top", "right"]].set_visible(False)
    axis.legend(frameon=False, markerscale=2, fontsize=8)
    _save(fig, "price_by_area.png", directory)
