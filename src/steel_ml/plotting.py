"""Dark-themed matplotlib figures for model evaluation."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

BACKGROUND = "#0b0d20"
PANEL = "#111827"
TEXT = "#e2e8f0"
MUTED = "#94a3b8"
GRID = "#334155"
CYAN = "#22d3ee"
PURPLE = "#a78bfa"
AMBER = "#f59e0b"
RED = "#fb7185"

MODEL_COLORS = {
    "Mean Baseline": MUTED,
    "Random Forest": CYAN,
    "PyTorch MLP": PURPLE,
}
FEATURE_SET_COLORS = {"magpie": CYAN, "fractions": AMBER, "both": PURPLE}

THEME = {
    "figure.facecolor": BACKGROUND,
    "axes.facecolor": PANEL,
    "savefig.facecolor": BACKGROUND,
    "axes.edgecolor": GRID,
    "axes.labelcolor": TEXT,
    "axes.titlecolor": TEXT,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "text.color": TEXT,
    "grid.color": GRID,
    "grid.alpha": 0.45,
    "grid.linestyle": "--",
    "axes.grid": True,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.titleweight": "bold",
    "axes.titlesize": 16,
    "axes.labelsize": 12,
    "font.family": "DejaVu Sans",
    "legend.frameon": True,
    "legend.facecolor": PANEL,
    "legend.edgecolor": GRID,
    "legend.labelcolor": TEXT,
}


def apply_theme():
    """Apply the SciencePlots base style plus the dark palette."""
    import scienceplots  # noqa: F401  registers the "science" styles

    plt.style.use(["science", "no-latex", "grid"])
    plt.rcParams.update(THEME)


def save(fig, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return path


def _metric_box(ax, metrics, n_samples):
    ax.text(
        0.04,
        0.96,
        f"MAE   {metrics['mae']:.1f} MPa\n"
        f"R²      {metrics['r2']:.3f}\n"
        f"Samples  {n_samples}",
        transform=ax.transAxes,
        verticalalignment="top",
        bbox={
            "boxstyle": "round,pad=0.6",
            "facecolor": BACKGROUND,
            "edgecolor": GRID,
            "alpha": 0.92,
        },
    )


def _colorbar(fig, mappable, ax, label):
    colorbar = fig.colorbar(mappable, ax=ax, pad=0.02)
    colorbar.set_label(label, color=TEXT)
    colorbar.ax.tick_params(colors=MUTED)
    colorbar.outline.set_edgecolor(GRID)
    return colorbar


def parity_plot(y_true, y_pred, metrics, title, path, cmap="viridis"):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    absolute_error = np.abs(y_true - y_pred)

    minimum_value = min(y_true.min(), y_pred.min())
    maximum_value = max(y_true.max(), y_pred.max())
    padding = (maximum_value - minimum_value) * 0.05
    minimum_value -= padding
    maximum_value += padding

    fig, ax = plt.subplots(figsize=(9, 7))
    scatter = ax.scatter(
        y_true,
        y_pred,
        c=absolute_error,
        cmap=cmap,
        s=85,
        alpha=0.88,
        edgecolors="white",
        linewidths=0.4,
    )
    ax.plot(
        [minimum_value, maximum_value],
        [minimum_value, maximum_value],
        linestyle="--",
        linewidth=2.2,
        color=CYAN,
        label="Perfect prediction",
    )
    ax.set_xlim(minimum_value, maximum_value)
    ax.set_ylim(minimum_value, maximum_value)
    ax.set_xlabel("Actual Yield Strength (MPa)")
    ax.set_ylabel("Predicted Yield Strength (MPa)")
    ax.set_title(title)

    _metric_box(ax, metrics, len(y_true))
    _colorbar(fig, scatter, ax, "Absolute Prediction Error (MPa)")
    ax.legend(loc="lower right")
    return save(fig, path)


def residual_plot(y_true, y_pred, title, path, cmap="magma"):
    y_pred = np.asarray(y_pred)
    residuals = np.asarray(y_true) - y_pred

    fig, ax = plt.subplots(figsize=(9, 6))
    scatter = ax.scatter(
        y_pred,
        residuals,
        c=np.abs(residuals),
        cmap=cmap,
        s=80,
        alpha=0.88,
        edgecolors="white",
        linewidths=0.4,
    )
    ax.axhline(0, linestyle="--", linewidth=2, color=CYAN)
    ax.set_xlabel("Predicted Yield Strength (MPa)")
    ax.set_ylabel("Residual: Actual - Predicted (MPa)")
    ax.set_title(title)

    _colorbar(fig, scatter, ax, "Absolute Residual (MPa)")
    return save(fig, path)


def feature_importance_plot(importances, path, top_n=10, ylabel="Composition Feature"):
    """``importances`` is a pandas Series indexed by feature name."""
    top = importances.sort_values(ascending=False).head(top_n).sort_values()
    bar_colors = plt.cm.viridis(np.linspace(0.25, 0.9, len(top)))

    fig, ax = plt.subplots(figsize=(11, 7))
    bars = ax.barh(top.index, top.values, color=bar_colors, height=0.68)
    ax.bar_label(bars, fmt="%.3f", padding=5, color=TEXT, fontsize=9)
    ax.set_xlabel("Relative Feature Importance")
    ax.set_ylabel(ylabel)
    ax.set_title("Most Influential Steel Composition Features")
    ax.set_xlim(0, top.max() * 1.18)
    return save(fig, path)


def learning_curve_plot(history, best_epoch, path):
    """Training and validation loss (standardized MSE) with the restored epoch."""
    train_loss = np.asarray(history["train_loss"])
    val_loss = np.asarray(history["val_loss"])
    epochs = np.arange(len(train_loss))
    floor = min(train_loss.min(), val_loss.min())

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(epochs, train_loss, color=PURPLE, linewidth=2.4, label="Training loss")
    ax.fill_between(epochs, train_loss, floor, color=PURPLE, alpha=0.12)
    ax.plot(epochs, val_loss, color=CYAN, linewidth=2.0, label="Validation loss")
    ax.scatter(
        best_epoch,
        val_loss[best_epoch],
        color=AMBER,
        s=95,
        edgecolors="white",
        linewidths=0.6,
        zorder=5,
        label=f"Best validation loss: epoch {best_epoch}",
    )
    ax.set_yscale("log")
    ax.set_xlabel("Training Epoch")
    ax.set_ylabel("Mean Squared Error Loss (standardized)")
    ax.set_title("PyTorch Neural Network Learning Curve")
    ax.legend()
    return save(fig, path)


def model_comparison_plot(names, mae_means, mae_stds, path, title="Model Performance Comparison"):
    colors = [MODEL_COLORS.get(name, AMBER) for name in names]
    tops = np.asarray(mae_means) + np.asarray(mae_stds)

    fig, ax = plt.subplots(figsize=(9, 6))
    bars = ax.bar(
        names,
        mae_means,
        yerr=mae_stds,
        color=colors,
        width=0.62,
        error_kw={"ecolor": TEXT, "capsize": 6, "elinewidth": 1.4},
    )
    ax.bar_label(
        bars,
        labels=[f"{value:.1f} MPa" for value in mae_means],
        padding=7,
        color=TEXT,
        fontsize=10,
    )
    ax.set_ylabel("Mean Absolute Error (MPa)")
    ax.set_title(title)
    ax.set_ylim(0, tops.max() * 1.2)
    return save(fig, path)


def feature_set_comparison_plot(summary, path):
    """Grouped bars of mean CV MAE: models on the x axis, one bar per feature set.

    ``summary`` is a DataFrame with columns model, feature_set, mae_mean, mae_std.
    """
    models = list(dict.fromkeys(summary["model"]))
    feature_sets = list(dict.fromkeys(summary["feature_set"]))
    width = 0.8 / len(feature_sets)
    x = np.arange(len(models))

    fig, ax = plt.subplots(figsize=(11, 6))
    for i, feature_set in enumerate(feature_sets):
        rows = summary[summary["feature_set"] == feature_set].set_index("model").reindex(models)
        bars = ax.bar(
            x + (i - (len(feature_sets) - 1) / 2) * width,
            rows["mae_mean"],
            width=width * 0.92,
            yerr=rows["mae_std"],
            color=FEATURE_SET_COLORS.get(feature_set, RED),
            label=feature_set,
            error_kw={"ecolor": TEXT, "capsize": 4, "elinewidth": 1.2},
        )
        ax.bar_label(bars, fmt="%.0f", padding=4, color=TEXT, fontsize=9)

    ax.set_xticks(x, models)
    ax.set_ylabel("Mean Absolute Error (MPa)")
    ax.set_title("Feature Set Comparison (Matbench 5-fold CV)")
    ax.set_ylim(0, (summary["mae_mean"] + summary["mae_std"]).max() * 1.2)
    ax.legend(title="Features", labelcolor=TEXT)
    return save(fig, path)
