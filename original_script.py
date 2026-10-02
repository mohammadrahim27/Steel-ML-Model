import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt


from matminer.featurizers.conversions import StrToComposition
from matminer.featurizers.composition import ElementProperty
from matminer.datasets import load_dataset
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score 
from sklearn.model_selection import KFold




if __name__ == "__main__":
    np.random.seed(42)
    torch.manual_seed(42)

    os.makedirs("figures", exist_ok=True)

    plt.style.use(["science", "no-latex", "grid"])

    background = "#0b0d20"
    panel = "#111827"
    text = "#e2e8f0"
    muted = "#94a3b8"
    grid = "#334155"
    cyan = "#22d3ee"
    purple = "#a78bfa"
    amber = "#f59e0b"
    red = "#fb7185"

    plt.rcParams.update({
        "figure.facecolor": background,
        "axes.facecolor": panel,
        "savefig.facecolor": background,
        "axes.edgecolor": grid,
        "axes.labelcolor": text,
        "axes.titlecolor": text,
        "xtick.color": muted,
        "ytick.color": muted,
        "text.color": text,
        "grid.color": grid,
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
        "legend.facecolor": panel,
        "legend.edgecolor": grid,
        "legend.labelcolor": text
    })

    stc = StrToComposition(target_col_id="composition_obj")
    ep = ElementProperty.from_preset("magpie")

    df = load_dataset("matbench_steels")
    df = stc.featurize_dataframe(df, "composition")
    df = ep.featurize_dataframe(df, "composition_obj")

    y = df["yield strength"]



    X = df.drop(
        columns=[
            "composition",
            "composition_obj",
            "yield strength"
        ]
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print(X.shape)
    print(y.shape)

    print(X_train.shape)
    print(X_test.shape)
    print(y_train.shape)
    print(y_test.shape)

    model = RandomForestRegressor(
        random_state=42
    )

    model.fit(X_train, y_train)

    prediction = model.predict(X_test)

    print(prediction[:5])
    print(y_test[:5].values)

    mae = mean_absolute_error(y_test, prediction)
    rf_r2 = r2_score(y_test, prediction)

    print("MAE:", mae)
    print("R2:", rf_r2)

    cv_scores = cross_val_score(model, X, y, cv=5, scoring="neg_mean_absolute_error")
    print("CV scores(mean):", -cv_scores)
    print("Mean CV MAE:", -cv_scores.mean())
    print("Std CV Mean:", cv_scores.std())

    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    cv_shuffled = cross_val_score(model, X, y, cv=kf, scoring="neg_mean_absolute_error")
    print("Shuffled CV MAE:", -cv_shuffled.mean())
    print("Shuffled CV scores:", -cv_shuffled)



    baseline_value = y_train.mean()

    baseline_prediction = np.full(
        len(y_test),
        baseline_value
    )

    baseline_mae = mean_absolute_error(
        y_test,
        baseline_prediction
    )

    print("Baseline MAE:", baseline_mae)
    print("Model MAE:", mae)

    actual_values = y_test.to_numpy()
    absolute_error = np.abs(actual_values - prediction)

    minimum_value = min(
        actual_values.min(),
        prediction.min()
    )

    maximum_value = max(
        actual_values.max(),
        prediction.max()
    )

    padding = (maximum_value - minimum_value) * 0.05

    minimum_value -= padding
    maximum_value += padding

    fig, ax = plt.subplots(figsize=(9, 7))

    scatter = ax.scatter(
        actual_values,
        prediction,
        c=absolute_error,
        cmap="viridis",
        s=85,
        alpha=0.88,
        edgecolors="white",
        linewidths=0.4
    )

    ax.plot(
        [minimum_value, maximum_value],
        [minimum_value, maximum_value],
        linestyle="--",
        linewidth=2.2,
        color=cyan,
        label="Perfect prediction"
    )

    ax.set_xlim(minimum_value, maximum_value)
    ax.set_ylim(minimum_value, maximum_value)

    ax.set_xlabel("Actual Yield Strength (MPa)")
    ax.set_ylabel("Predicted Yield Strength (MPa)")
    ax.set_title("Random Forest Prediction Performance")

    metric_text = (
        f"MAE   {mae:.1f} MPa\n"
        f"R²      {rf_r2:.3f}\n"
        f"Samples  {len(y_test)}"
    )

    ax.text(
        0.04,
        0.96,
        metric_text,
        transform=ax.transAxes,
        verticalalignment="top",
        bbox={
            "boxstyle": "round,pad=0.6",
            "facecolor": background,
            "edgecolor": grid,
            "alpha": 0.92
        }
    )

    colorbar = fig.colorbar(
        scatter,
        ax=ax,
        pad=0.02
    )

    colorbar.set_label(
        "Absolute Prediction Error (MPa)",
        color=text
    )

    colorbar.ax.tick_params(
        colors=muted
    )

    colorbar.outline.set_edgecolor(grid)

    ax.legend(loc="lower right")

    plt.tight_layout()

    plt.savefig(
        "figures/random_forest_predictions.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.show()

    residuals = actual_values - prediction

    fig, ax = plt.subplots(figsize=(9, 6))

    residual_scatter = ax.scatter(
        prediction,
        residuals,
        c=np.abs(residuals),
        cmap="magma",
        s=80,
        alpha=0.88,
        edgecolors="white",
        linewidths=0.4
    )

    ax.axhline(
        0,
        linestyle="--",
        linewidth=2,
        color=cyan
    )

    ax.set_xlabel("Predicted Yield Strength (MPa)")
    ax.set_ylabel("Residual: Actual - Predicted (MPa)")
    ax.set_title("Random Forest Residual Analysis")

    residual_colorbar = fig.colorbar(
        residual_scatter,
        ax=ax,
        pad=0.02
    )

    residual_colorbar.set_label(
        "Absolute Residual (MPa)",
        color=text
    )

    residual_colorbar.ax.tick_params(
        colors=muted
    )

    residual_colorbar.outline.set_edgecolor(grid)

    plt.tight_layout()

    plt.savefig(
        "figures/random_forest_residuals.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.show()

    importance = model.feature_importances_

    feat_importances = pd.Series(
        importance,
        index=X.columns
    )

    top10 = (
        feat_importances
        .sort_values(ascending=False)
        .head(10)
    )

    top10_for_graph = top10.sort_values(
        ascending=True
    )

    bar_colors = plt.cm.viridis(
        np.linspace(
            0.25,
            0.9,
            len(top10_for_graph)
        )
    )

    fig, ax = plt.subplots(figsize=(11, 7))

    bars = ax.barh(
        top10_for_graph.index,
        top10_for_graph.values,
        color=bar_colors,
        height=0.68
    )

    ax.bar_label(
        bars,
        fmt="%.3f",
        padding=5,
        color=text,
        fontsize=9
    )

    ax.set_xlabel("Relative Feature Importance")
    ax.set_ylabel("Magpie Descriptor")
    ax.set_title("Most Influential Steel Composition Features")

    ax.set_xlim(
        0,
        top10_for_graph.max() * 1.18
    )

    plt.tight_layout()

    plt.savefig(
        "figures/feature_importance.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.show()

    print(top10)

    X_train_t = torch.tensor(
        X_train_scaled,
        dtype=torch.float32
    )

    X_test_t = torch.tensor(
        X_test_scaled,
        dtype=torch.float32
    )

    y_train_t = torch.tensor(
        y_train.values,
        dtype=torch.float32
    ).reshape(-1, 1)

    y_test_t = torch.tensor(
        y_test.values,
        dtype=torch.float32
    ).reshape(-1, 1)

    net = nn.Sequential(
        nn.Linear(X_train_t.shape[1], 64),
        nn.ReLU(),
        nn.Linear(64, 32),
        nn.ReLU(),
        nn.Linear(32, 1)
    )

    loss_fn = nn.MSELoss()

    optimizer = torch.optim.Adam(
        net.parameters(),
        lr=0.01
    )

    loss_history = []

    net.train()

    for epoch in range(500):
        predictions = net(X_train_t)

        loss = loss_fn(
            predictions,
            y_train_t
        )

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_history.append(
            loss.item()
        )

        if epoch % 50 == 0:
            print(
                f"Epoch {epoch}, Loss: {loss.item()}"
            )

    net.eval()

    with torch.no_grad():
        nn_predictions = net(X_test_t)

    nn_predictions_array = (
        nn_predictions
        .numpy()
        .reshape(-1)
    )

    nn_mae = mean_absolute_error(
        y_test,
        nn_predictions_array
    )

    nn_r2 = r2_score(
        y_test,
        nn_predictions_array
    )

    print("Neural Net MAE:", nn_mae)
    print("Neural Net R2:", nn_r2)
    print("Random Forest MAE:", mae)

    nn_absolute_error = np.abs(
        actual_values - nn_predictions_array
    )

    nn_minimum_value = min(
        actual_values.min(),
        nn_predictions_array.min()
    )

    nn_maximum_value = max(
        actual_values.max(),
        nn_predictions_array.max()
    )

    nn_padding = (
        nn_maximum_value - nn_minimum_value
    ) * 0.05

    nn_minimum_value -= nn_padding
    nn_maximum_value += nn_padding

    fig, ax = plt.subplots(figsize=(9, 7))

    nn_scatter = ax.scatter(
        actual_values,
        nn_predictions_array,
        c=nn_absolute_error,
        cmap="plasma",
        s=85,
        alpha=0.88,
        edgecolors="white",
        linewidths=0.4
    )

    ax.plot(
        [nn_minimum_value, nn_maximum_value],
        [nn_minimum_value, nn_maximum_value],
        linestyle="--",
        linewidth=2.2,
        color=cyan,
        label="Perfect prediction"
    )

    ax.set_xlim(
        nn_minimum_value,
        nn_maximum_value
    )

    ax.set_ylim(
        nn_minimum_value,
        nn_maximum_value
    )

    ax.set_xlabel("Actual Yield Strength (MPa)")
    ax.set_ylabel("Predicted Yield Strength (MPa)")
    ax.set_title("PyTorch Neural Network Performance")

    nn_metric_text = (
        f"MAE   {nn_mae:.1f} MPa\n"
        f"R²      {nn_r2:.3f}\n"
        f"Samples  {len(y_test)}"
    )

    ax.text(
        0.04,
        0.96,
        nn_metric_text,
        transform=ax.transAxes,
        verticalalignment="top",
        bbox={
            "boxstyle": "round,pad=0.6",
            "facecolor": background,
            "edgecolor": grid,
            "alpha": 0.92
        }
    )

    nn_colorbar = fig.colorbar(
        nn_scatter,
        ax=ax,
        pad=0.02
    )

    nn_colorbar.set_label(
        "Absolute Prediction Error (MPa)",
        color=text
    )

    nn_colorbar.ax.tick_params(
        colors=muted
    )

    nn_colorbar.outline.set_edgecolor(grid)

    ax.legend(loc="lower right")

    plt.tight_layout()

    plt.savefig(
        "figures/neural_network_predictions.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.show()

    loss_values = np.array(
        loss_history
    )

    epochs = np.arange(
        len(loss_values)
    )

    best_epoch = int(
        np.argmin(loss_values)
    )

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(
        epochs,
        loss_values,
        color=purple,
        linewidth=2.4,
        label="Training loss"
    )

    ax.fill_between(
        epochs,
        loss_values,
        loss_values.min(),
        color=purple,
        alpha=0.12
    )

    ax.scatter(
        best_epoch,
        loss_values[best_epoch],
        color=amber,
        s=95,
        edgecolors="white",
        linewidths=0.6,
        zorder=5,
        label=f"Lowest loss: epoch {best_epoch}"
    )

    ax.set_yscale("log")

    ax.set_xlabel("Training Epoch")
    ax.set_ylabel("Mean Squared Error Loss")
    ax.set_title("PyTorch Neural Network Learning Curve")

    ax.legend()

    plt.tight_layout()

    plt.savefig(
        "figures/neural_network_training_loss.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.show()

    model_names = [
        "Mean Baseline",
        "Random Forest",
        "PyTorch MLP"
    ]

    model_mae_values = [
        baseline_mae,
        mae,
        nn_mae
    ]

    model_colors = [
        muted,
        cyan,
        purple
    ]

    fig, ax = plt.subplots(figsize=(9, 6))

    model_bars = ax.bar(
        model_names,
        model_mae_values,
        color=model_colors,
        width=0.62
    )

    ax.bar_label(
        model_bars,
        labels=[
            f"{value:.1f} MPa"
            for value in model_mae_values
        ],
        padding=7,
        color=text,
        fontsize=10
    )

    ax.set_ylabel(
        "Mean Absolute Error (MPa)"
    )

    ax.set_title(
        "Model Performance Comparison"
    )

    ax.set_ylim(
        0,
        max(model_mae_values) * 1.2
    )

    plt.tight_layout()

    plt.savefig(
        "figures/model_comparison.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.show()

    print(net)