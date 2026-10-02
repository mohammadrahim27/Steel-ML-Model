"""Run every model on every feature set with the Matbench 5-fold protocol.

Usage:
    python scripts/benchmark.py
    python scripts/benchmark.py --feature-sets magpie --quick
"""

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from steel_ml import plotting
from steel_ml.data import FEATURE_SETS, load_features
from steel_ml.evaluation import METRICS, cross_validate, matbench_folds
from steel_ml.models import MeanBaseline, TorchMLPRegressor, make_random_forest

ROOT = Path(__file__).resolve().parents[1]

FIGURE_NAMES = {
    "Random Forest": ("random_forest", "Random Forest", "viridis"),
    "PyTorch MLP": ("neural_network", "PyTorch Neural Network", "plasma"),
}


def model_factories(quick):
    rf_trees = 100 if quick else 500
    mlp_epochs = 100 if quick else 1000
    return {
        "Mean Baseline": MeanBaseline,
        "Random Forest": lambda: make_random_forest(n_estimators=rf_trees),
        "PyTorch MLP": lambda: TorchMLPRegressor(max_epochs=mlp_epochs),
    }


def to_markdown(df, floatfmt=".3f"):
    """Markdown table without requiring the optional ``tabulate`` dependency."""
    def cell(value):
        return format(value, floatfmt) if isinstance(value, float) else str(value)

    lines = [
        "| " + " | ".join(df.columns) + " |",
        "|" + "|".join(" --- " for _ in df.columns) + "|",
    ]
    lines += ["| " + " | ".join(cell(v) for v in row) + " |" for row in df.itertuples(index=False)]
    return "\n".join(lines) + "\n"


def make_figures(feature_set, X, y, results, figures_dir):
    suffix = f"_{feature_set}"
    for name, (stem, label, cmap) in FIGURE_NAMES.items():
        if name not in results:
            continue
        result = results[name]
        mean_metrics = result.per_fold[list(METRICS)].mean()
        plotting.parity_plot(
            y, result.oof_predictions, mean_metrics,
            f"{label} Prediction Performance",
            figures_dir / f"{stem}_predictions{suffix}.png",
            cmap=cmap,
        )
        plotting.residual_plot(
            y, result.oof_predictions,
            f"{label} Residual Analysis",
            figures_dir / f"{stem}_residuals{suffix}.png",
        )

    if "Random Forest" in results:
        importances = pd.Series(
            np.mean([m.feature_importances_ for m in results["Random Forest"].models], axis=0),
            index=X.columns,
        ).rename(lambda c: c.removeprefix("MagpieData "))
        plotting.feature_importance_plot(
            importances,
            figures_dir / f"feature_importance{suffix}.png",
            ylabel={"magpie": "Magpie Descriptor", "fractions": "Element Fraction"}.get(
                feature_set, "Composition Feature"
            ),
        )

    if "PyTorch MLP" in results:
        first_fold = results["PyTorch MLP"].models[0]
        plotting.learning_curve_plot(
            first_fold.history_,
            first_fold.best_epoch_,
            figures_dir / f"neural_network_training_loss{suffix}.png",
        )

    names = list(results)
    plotting.model_comparison_plot(
        names,
        [results[n].per_fold["mae"].mean() for n in names],
        [results[n].per_fold["mae"].std() for n in names],
        figures_dir / f"model_comparison{suffix}.png",
        title=f"Model Performance Comparison ({feature_set})",
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--feature-sets", nargs="+", default=list(FEATURE_SETS), choices=FEATURE_SETS)
    parser.add_argument("--figures-dir", type=Path, default=ROOT / "figures")
    parser.add_argument("--results-dir", type=Path, default=ROOT / "results")
    parser.add_argument("--quick", action="store_true", help="fewer trees and epochs, for smoke tests")
    args = parser.parse_args()

    np.random.seed(42)
    torch.manual_seed(42)
    plotting.apply_theme()
    args.results_dir.mkdir(parents=True, exist_ok=True)

    per_fold, oof_frames = [], []
    for feature_set in args.feature_sets:
        X, y = load_features(feature_set)
        folds = matbench_folds(len(y))
        print(f"\n=== {feature_set}: {X.shape[0]} samples x {X.shape[1]} features ===")

        results = {}
        for name, factory in model_factories(args.quick).items():
            start = time.perf_counter()
            print(f"{name}")
            result = cross_validate(factory, X, y, folds, verbose=True)
            results[name] = result
            mean = result.per_fold[list(METRICS)].mean()
            print(
                f"  mean: MAE {mean['mae']:.1f} MPa  RMSE {mean['rmse']:.1f}  "
                f"R² {mean['r2']:.3f}  MAE/MAD {mean['mae_mad']:.3f}  "
                f"({time.perf_counter() - start:.0f}s)"
            )
            per_fold.append(result.per_fold.assign(model=name, feature_set=feature_set))
            oof_frames.append(pd.DataFrame({
                "feature_set": feature_set,
                "model": name,
                "row": np.arange(len(y)),
                "y_true": y.to_numpy(),
                "y_pred": result.oof_predictions,
            }))

        make_figures(feature_set, X, y, results, args.figures_dir)

    per_fold = pd.concat(per_fold, ignore_index=True)
    per_fold = per_fold[["feature_set", "model", "fold", "n_test", *METRICS]]
    summary = per_fold.groupby(["feature_set", "model"], sort=False)[list(METRICS)].agg(["mean", "std"])
    summary.columns = [f"{metric}_{stat}" for metric, stat in summary.columns]
    summary = summary.reset_index()

    if len(args.feature_sets) > 1:
        plotting.feature_set_comparison_plot(summary, args.figures_dir / "feature_set_comparison.png")

    per_fold.to_csv(args.results_dir / "per_fold_metrics.csv", index=False)
    summary.to_csv(args.results_dir / "summary.csv", index=False)
    pd.concat(oof_frames, ignore_index=True).to_csv(args.results_dir / "oof_predictions.csv", index=False)

    table = pd.DataFrame({
        "Features": summary["feature_set"],
        "Model": summary["model"],
        **{
            label: [f"{m:.{p}f} ± {s:.{p}f}" for m, s in zip(summary[f"{key}_mean"], summary[f"{key}_std"])]
            for key, label, p in [("mae", "MAE (MPa)", 1), ("rmse", "RMSE (MPa)", 1),
                                  ("r2", "R²", 3), ("mae_mad", "MAE/MAD", 3)]
        },
    })
    (args.results_dir / "summary.md").write_text(to_markdown(table), encoding="utf-8")
    (args.results_dir / "per_fold_metrics.md").write_text(to_markdown(per_fold), encoding="utf-8")

    print("\n" + to_markdown(table))
    print(f"Results written to {args.results_dir}, figures to {args.figures_dir}")


if __name__ == "__main__":
    main()
