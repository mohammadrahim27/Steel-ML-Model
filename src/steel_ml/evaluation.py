"""Matbench cross-validation protocol and regression metrics."""

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold

# Matbench v0.1 regression tasks use this exact splitter on the dataset's
# published row order (matbench.constants.VALIDATION_CROSS_VALIDATOR).
MATBENCH_N_FOLDS = 5
MATBENCH_SEED = 18012019

METRICS = ("mae", "rmse", "r2", "mae_mad")


def matbench_folds(n_samples):
    """List of (train_idx, test_idx) arrays for the official Matbench folds."""
    splitter = KFold(n_splits=MATBENCH_N_FOLDS, shuffle=True, random_state=MATBENCH_SEED)
    return list(splitter.split(np.arange(n_samples)))


def mean_absolute_deviation(y):
    y = np.asarray(y, dtype=float)
    return float(np.mean(np.abs(y - y.mean())))


def regression_metrics(y_true, y_pred, mad=None):
    """MAE, RMSE, R² and MAE/MAD.

    ``mad`` defaults to the mean absolute deviation of ``y_true``. Matbench
    reports MAE/MAD against the MAD of the full dataset, so pass that for
    leaderboard-comparable numbers. MAE/MAD = 1 means no better than predicting
    the mean; lower is better.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    if mad is None:
        mad = mean_absolute_deviation(y_true)
    mae = mean_absolute_error(y_true, y_pred)
    return {
        "mae": mae,
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": r2_score(y_true, y_pred),
        "mae_mad": mae / mad,
    }


@dataclass
class CVResult:
    """Per-fold metrics, out-of-fold predictions, and the fitted fold models."""

    per_fold: pd.DataFrame
    oof_predictions: np.ndarray
    models: list = field(default_factory=list)

    def summary(self):
        """Mean and standard deviation of each metric across folds."""
        return self.per_fold[list(METRICS)].agg(["mean", "std"])


def cross_validate(model_factory, X, y, folds=None, verbose=False):
    """Fit a fresh model on each fold's training data and score its test fold.

    ``model_factory`` is a zero-argument callable returning an unfitted model
    with ``fit(X, y)`` and ``predict(X)``.
    """
    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y, dtype=float)
    if folds is None:
        folds = matbench_folds(len(y))
    dataset_mad = mean_absolute_deviation(y)

    rows, models = [], []
    oof = np.full(len(y), np.nan)
    for fold, (train_idx, test_idx) in enumerate(folds, start=1):
        model = model_factory()
        model.fit(X[train_idx], y[train_idx])
        pred = np.asarray(model.predict(X[test_idx]), dtype=float).reshape(-1)
        oof[test_idx] = pred
        scores = regression_metrics(y[test_idx], pred, mad=dataset_mad)
        rows.append({"fold": fold, "n_test": len(test_idx), **scores})
        models.append(model)
        if verbose:
            print(f"  fold {fold}: MAE {scores['mae']:.1f}  R² {scores['r2']:.3f}")

    return CVResult(pd.DataFrame(rows), oof, models)
