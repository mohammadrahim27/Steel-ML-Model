"""Reference models: a mean predictor and a random forest."""

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.ensemble import RandomForestRegressor


class MeanBaseline(RegressorMixin, BaseEstimator):
    """Predicts the training-set mean for every sample."""

    def fit(self, X, y):
        self.mean_ = float(np.mean(y))
        return self

    def predict(self, X):
        return np.full(len(X), self.mean_)


def make_random_forest(n_estimators=500, random_state=42, n_jobs=-1, **kwargs):
    return RandomForestRegressor(
        n_estimators=n_estimators,
        random_state=random_state,
        n_jobs=n_jobs,
        **kwargs,
    )
