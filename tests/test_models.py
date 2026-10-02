import numpy as np
import pytest

from steel_ml.models import MeanBaseline, TorchMLPRegressor, make_random_forest


@pytest.fixture
def linear_data():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(200, 5))
    # Large offset and scale, like yield strength in MPa, to exercise target scaling.
    y = 1000 + 300 * X[:, 0] - 150 * X[:, 1] + rng.normal(scale=10, size=200)
    return X, y


def test_mean_baseline(linear_data):
    X, y = linear_data
    pred = MeanBaseline().fit(X, y).predict(X[:7])
    np.testing.assert_allclose(pred, y.mean())
    assert pred.shape == (7,)


def test_random_forest_beats_baseline(linear_data):
    X, y = linear_data
    rf = make_random_forest(n_estimators=50).fit(X[:150], y[:150])
    rf_mae = np.mean(np.abs(rf.predict(X[150:]) - y[150:]))
    base_mae = np.mean(np.abs(y[:150].mean() - y[150:]))
    assert rf_mae < base_mae


def test_mlp_learns_in_original_units(linear_data):
    X, y = linear_data
    mlp = TorchMLPRegressor(max_epochs=300, patience=30).fit(X[:150], y[:150])
    pred = mlp.predict(X[150:])
    assert pred.shape == (50,)
    mae = np.mean(np.abs(pred - y[150:]))
    base_mae = np.mean(np.abs(y[:150].mean() - y[150:]))
    assert mae < 0.3 * base_mae


def test_mlp_early_stopping_records_history(linear_data):
    X, y = linear_data
    mlp = TorchMLPRegressor(max_epochs=500, patience=5).fit(X, y)
    n_epochs = len(mlp.history_["val_loss"])
    assert n_epochs < 500
    assert mlp.best_epoch_ == int(np.argmin(mlp.history_["val_loss"]))
    assert len(mlp.history_["train_loss"]) == len(mlp.history_["lr"]) == n_epochs


def test_mlp_is_reproducible(linear_data):
    X, y = linear_data
    a = TorchMLPRegressor(max_epochs=20).fit(X, y).predict(X[:5])
    b = TorchMLPRegressor(max_epochs=20).fit(X, y).predict(X[:5])
    np.testing.assert_allclose(a, b)
