import numpy as np
import pytest

from steel_ml.evaluation import (
    METRICS,
    cross_validate,
    matbench_folds,
    mean_absolute_deviation,
    regression_metrics,
)
from steel_ml.models import MeanBaseline


def test_matbench_folds_partition_steels():
    folds = matbench_folds(312)
    assert len(folds) == 5
    assert sorted(len(test) for _, test in folds) == [62, 62, 62, 63, 63]
    all_test = np.concatenate([test for _, test in folds])
    np.testing.assert_array_equal(np.sort(all_test), np.arange(312))
    for train, test in folds:
        assert not set(train) & set(test)


def test_matbench_folds_are_deterministic():
    a, b = matbench_folds(312), matbench_folds(312)
    for (_, test_a), (_, test_b) in zip(a, b):
        np.testing.assert_array_equal(test_a, test_b)


def test_regression_metrics_known_values():
    y_true = np.array([1.0, 2.0, 3.0, 4.0])
    y_pred = np.array([1.0, 2.0, 3.0, 6.0])
    m = regression_metrics(y_true, y_pred)
    assert m["mae"] == pytest.approx(0.5)
    assert m["rmse"] == pytest.approx(1.0)
    assert m["r2"] == pytest.approx(1 - 4 / 5)
    assert m["mae_mad"] == pytest.approx(0.5 / 1.0)


def test_mean_prediction_has_mae_mad_of_one():
    y = np.array([3.0, 7.0, 10.0, 20.0])
    m = regression_metrics(y, np.full_like(y, y.mean()))
    assert m["mae_mad"] == pytest.approx(1.0)
    assert mean_absolute_deviation(y) == pytest.approx(5.0)


def test_cross_validate_fills_every_oof_prediction():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(50, 3))
    y = rng.normal(size=50)
    result = cross_validate(MeanBaseline, X, y)
    assert len(result.per_fold) == 5
    assert set(METRICS) <= set(result.per_fold.columns)
    assert not np.isnan(result.oof_predictions).any()
    assert list(result.summary().index) == ["mean", "std"]
