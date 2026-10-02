import numpy as np
import pandas as pd
import pytest

from steel_ml.data import build_features, fraction_features, present_elements, add_compositions

COMPOSITIONS = ["Fe0.7Cr0.2Ni0.1", "Fe0.95C0.05", "Fe0.9Mn0.05Si0.05"]


@pytest.fixture
def df():
    return add_compositions(pd.DataFrame({"composition": COMPOSITIONS}))


def test_present_elements_sorted_by_atomic_number(df):
    assert present_elements(df["composition_obj"]) == ["C", "Si", "Cr", "Mn", "Fe", "Ni"]


def test_fraction_features_sum_to_one(df):
    fractions = fraction_features(df["composition_obj"])
    assert list(fractions.columns) == ["frac_C", "frac_Si", "frac_Cr", "frac_Mn", "frac_Fe", "frac_Ni"]
    np.testing.assert_allclose(fractions.sum(axis=1), 1.0)
    assert fractions.loc[0, "frac_Cr"] == pytest.approx(0.2)
    assert fractions.loc[1, "frac_Ni"] == 0.0


def test_fraction_features_fixed_element_list(df):
    fractions = fraction_features(df["composition_obj"], elements=["Fe", "Mo"])
    assert list(fractions.columns) == ["frac_Fe", "frac_Mo"]
    assert (fractions["frac_Mo"] == 0).all()


@pytest.mark.parametrize("kind, n_magpie, n_fraction", [
    ("magpie", 132, 0),
    ("fractions", 0, 6),
    ("both", 132, 6),
])
def test_build_features_kinds(df, kind, n_magpie, n_fraction):
    X = build_features(df, kind)
    assert X.shape == (len(COMPOSITIONS), n_magpie + n_fraction)
    assert sum(c.startswith("MagpieData") for c in X.columns) == n_magpie
    assert sum(c.startswith("frac_") for c in X.columns) == n_fraction
    assert not X.isna().any().any()


def test_build_features_rejects_unknown_kind(df):
    with pytest.raises(ValueError):
        build_features(df, "unknown")
