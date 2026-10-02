"""Loading and featurizing the Matbench steels dataset."""

from pathlib import Path

import pandas as pd
from matminer.featurizers.composition import ElementProperty
from pymatgen.core import Composition

TARGET = "yield strength"
FEATURE_SETS = ("magpie", "fractions", "both")
FRACTION_PREFIX = "frac_"

DEFAULT_DATA_DIR = Path(__file__).resolve().parents[2] / "data"


def load_steels(data_dir=DEFAULT_DATA_DIR):
    """Load matbench_steels in its published row order.

    The row order matters: the Matbench folds are defined on row positions.
    """
    from matminer.datasets import load_dataset

    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    df = load_dataset("matbench_steels", data_home=str(data_dir))
    return df.reset_index(drop=True)


def add_compositions(df, column="composition"):
    """Add a pymatgen Composition column parsed from the formula strings."""
    df = df.copy()
    df["composition_obj"] = df[column].map(Composition)
    return df


def magpie_features(compositions):
    """Magpie elemental-property statistics (132 descriptors)."""
    featurizer = ElementProperty.from_preset("magpie")
    featurizer.set_n_jobs(1)
    frame = pd.DataFrame({"composition_obj": list(compositions)})
    frame = featurizer.featurize_dataframe(frame, "composition_obj", pbar=False)
    return frame.drop(columns=["composition_obj"])


def present_elements(compositions):
    """Symbols of every element in any composition, ordered by atomic number."""
    elements = {el for comp in compositions for el in comp.elements}
    return [el.symbol for el in sorted(elements, key=lambda el: el.Z)]


def fraction_features(compositions, elements=None):
    """Atomic fraction of each element, one column per element.

    ``elements`` fixes the column set so new compositions can be featurized
    consistently with the training data. By default it is every element present.
    """
    compositions = list(compositions)
    if elements is None:
        elements = present_elements(compositions)
    rows = [
        [comp.get_atomic_fraction(el) for el in elements]
        for comp in compositions
    ]
    return pd.DataFrame(rows, columns=[f"{FRACTION_PREFIX}{el}" for el in elements])


def build_features(df, kind="magpie", elements=None):
    """Return the feature matrix for ``kind`` in {"magpie", "fractions", "both"}."""
    if kind not in FEATURE_SETS:
        raise ValueError(f"kind must be one of {FEATURE_SETS}, got {kind!r}")
    if "composition_obj" not in df:
        df = add_compositions(df)

    parts = []
    if kind in ("magpie", "both"):
        parts.append(magpie_features(df["composition_obj"]))
    if kind in ("fractions", "both"):
        parts.append(fraction_features(df["composition_obj"], elements))

    X = pd.concat(parts, axis=1)
    X.index = df.index
    return X


def load_features(kind="magpie", data_dir=DEFAULT_DATA_DIR, use_cache=True):
    """Load the dataset and featurize it, caching the result under ``data_dir``."""
    data_dir = Path(data_dir)
    cache = data_dir / "cache" / f"steels_{kind}.pkl"
    if use_cache and cache.exists():
        cached = pd.read_pickle(cache)
        return cached.drop(columns=[TARGET]), cached[TARGET]

    df = load_steels(data_dir)
    X = build_features(df, kind)
    y = df[TARGET]

    cache.parent.mkdir(parents=True, exist_ok=True)
    X.assign(**{TARGET: y}).to_pickle(cache)
    return X, y
