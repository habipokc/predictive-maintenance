"""Loading and reshaping the AI4I 2020 predictive maintenance dataset.

UCI hands back the column names WITHOUT their physical units, so we record
them here once and never guess again downstream:

    Air temperature      K     ambient air around the machine
    Process temperature  K     the machine's own process temperature
    Rotational speed     rpm   spindle speed
    Torque               Nm    torque applied by the tool
    Tool wear            min   cumulative minutes of use on the current tool

The five failure-mode columns (TWF/HDF/PWF/OSF/RNF) are LABELS, not inputs.
They are kept in the frame on purpose so Phase 3 can demonstrate what happens
when they leak into the feature set, and so Phase 10 can break the misses down
by mode. Any code that trains a model must go through `feature_columns()`.
"""

import math

import pandas as pd
from ucimlrepo import fetch_ucirepo

from src.config import DATA_RAW, RANDOM_SEED, ensure_dirs

UCI_DATASET_ID = 601
CACHE_FILE = DATA_RAW / "ai4i2020.csv"

ID_COLUMNS = ["UID", "Product ID"]
TARGET = "Machine failure"
FAILURE_MODES = ["TWF", "HDF", "PWF", "OSF", "RNF"]

# The only columns a sensor on the shop floor could actually give us.
SENSOR_COLUMNS = [
    "Air temperature",
    "Process temperature",
    "Rotational speed",
    "Torque",
    "Tool wear",
]
CATEGORICAL_COLUMNS = ["Type"]

# Derived in Phase 4 from the sensor columns above. Every one of them is
# computable at prediction time from readings a machine already produces, so
# none of them is leakage - see `add_physical_features` for the physics.
DERIVED_COLUMNS = ["power", "temp_diff", "wear_torque"]


def load_raw() -> pd.DataFrame:
    """Return the full dataset: ids + features + target + failure modes.

    Fetched from UCI on first call and cached on disk, so that every later run
    reads the exact same bytes instead of depending on a network round trip.
    """
    if CACHE_FILE.exists():
        return pd.read_csv(CACHE_FILE)

    dataset = fetch_ucirepo(id=UCI_DATASET_ID)
    df = pd.concat(
        [dataset.data.ids, dataset.data.features, dataset.data.targets], axis=1
    )

    ensure_dirs()
    df.to_csv(CACHE_FILE, index=False)
    return df


def add_physical_features(df: pd.DataFrame) -> pd.DataFrame:
    """Derive the three quantities the machine actually fails on.

    The raw columns describe the machine one sensor at a time, but the
    failure mechanisms live in combinations of them:

        power       torque x angular velocity. The rpm reading has to be
                    converted to rad/s first, hence 2*pi/60. This is the
                    number on a motor's nameplate, and a spindle is
                    overloaded or stalling according to it - never according
                    to torque alone.
        temp_diff   how much hotter the process runs than the air around it.
                    Both temperatures drift together with the season and the
                    shop floor; the gradient is what drives heat away, so the
                    gradient is what matters.
        wear_torque a blunt tool cutting hard is a different situation from a
                    blunt tool cutting soft. Neither factor alone says so.

    A decision tree can only threshold one column per split, so a rule of the
    form `a - b < 8.6` costs it several levels of depth to approximate. Handing
    it the difference directly buys that rule for one split.

    Note what this function does NOT do: it does not encode any of the
    documented failure thresholds. We derive the quantities; the model has to
    find the cut points on its own, or it has learned nothing.
    """
    df = df.copy()
    df["power"] = df["Torque"] * df["Rotational speed"] * 2 * math.pi / 60
    df["temp_diff"] = df["Process temperature"] - df["Air temperature"]
    df["wear_torque"] = df["Tool wear"] * df["Torque"]
    return df


def feature_columns(
    include_failure_modes: bool = False,
    include_derived: bool = False,
) -> list[str]:
    """The columns a model is allowed to see.

    Both flags default to False so that every call written before Phase 4
    keeps returning exactly the columns it used to, and the earlier phases
    stay reproducible.

    `include_failure_modes=True` is deliberately available: Phase 3 uses it to
    reproduce the leakage experiment. It must never be used anywhere else.

    `include_derived=True` requires the frame to have been passed through
    `add_physical_features` first.
    """
    columns = CATEGORICAL_COLUMNS + SENSOR_COLUMNS
    if include_derived:
        columns = columns + DERIVED_COLUMNS
    if include_failure_modes:
        columns = columns + FAILURE_MODES
    return columns


def make_stress_variant(
    df: pd.DataFrame,
    n_positives: int = 50,
    seed: int = RANDOM_SEED,
) -> pd.DataFrame:
    """Down-sample the failures to imitate a far rarer-failure factory.

    Every healthy row is kept and only the failures are thinned out, so this
    reads as "same line, same output volume, better machines" rather than as a
    smaller dataset. Row order is preserved so that the ordered-split
    experiment in Phase 3 still sees a meaningful UID sequence.
    """
    positives = df[df[TARGET] == 1]
    if n_positives > len(positives):
        raise ValueError(
            f"asked for {n_positives} failures but only {len(positives)} exist"
        )

    kept_positives = positives.sample(n=n_positives, random_state=seed)
    negatives = df[df[TARGET] == 0]

    stressed = pd.concat([negatives, kept_positives])
    return stressed.sort_values("UID").reset_index(drop=True)
