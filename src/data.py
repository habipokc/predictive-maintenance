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


def feature_columns(include_failure_modes: bool = False) -> list[str]:
    """The columns a model is allowed to see.

    `include_failure_modes=True` is deliberately available: Phase 3 uses it to
    reproduce the leakage experiment. It must never be used anywhere else.
    """
    columns = CATEGORICAL_COLUMNS + SENSOR_COLUMNS
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
