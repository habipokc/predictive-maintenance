"""Single source of truth for randomness and paths.

Every place in this project that can behave randomly - the train/test split,
the cross-validation folds, the model initialisation, the resamplers - must
read its seed from here. Never write a literal seed value anywhere else.
"""

import random
from pathlib import Path

import numpy as np

# The VALUE is arbitrary; 42 has no special property. What matters is that a
# single fixed value is shared by every random operation, so that when two
# experiments produce different scores the difference comes from the change we
# made, not from a different shuffle of the data.
RANDOM_SEED = 42

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_RAW = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
FIGURES_DIR = PROJECT_ROOT / "reports" / "figures"


def set_global_seed(seed: int = RANDOM_SEED) -> None:
    """Seed the interpreter-level generators.

    This covers library code that reaches for the global random state instead
    of accepting a random_state argument. It is a safety net, not a substitute
    for passing random_state=RANDOM_SEED explicitly wherever the API allows it.
    """
    random.seed(seed)
    np.random.seed(seed)


def ensure_dirs() -> None:
    """Create the output directories if they do not exist yet."""
    for path in (DATA_RAW, DATA_PROCESSED, FIGURES_DIR):
        path.mkdir(parents=True, exist_ok=True)
