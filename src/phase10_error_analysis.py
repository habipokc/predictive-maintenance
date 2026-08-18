"""Phase 10 - what the system still gets wrong, broken down by failure mode.

An aggregate recall of 0.94 says nothing about whether the remaining 6% is a
gap worth closing. Splitting the misses by failure mode turns one number into
five, and the five have completely different meanings:

    PWF/HDF/OSF   deterministic modes with a physical trigger. Phase 4 gave
                  the model a column for each of them. A miss here is a real
                  gap and would be worth engineering effort.
    TWF           the tool fails at a randomly chosen wear time inside a
                  200-240 minute window. The window is learnable; which row
                  inside it fails is a coin toss.
    RNF           "random failure" - generated independently of every sensor
                  reading. Unlearnable by construction, and on top of that
                  barely present in the label at all.

Being able to point at the part of the error that is irreducible, and to
decline to chase it, is the difference between a model that is finished and a
model that is still being tuned into a corner.

Both thresholds are reported: the conventional 0.50 and the cost-optimal 0.03
from Phase 8, so the effect of the decision rule on each mode is visible.

Run with:  python -m src.phase10_error_analysis
"""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_predict

from src.config import FIGURES_DIR, RANDOM_SEED, ensure_dirs, set_global_seed
from src.data import (
    FAILURE_MODES,
    TARGET,
    add_physical_features,
    feature_columns,
    load_raw,
)
from src.phase3_leakage import encode
from src.phase5_models import N_ESTIMATORS, N_SPLITS

RULE = "=" * 78

COST_OPTIMAL_THRESHOLD = 0.03
DEFAULT_THRESHOLD = 0.50


def out_of_fold_probabilities(X, y):
    folds = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_SEED)
    model = RandomForestClassifier(
        n_estimators=N_ESTIMATORS, random_state=RANDOM_SEED, n_jobs=-1
    )
    return cross_val_predict(model, X, y, cv=folds, method="predict_proba", n_jobs=1)[
        :, 1
    ]


def recall_by_mode(df, probabilities, threshold):
    """Per-mode recall over the whole dataset, scored out of fold."""
    predicted = probabilities >= threshold
    is_failure = df[TARGET] == 1

    rows = []
    for mode in FAILURE_MODES:
        in_mode = is_failure & (df[mode] == 1)
        total = int(in_mode.sum())
        caught = int((in_mode & predicted).sum())
        rows.append(
            {
                "mode": mode,
                "labelled failures": total,
                "caught": caught,
                "missed": total - caught,
                "recall": caught / total if total else np.nan,
            }
        )

    # Failures whose row carries no mode flag at all - Phase 1 found nine of
    # them. Nothing in the features distinguishes them, by construction.
    no_mode = is_failure & (df[FAILURE_MODES].sum(axis=1) == 0)
    total = int(no_mode.sum())
    caught = int((no_mode & predicted).sum())
    rows.append(
        {
            "mode": "(no mode)",
            "labelled failures": total,
            "caught": caught,
            "missed": total - caught,
            "recall": caught / total if total else np.nan,
        }
    )

    return pd.DataFrame(rows)


def print_table(table, threshold):
    print(f"\n{RULE}")
    print(f"RECALL BY FAILURE MODE - threshold {threshold:.2f}")
    print(RULE)
    print(f"  {'mode':<12}{'failures':>10}{'caught':>9}{'missed':>9}{'recall':>10}")
    for _, row in table.iterrows():
        recall = "     -" if np.isnan(row["recall"]) else f"{row['recall']:.3f}"
        print(
            f"  {row['mode']:<12}{row['labelled failures']:>10}"
            f"{row['caught']:>9}{row['missed']:>9}{recall:>10}"
        )


def report_rnf(df, probabilities):
    """The mode that cannot be learned, and the evidence that it cannot."""
    print(f"\n{RULE}")
    print("RNF - THE PART THAT IS NOT A GAP")
    print(RULE)

    rnf_rows = df[df["RNF"] == 1]
    labelled = int(rnf_rows[TARGET].sum())

    print(f"  rows flagged RNF                     : {len(rnf_rows):>4}")
    print(f"  of those, labelled Machine failure=1 : {labelled:>4}")
    print(
        f"  so RNF contributes {labelled} of the {int(df[TARGET].sum())} "
        "positives in total."
    )

    print("\n  Two separate reasons the model cannot learn RNF:")
    print("    1. By construction it is generated independently of every sensor")
    print("       reading, so no feature carries information about it.")
    print(
        f"    2. The label itself is inconsistent: {len(rnf_rows) - labelled} of the "
        f"{len(rnf_rows)} RNF rows"
    )
    print("       are not even counted as failures.")

    # If RNF were learnable, the model would score those rows above the rest.
    rnf_scores = probabilities[df["RNF"] == 1]
    healthy_scores = probabilities[df[TARGET] == 0]
    print(f"\n  mean out-of-fold score, RNF rows     : {rnf_scores.mean():.4f}")
    print(f"  mean out-of-fold score, healthy rows : {healthy_scores.mean():.4f}")
    print("  The model treats RNF rows like healthy ones, which is the correct")
    print("  behaviour for a signal that does not exist in the inputs.")


def report_false_alarms(df, probabilities, threshold):
    """Are the false alarms random, or are they near-misses?"""
    predicted = probabilities >= threshold
    false_alarms = df[(df[TARGET] == 0) & predicted]
    true_healthy = df[(df[TARGET] == 0) & ~predicted]

    print(f"\n{RULE}")
    print(f"FALSE ALARMS AT THRESHOLD {threshold:.2f} - are they arbitrary?")
    print(RULE)
    print(f"  healthy rows that raised an alarm : {len(false_alarms):>5}")
    print(f"  healthy rows that stayed quiet    : {len(true_healthy):>5}")

    print(f"\n  {'column':<22}{'alarmed':>12}{'quiet':>12}{'gap':>12}")
    for column in ["power", "temp_diff", "wear_torque", "Tool wear", "Torque"]:
        alarmed_mean = false_alarms[column].mean()
        quiet_mean = true_healthy[column].mean()
        print(
            f"  {column:<22}{alarmed_mean:>12.1f}{quiet_mean:>12.1f}"
            f"{alarmed_mean - quiet_mean:>+12.1f}"
        )

    print("\n  The alarmed rows sit closer to the failure regions than the quiet")
    print("  ones do. These are near-misses, not noise - which is what makes the")
    print("  alarms defensible to a maintenance crew even when nothing is found.")


def plot_modes(default_table, optimal_table):
    modes = default_table["mode"].tolist()
    positions = np.arange(len(modes))
    width = 0.38

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(
        positions - width / 2,
        default_table["recall"].fillna(0),
        width,
        label=f"threshold {DEFAULT_THRESHOLD:.2f}",
        color="#4c72b0",
    )
    ax.bar(
        positions + width / 2,
        optimal_table["recall"].fillna(0),
        width,
        label=f"threshold {COST_OPTIMAL_THRESHOLD:.2f} (cost optimal)",
        color="#55a868",
    )

    for position, (_, row) in zip(positions, optimal_table.iterrows()):
        ax.text(
            position,
            1.02,
            f"n={row['labelled failures']}",
            ha="center",
            fontsize=8,
            color="#555555",
        )

    ax.set_xticks(positions)
    ax.set_xticklabels(modes)
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("recall within the mode")
    ax.set_title(
        "Where the remaining error lives\n"
        "TWF is partly random by construction; RNF is entirely so"
    )
    ax.legend(loc="lower left")

    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "14_error_by_mode.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    set_global_seed()
    ensure_dirs()

    df = add_physical_features(load_raw())
    X = encode(df[feature_columns(include_derived=True)])
    y = df[TARGET]

    probabilities = out_of_fold_probabilities(X, y)

    default_table = recall_by_mode(df, probabilities, DEFAULT_THRESHOLD)
    optimal_table = recall_by_mode(df, probabilities, COST_OPTIMAL_THRESHOLD)

    print_table(default_table, DEFAULT_THRESHOLD)
    print_table(optimal_table, COST_OPTIMAL_THRESHOLD)

    report_rnf(df, probabilities)
    report_false_alarms(df, probabilities, COST_OPTIMAL_THRESHOLD)

    plot_modes(default_table, optimal_table)
    print(f"\n  Figure written to {FIGURES_DIR / '14_error_by_mode.png'}")


if __name__ == "__main__":
    main()
