"""Phase 4 - features derived from the physics of the machine.

Phase 3 left an honest baseline: the same decision tree, given only the raw
sensor columns, caught 20 of 68 failures. The reason was visible in the AI4I
failure definitions - the machine does not fail on any single reading, it
fails on combinations of them. A tree thresholds one column per split, so a
rule like `process_temp - air_temp < 8.6` costs it several levels of depth to
approximate. This phase hands it those combinations directly.

Two things happen here, in this order:

    1. Verification. Before claiming the derived quantities are the physics,
       we check them row by row against the failure-mode labels. Counting
       matching totals is not enough; the rows themselves have to line up.

    2. Comparison. The Phase 3 estimator and split are IMPORTED rather than
       copied, so the only difference between the baseline and this run is
       three extra columns.

What is deliberately NOT done: none of the documented thresholds (8.6 K,
1380 rpm, 3500-9000 W, 11000-13000 min*Nm) is written into a feature. Encoding
them would be copying the answer key, exactly as Phase 3's leaked columns did.
We derive the quantities and leave the cut points for the model to find.

Run with:  python -m src.phase4_features
"""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from src.config import FIGURES_DIR, ensure_dirs, set_global_seed
from src.data import TARGET, add_physical_features, feature_columns, load_raw
from src.phase3_leakage import MAX_DEPTH, fit_and_score, print_result, random_split

RULE = "=" * 74

# Thresholds quoted from the AI4I 2020 dataset documentation. They appear here
# ONLY to audit the derived columns against the published failure modes. They
# are never handed to the model - see the module docstring.
PWF_MIN_WATT, PWF_MAX_WATT = 3500, 9000
HDF_MAX_TEMP_DIFF, HDF_MAX_SPEED = 8.6, 1380
OSF_THRESHOLD_BY_TYPE = {"L": 11000, "M": 12000, "H": 13000}


def audit(name, rule_rows, label_rows):
    """Compare the rows a physical rule selects against the rows carrying a label.

    Equal counts prove nothing on their own - two different sets of 95 rows
    would also produce 95 and 95. What settles it is the overlap.
    """
    rule, label = set(rule_rows), set(label_rows)
    both = rule & label

    print(f"\n  {name}")
    print(f"    rows the physical rule selects : {len(rule):>4}")
    print(f"    rows carrying the label        : {len(label):>4}")
    print(f"    rows in BOTH                   : {len(both):>4}")
    print(f"    rule fires, label absent       : {len(rule - label):>4}")
    print(f"    label present, rule silent     : {len(label - rule):>4}")

    if rule == label:
        print("    -> identical row sets. The derived column IS this failure mode.")
    elif both:
        print(f"    -> {len(both) / len(rule | label):.1%} agreement over the union.")
    else:
        print("    -> no overlap at all; the rule does not describe this mode.")


def verify_physics(df):
    """Check the three derived quantities against the documented failure modes."""
    print(f"\n{RULE}")
    print("DO THE DERIVED COLUMNS REPRODUCE THE FAILURE MODES?")
    print(RULE)
    print("  Documented thresholds are used here to AUDIT the columns, never")
    print("  to build them. The model below never sees these numbers.")

    outside_power_band = df.index[
        (df["power"] < PWF_MIN_WATT) | (df["power"] > PWF_MAX_WATT)
    ]
    audit(
        "PWF - power outside 3500-9000 W",
        outside_power_band,
        df.index[df["PWF"] == 1],
    )

    overheating = df.index[
        (df["temp_diff"] < HDF_MAX_TEMP_DIFF) & (df["Rotational speed"] < HDF_MAX_SPEED)
    ]
    audit(
        "HDF - temp_diff < 8.6 K AND speed < 1380 rpm",
        overheating,
        df.index[df["HDF"] == 1],
    )

    # OSF's threshold depends on the product grade, so the comparison value
    # differs row by row. A tougher grade tolerates more wear-times-torque.
    osf_limit = df["Type"].map(OSF_THRESHOLD_BY_TYPE)
    overstrained = df.index[df["wear_torque"] > osf_limit]
    audit(
        "OSF - wear_torque above the grade's limit (L 11k / M 12k / H 13k)",
        overstrained,
        df.index[df["OSF"] == 1],
    )


def describe_derived(df):
    """Show that the derived columns separate healthy from failed rows better."""
    print(f"\n{RULE}")
    print("WHAT THE DERIVED COLUMNS LOOK LIKE")
    print(RULE)

    healthy = df[df[TARGET] == 0]
    failed = df[df[TARGET] == 1]

    print(f"  {'column':<14}{'healthy':>12}{'failure':>12}{'gap':>12}")
    for column in ["power", "temp_diff", "wear_torque"]:
        healthy_mean, failed_mean = healthy[column].mean(), failed[column].mean()
        print(
            f"  {column:<14}{healthy_mean:>12.1f}{failed_mean:>12.1f}"
            f"{failed_mean - healthy_mean:>+12.1f}"
        )

    print(f"\n  correlation with `{TARGET}` (Pearson):")
    for column in ["power", "temp_diff", "wear_torque", "Torque", "Tool wear"]:
        print(f"    {column:<14}{df[column].corr(df[TARGET]):>+7.3f}")


def compare(df):
    """The Phase 3 baseline against the same model with three extra columns."""
    print(f"\n{RULE}")
    print("RAW SENSORS vs RAW SENSORS + PHYSICS")
    print(RULE)
    print(f"  Identical DecisionTree(max_depth={MAX_DEPTH}), identical split, identical")
    print("  seed. The only difference is three derived columns.")

    baseline = fit_and_score(
        *random_split(df, feature_columns()),
        "raw sensor columns only (Phase 3 baseline)",
    )
    enriched = fit_and_score(
        *random_split(df, feature_columns(include_derived=True)),
        "raw sensors + power, temp_diff, wear_torque",
    )

    print_result(baseline)
    print_result(enriched)

    caught_extra = enriched["tp"] - baseline["tp"]
    print(
        f"\n  recall    {baseline['recall']:.4f} -> {enriched['recall']:.4f} "
        f"({enriched['recall'] - baseline['recall']:+.4f})"
    )
    print(
        f"  precision {baseline['precision']:.4f} -> {enriched['precision']:.4f} "
        f"({enriched['precision'] - baseline['precision']:+.4f})"
    )
    print(
        f"  in plain words: {caught_extra:+d} more of the same 68 failures are caught, "
        f"and missed failures fall from {baseline['fn']} to {enriched['fn']}."
    )

    derived_share = sum(
        enriched["importances"].get(column, 0.0)
        for column in ["power", "temp_diff", "wear_torque"]
    )
    print(
        f"  the tree now spends {derived_share:.1%} of its attention on the three "
        "columns that did not exist an hour ago."
    )

    return baseline, enriched


def plot_comparison(baseline, enriched):
    """Left: what changed in the scores. Right: where the tree now looks."""
    fig, (score_ax, importance_ax) = plt.subplots(1, 2, figsize=(12, 4.8))

    metrics = ["recall", "precision"]
    positions = range(len(metrics))
    width = 0.35

    baseline_bars = score_ax.bar(
        [p - width / 2 for p in positions],
        [baseline[m] for m in metrics],
        width,
        label="raw sensors",
        color="#4c72b0",
    )
    enriched_bars = score_ax.bar(
        [p + width / 2 for p in positions],
        [enriched[m] for m in metrics],
        width,
        label="+ physics",
        color="#55a868",
    )
    for bars in (baseline_bars, enriched_bars):
        for bar in bars:
            score_ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.015,
                f"{bar.get_height():.3f}",
                ha="center",
                va="bottom",
            )

    score_ax.set_xticks(list(positions))
    score_ax.set_xticklabels(metrics)
    score_ax.set_ylim(0, 1.05)
    score_ax.set_ylabel("score")
    score_ax.set_title("Same tree, same split, three extra columns")
    score_ax.legend()

    importances = enriched["importances"].sort_values()
    colors = [
        "#55a868" if name in ("power", "temp_diff", "wear_torque") else "#bbbbbb"
        for name in importances.index
    ]
    importance_ax.barh(importances.index, importances.values, color=colors)
    importance_ax.set_xlabel("feature importance")
    importance_ax.set_title("Green = derived from physics")

    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "07_physical_features.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    set_global_seed()
    ensure_dirs()

    df = add_physical_features(load_raw())

    verify_physics(df)
    describe_derived(df)
    baseline, enriched = compare(df)

    plot_comparison(baseline, enriched)
    print(f"\n  Figure written to {FIGURES_DIR / '07_physical_features.png'}")


if __name__ == "__main__":
    main()
