"""Phase 1 - EDA: what does "imbalanced" actually mean in this dataset?

Nothing here is modelling. The point is to know the shape of the problem
BEFORE the first model, so that Phase 2's dummy baseline is unsurprising
rather than magical.

Run with:  python -m src.phase1_eda
"""

import matplotlib

# Write figures to disk instead of opening windows, so the script behaves the
# same whether it is run from a terminal, an IDE or a future CI job.
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import seaborn as sns

from src.config import FIGURES_DIR, ensure_dirs, set_global_seed
from src.data import (
    FAILURE_MODES,
    SENSOR_COLUMNS,
    TARGET,
    load_raw,
    make_stress_variant,
)

RULE = "=" * 72


def report_class_balance(df, name):
    """Print the imbalance in the two ways an engineer actually thinks about it."""
    n_rows = len(df)
    n_failures = int(df[TARGET].sum())
    rate = n_failures / n_rows
    print(
        f"  {name:<10} {n_rows:>6,} rows | {n_failures:>4} failures | "
        f"{rate:>7.3%} | 1 failure every {n_rows / n_failures:>5,.0f} parts"
    )


def report_label_consistency(df):
    """Cross-check the binary target against the five failure-mode flags.

    A row is supposed to be a failure exactly when at least one mode fired.
    Where that does not hold, the label definition itself is ambiguous - which
    is the single most common source of trouble in real maintenance data.
    """
    modes_fired = df[FAILURE_MODES].sum(axis=1)
    is_failure = df[TARGET] == 1

    print(f"  rows with more than one mode firing : {int((modes_fired > 1).sum())}")
    print(f"  labelled failure but no mode set    : {int((is_failure & (modes_fired == 0)).sum())}")
    print(f"  mode set but not labelled failure   : {int((~is_failure & (modes_fired > 0)).sum())}")


def plot_class_balance(full_df, stress_df):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    for ax, (df, title) in zip(
        axes, [(full_df, "As published"), (stress_df, "Stress variant")]
    ):
        counts = df[TARGET].value_counts().sort_index()
        rate = counts.get(1, 0) / len(df)
        ax.bar(["healthy", "failure"], counts.values, color=["#4c72b0", "#c44e52"])
        ax.set_title(f"{title}\nfailure rate = {rate:.2%}")
        ax.set_ylabel("rows")
        for i, value in enumerate(counts.values):
            ax.text(i, value, f"{value:,}", ha="center", va="bottom")

    fig.suptitle("Class balance in the two regimes we will report on")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "01_class_balance.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_failure_modes(df):
    counts = df[FAILURE_MODES].sum().sort_values(ascending=False)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(counts.index, counts.values, color="#c44e52")
    ax.set_ylabel("rows where the mode fired")
    ax.set_title("How often each failure mode occurs (10,000 rows)")
    for i, value in enumerate(counts.values):
        ax.text(i, value, str(int(value)), ha="center", va="bottom")

    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "02_failure_modes.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_sensor_distributions(df):
    """Healthy vs failed for every raw sensor column.

    If a variable separates the two groups on its own, it will show up here as
    two boxes that barely overlap. If none of them do, the signal lives in the
    COMBINATION of variables - which is exactly the argument for Phase 4.
    """
    fig, axes = plt.subplots(1, len(SENSOR_COLUMNS), figsize=(18, 4))
    labelled = df.assign(status=df[TARGET].map({0: "healthy", 1: "failure"}))

    for ax, column in zip(axes, SENSOR_COLUMNS):
        sns.boxplot(
            data=labelled,
            x="status",
            y=column,
            hue="status",
            order=["healthy", "failure"],
            palette={"healthy": "#4c72b0", "failure": "#c44e52"},
            legend=False,
            ax=ax,
        )
        ax.set_title(column)
        ax.set_xlabel("")

    fig.suptitle("Raw sensor readings: healthy vs failed")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "03_sensor_distributions.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_correlations(df):
    numeric = df[SENSOR_COLUMNS + [TARGET]]

    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(
        numeric.corr(),
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        center=0,
        square=True,
        ax=ax,
    )
    ax.set_title("Linear correlation (Pearson) between raw columns")

    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "04_correlation.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    set_global_seed()
    ensure_dirs()
    sns.set_theme(style="whitegrid")

    df = load_raw()
    stress_df = make_stress_variant(df)

    print(RULE)
    print("SHAPE AND COMPLETENESS")
    print(RULE)
    print(f"  rows x columns: {df.shape[0]:,} x {df.shape[1]}")
    print(f"  missing values in the whole table: {int(df.isna().sum().sum())}")
    print(f"  duplicated UIDs: {int(df['UID'].duplicated().sum())}")

    print(f"\n{RULE}")
    print("CLASS BALANCE - THE NUMBER THAT DECIDES EVERY LATER METRIC CHOICE")
    print(RULE)
    report_class_balance(df, "published")
    report_class_balance(stress_df, "stress")

    print(f"\n{RULE}")
    print("FAILURE MODES")
    print(RULE)
    for mode in FAILURE_MODES:
        count = int(df[mode].sum())
        print(f"  {mode}: {count:>4}  ({count / len(df):.2%} of all rows)")
    print(f"  sum of modes = {int(df[FAILURE_MODES].sum().sum())} "
          f"vs {int(df[TARGET].sum())} labelled failures")
    print()
    report_label_consistency(df)

    print(f"\n{RULE}")
    print("FAILURE RATE BY PRODUCT QUALITY CLASS")
    print(RULE)
    by_type = df.groupby("Type")[TARGET].agg(["count", "sum", "mean"])
    by_type.columns = ["rows", "failures", "rate"]
    for type_name, row in by_type.iterrows():
        print(f"  Type {type_name}: {int(row['rows']):>5,} rows | "
              f"{int(row['failures']):>3} failures | {row['rate']:.2%}")

    print(f"\n{RULE}")
    print("SENSOR READINGS: HEALTHY vs FAILED (means)")
    print(RULE)
    means = df.groupby(TARGET)[SENSOR_COLUMNS].mean().T
    means.columns = ["healthy", "failure"]
    means["difference"] = means["failure"] - means["healthy"]
    print(means.round(2).to_string())

    plot_class_balance(df, stress_df)
    plot_failure_modes(df)
    plot_sensor_distributions(df)
    plot_correlations(df)

    print(f"\n{RULE}")
    print(f"Figures written to {FIGURES_DIR}")
    print(RULE)


if __name__ == "__main__":
    main()
