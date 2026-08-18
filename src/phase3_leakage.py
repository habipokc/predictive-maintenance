"""Phase 3 - two demonstrations of data leakage.

Leakage means the model was fed information it will NOT have at prediction
time. The score that comes out is real arithmetic on real rows, but it is a
measurement of the wrong thing, so it does not survive contact with the shop
floor. This phase produces the number that proves it, twice.

    Experiment A - column leakage
        TWF/HDF/PWF/OSF/RNF describe WHY a machine failed. They are written
        after the breakdown, and `Machine failure` is essentially their union.
        Handing them to the model as inputs is handing it the answer key.

    Experiment B - split leakage
        `Tool wear` is cumulative: it climbs while a tool is used and drops
        back to zero when the tool is replaced. That means the rows carry an
        implicit time order. A random split scatters neighbouring rows from
        the same tool life across train and test, so the model can recognise
        a neighbourhood instead of learning the physics. Splitting by row
        order instead simulates the only thing that ever happens in
        production: train on the past, predict the future.

The model is deliberately held constant across every run so that the
difference in the scores can only be attributed to the leakage, never to a
different model.

Run with:  python -m src.phase3_leakage
"""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import confusion_matrix, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

from src.config import FIGURES_DIR, RANDOM_SEED, ensure_dirs, set_global_seed
from src.data import TARGET, feature_columns, load_raw

RULE = "=" * 74
TEST_SIZE = 0.2

# `Type` is a product quality grade, so its three values have a genuine order
# (low -> medium -> high). Mapping it to 0/1/2 keeps that order and costs one
# column; one-hot encoding would add three columns and throw the order away.
TYPE_ORDER = {"L": 0, "M": 1, "H": 2}

# Depth budget, not a magic number. The published set holds 339 failures, of
# which roughly 271 land in training. Depth 5 allows at most 32 leaves, so a
# leaf holds ~8 failures on average. Going deeper drops that to two or three
# failures per leaf, which is noise being fitted rather than a pattern.
MAX_DEPTH = 5


def encode(X: pd.DataFrame) -> pd.DataFrame:
    """Turn the one categorical column into numbers scikit-learn can consume."""
    X = X.copy()
    X["Type"] = X["Type"].map(TYPE_ORDER)
    return X


def fit_and_score(X_train, X_test, y_train, y_test, label):
    """Train one tree and report what it caught, missed and paid attention to.

    Accuracy is not reported anywhere in this phase. Phase 2 already showed it
    rewards a model that catches nothing, so it cannot referee this comparison.
    """
    model = DecisionTreeClassifier(max_depth=MAX_DEPTH, random_state=RANDOM_SEED)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    tn, fp, fn, tp = confusion_matrix(y_test, predictions, labels=[0, 1]).ravel()

    importances = pd.Series(
        model.feature_importances_, index=X_train.columns
    ).sort_values(ascending=False)

    return {
        "label": label,
        "precision": precision_score(y_test, predictions, zero_division=0),
        "recall": recall_score(y_test, predictions, zero_division=0),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "importances": importances,
    }


def print_result(result):
    print(f"\n  --- {result['label']} ---")
    print(f"  precision {result['precision']:.4f}")
    print(f"  recall    {result['recall']:.4f}")
    print(
        f"  confusion  TN {result['tn']:>5}  FP {result['fp']:>4}  "
        f"FN {result['fn']:>4}  TP {result['tp']:>4}"
    )
    print(
        f"  plain words: of {result['tp'] + result['fn']} real failures in the "
        f"test set, this model caught {result['tp']} and missed {result['fn']}."
    )
    print("  where the model spent its attention:")
    for name, share in result["importances"].head(5).items():
        if share > 0:
            print(f"      {name:<22} {share:6.1%}")


def random_split(df, columns):
    """The split used so far: shuffle everything, keep the failure ratio equal."""
    X = encode(df[columns])
    y = df[TARGET]
    return train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_SEED
    )


def ordered_split(df, columns):
    """Split by row order instead: the first 80% of UIDs train, the last 20% test.

    There is no `stratify` here and there cannot be. Stratifying would require
    looking at the labels of future rows in order to decide the split, which is
    exactly the kind of peeking this experiment exists to avoid.
    """
    ordered = df.sort_values("UID")
    cut = int(len(ordered) * (1 - TEST_SIZE))

    train, test = ordered.iloc[:cut], ordered.iloc[cut:]
    return (
        encode(train[columns]),
        encode(test[columns]),
        train[TARGET],
        test[TARGET],
    )


def report_order_evidence(df):
    """Check that the rows really do carry an implicit order before relying on it.

    If `Tool wear` were unordered noise it would go up about as often as it
    goes down. If it is a cumulative counter that resets on tool changes, it
    should climb nearly every row and fall only at the resets.
    """
    ordered = df.sort_values("UID")
    delta = ordered["Tool wear"].diff().dropna()

    rises = int((delta > 0).sum())
    falls = int((delta < 0).sum())
    flats = int((delta == 0).sum())

    print(f"\n{RULE}")
    print("IS THERE AN ORDER IN THESE ROWS AT ALL?")
    print(RULE)
    print(f"  Tool wear rises from one row to the next : {rises:>6,}")
    print(f"  Tool wear falls (tool replaced)          : {falls:>6,}")
    print(f"  Tool wear unchanged                      : {flats:>6,}")
    print(
        f"  -> it climbs on {rises / len(delta):.1%} of steps, so the rows are not "
        "independent draws;\n     consecutive rows belong to the same tool life."
    )


def run_column_leakage(df):
    print(f"\n{RULE}")
    print("EXPERIMENT A - COLUMN LEAKAGE")
    print(RULE)
    print("  Same model, same split, same rows. The only change is WHICH")
    print("  columns the model is allowed to look at.")

    leaked_columns = feature_columns(include_failure_modes=True)
    clean_columns = feature_columns()

    leaked = fit_and_score(
        *random_split(df, leaked_columns),
        "with TWF/HDF/PWF/OSF/RNF as features (LEAKED)",
    )
    clean = fit_and_score(
        *random_split(df, clean_columns),
        "sensor columns only (honest)",
    )

    print_result(leaked)
    print_result(clean)

    print(
        f"\n  recall drops {leaked['recall']:.4f} -> {clean['recall']:.4f} "
        f"({clean['recall'] - leaked['recall']:+.4f}) the moment the answer key is removed."
    )
    return leaked, clean


def run_split_leakage(df):
    print(f"\n{RULE}")
    print("EXPERIMENT B - SPLIT LEAKAGE")
    print(RULE)
    print("  Same model, same honest columns. The only change is HOW the rows")
    print("  are divided into training and test.")

    columns = feature_columns()

    random_result = fit_and_score(
        *random_split(df, columns), "random split (rows shuffled)"
    )
    ordered_result = fit_and_score(
        *ordered_split(df, columns), "ordered split (last 20% of UIDs = test)"
    )

    # The two test sets are not the same rows, so their failure counts differ.
    # That difference is itself part of the lesson and must be visible.
    ordered = df.sort_values("UID")
    cut = int(len(ordered) * (1 - TEST_SIZE))
    print(
        f"\n  failures in the random  test set: "
        f"{random_result['tp'] + random_result['fn']:>4}"
    )
    print(
        f"  failures in the ordered test set: "
        f"{ordered_result['tp'] + ordered_result['fn']:>4}"
        f"   (train part holds {int(ordered.iloc[:cut][TARGET].sum())})"
    )

    print_result(random_result)
    print_result(ordered_result)

    print(
        f"\n  recall {random_result['recall']:.4f} -> {ordered_result['recall']:.4f} "
        f"({ordered_result['recall'] - random_result['recall']:+.4f}) when the model is "
        "forced to predict rows it could not have seen neighbours of."
    )
    return random_result, ordered_result


def plot_leakage(leaked, clean, ordered_result):
    """One figure: what the answer key was worth, and what shuffling was worth."""
    results = [leaked, clean, ordered_result]
    labels = [
        "A: failure-mode\ncolumns leaked",
        "B: sensors only\nrandom split",
        "B: sensors only\nordered split",
    ]
    colors = ["#c44e52", "#4c72b0", "#55a868"]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    bars = ax.bar(labels, [r["recall"] for r in results], color=colors, width=0.55)

    for bar, result in zip(bars, results):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.02,
            f"recall {result['recall']:.2f}\n"
            f"{result['tp']}/{result['tp'] + result['fn']} caught",
            ha="center",
            va="bottom",
        )

    ax.set_ylim(0, 1.2)
    ax.set_ylabel("recall (share of real failures caught)")
    ax.set_title(
        f"DecisionTree(max_depth={MAX_DEPTH}), identical everywhere.\n"
        "Only the information the model was allowed to see changes."
    )

    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "06_leakage.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    set_global_seed()
    ensure_dirs()

    df = load_raw()

    report_order_evidence(df)
    leaked, clean = run_column_leakage(df)
    random_result, ordered_result = run_split_leakage(df)

    print(f"\n{RULE}")
    print("THE HEADLINE")
    print(RULE)
    print(
        f"  With the failure-mode columns in the feature set the model reaches "
        f"recall {leaked['recall']:.2f}."
    )
    print(
        f"  Remove them and the same model reaches {clean['recall']:.2f}. "
        "Nothing about the machine changed;"
    )
    print("  only our honesty about what is knowable at prediction time did.")

    plot_leakage(leaked, clean, ordered_result)
    print(f"\n  Figure written to {FIGURES_DIR / '06_leakage.png'}")


if __name__ == "__main__":
    main()
