"""Phase 2 - the dummy baseline that makes accuracy look ridiculous.

A DummyClassifier ignores every feature. It is not a model; it is a ruler.
Its only job is to tell us what score a system earns for doing nothing, so
that every number produced later in this project has something to be compared
against.

Two strategies are run on purpose:

    most_frequent : always predict the majority class ("nothing ever breaks")
    stratified    : guess at random, respecting the observed class ratio

The pair matters more than either one alone. `most_frequent` wins on accuracy
while catching zero failures; `stratified` catches a few failures and its
accuracy DROPS. Anyone optimising for accuracy would therefore reject the
model that actually does something.

Run with:  python -m src.phase2_baseline
"""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from sklearn.dummy import DummyClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split

from src.config import FIGURES_DIR, RANDOM_SEED, ensure_dirs, set_global_seed
from src.data import TARGET, feature_columns, load_raw, make_stress_variant

RULE = "=" * 72
TEST_SIZE = 0.2
STRATEGIES = ["most_frequent", "stratified"]


def split(df):
    """Hold out 20% of the rows, keeping the failure ratio identical in both parts.

    `stratify=y` is not cosmetic here. With only 339 failures, an unstratified
    draw can easily hand the test set 60 failures in one run and 78 in the
    next, which moves recall by several points on its own.
    """
    X = df[feature_columns()]
    y = df[TARGET]
    return train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        stratify=y,
        random_state=RANDOM_SEED,
    )


def score_baseline(strategy, X_train, X_test, y_train, y_test):
    """Fit one dummy strategy and return its scores plus its confusion counts."""
    model = DummyClassifier(strategy=strategy, random_state=RANDOM_SEED)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    tn, fp, fn, tp = confusion_matrix(y_test, predictions, labels=[0, 1]).ravel()

    return {
        "strategy": strategy,
        "accuracy": accuracy_score(y_test, predictions),
        # zero_division=0: a model that never predicts "failure" has no
        # predictions to be right or wrong about, so precision is undefined.
        # Reporting it as 0 keeps the table readable and is honest here,
        # because the model is certainly not precise about anything.
        "precision": precision_score(y_test, predictions, zero_division=0),
        "recall": recall_score(y_test, predictions, zero_division=0),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def run_regime(df, regime_name):
    X_train, X_test, y_train, y_test = split(df)

    print(f"\n{RULE}")
    print(f"REGIME: {regime_name}")
    print(RULE)
    print(f"  train rows {len(X_train):>6,} | failures in train {int(y_train.sum()):>4}")
    print(f"  test  rows {len(X_test):>6,} | failures in test  {int(y_test.sum()):>4}")

    results = []
    for strategy in STRATEGIES:
        result = score_baseline(strategy, X_train, X_test, y_train, y_test)
        results.append(result)

        print(f"\n  --- strategy = {strategy} ---")
        print(f"  accuracy  {result['accuracy']:.4f}   ({result['accuracy']:.2%})")
        print(f"  precision {result['precision']:.4f}")
        print(f"  recall    {result['recall']:.4f}")
        print("  confusion matrix, absolute counts:")
        print(f"      true healthy, called healthy (TN): {result['tn']:>5}")
        print(f"      true healthy, called failure (FP): {result['fp']:>5}")
        print(f"      true failure, called healthy (FN): {result['fn']:>5}   <- missed failures")
        print(f"      true failure, called failure (TP): {result['tp']:>5}   <- caught failures")
        print(
            f"  plain words: of {result['tp'] + result['fn']} real failures "
            f"in the test set, this baseline caught {result['tp']}."
        )

    return results


def plot_accuracy_vs_recall(published, stress):
    """One figure, one message: accuracy is high exactly where recall is zero."""
    labels = ["Published\n(3.39% failures)", "Stress\n(0.51% failures)"]
    accuracies = [published[0]["accuracy"], stress[0]["accuracy"]]
    recalls = [published[0]["recall"], stress[0]["recall"]]

    x_positions = range(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(7, 4.5))
    accuracy_bars = ax.bar(
        [x - width / 2 for x in x_positions], accuracies, width,
        label="Accuracy", color="#4c72b0",
    )
    recall_bars = ax.bar(
        [x + width / 2 for x in x_positions], recalls, width,
        label="Recall", color="#c44e52",
    )

    for bars in (accuracy_bars, recall_bars):
        for bar in bars:
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.02,
                f"{bar.get_height():.1%}",
                ha="center", va="bottom",
            )

    ax.set_ylim(0, 1.15)
    ax.set_xticks(list(x_positions))
    ax.set_xticklabels(labels)
    ax.set_ylabel("score")
    ax.set_title(
        'DummyClassifier(strategy="most_frequent")\n'
        'predicts "no failure" for every single part'
    )
    ax.legend()

    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "05_dummy_baseline.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    set_global_seed()
    ensure_dirs()

    df = load_raw()
    stress_df = make_stress_variant(df)

    published_results = run_regime(df, "published (3.39% failures)")
    stress_results = run_regime(stress_df, "stress (0.51% failures)")

    print(f"\n{RULE}")
    print("THE HEADLINE")
    print(RULE)
    for name, results in [("published", published_results), ("stress", stress_results)]:
        most_frequent = results[0]
        print(
            f'  {name:<10} a model that predicts "healthy" for every part scores '
            f"{most_frequent['accuracy']:.2%} accuracy and catches "
            f"{most_frequent['tp']} of {most_frequent['tp'] + most_frequent['fn']} failures."
        )

    print()
    for name, results in [("published", published_results), ("stress", stress_results)]:
        most_frequent, stratified = results
        print(
            f"  {name:<10} random guessing catches {stratified['tp']} failures, "
            f"and its accuracy FALLS from {most_frequent['accuracy']:.2%} "
            f"to {stratified['accuracy']:.2%}."
        )

    plot_accuracy_vs_recall(published_results, stress_results)
    print(f"\n  Figure written to {FIGURES_DIR / '05_dummy_baseline.png'}")


if __name__ == "__main__":
    main()
