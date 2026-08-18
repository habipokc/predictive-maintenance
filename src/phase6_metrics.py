"""Phase 6 - PR-AUC against ROC-AUC, and why one of them lies.

Both curves are drawn from the same probabilities. The difference is what
each one puts in the denominator.

    recall  = TP / (TP + FN)      out of the real failures, how many caught
    FPR     = FP / (FP + TN)      out of the healthy parts, how many alarmed
    precision = TP / (TP + FP)    out of the alarms raised, how many were real

ROC plots recall against FPR. FPR is divided by the number of HEALTHY rows,
and there are thousands of those, so a flood of false alarms barely moves it.
Precision is divided by the number of ALARMS, and when failures are rare the
alarms are mostly false, so precision feels every one of them.

The consequence shows up in the floor of each curve. A coin-flip model sits at
ROC-AUC 0.5 no matter how rare failures are - the floor never moves. The same
coin-flip sits at PR-AUC equal to the failure rate itself: 0.034 in the
published data, 0.005 in the stress variant. So a PR-AUC of 0.4 is a triumph
in one regime and mediocre in the other, while ROC-AUC gives the same
comforting number in both.

Two regimes are run for exactly this reason:

    published   3.39% failures
    stress      0.515% failures (Phase 1's down-sampled variant)

Model: RandomForest, the Phase 5 winner on PR-AUC and the only rung producing
probabilities fine-grained enough for the threshold sweep in Phase 8.

Run with:  python -m src.phase6_metrics
"""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from src.config import FIGURES_DIR, RANDOM_SEED, ensure_dirs, set_global_seed
from src.data import (
    TARGET,
    add_physical_features,
    feature_columns,
    load_raw,
    make_stress_variant,
)
from src.phase3_leakage import random_split
from src.phase5_models import N_ESTIMATORS

RULE = "=" * 74

# The conventional cut point, used here only as a reference. Phase 8 replaces
# it with one chosen from the cost of being wrong in each direction.
DEFAULT_THRESHOLD = 0.5


def evaluate_regime(df, regime_name):
    """Fit the model once and report both families of number for one regime."""
    X_train, X_test, y_train, y_test = random_split(
        df, feature_columns(include_derived=True)
    )

    model = RandomForestClassifier(
        n_estimators=N_ESTIMATORS, random_state=RANDOM_SEED, n_jobs=-1
    )
    model.fit(X_train, y_train)

    # Column 1 is the probability of the positive class - the failure.
    probabilities = model.predict_proba(X_test)[:, 1]
    predictions = (probabilities >= DEFAULT_THRESHOLD).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_test, predictions, labels=[0, 1]).ravel()
    failure_rate = y_test.mean()

    result = {
        "regime": regime_name,
        "failure_rate": failure_rate,
        "test_rows": len(y_test),
        "test_failures": int(y_test.sum()),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "precision": precision_score(y_test, predictions, zero_division=0),
        "recall": recall_score(y_test, predictions, zero_division=0),
        "f1": f1_score(y_test, predictions, zero_division=0),
        "pr_auc": average_precision_score(y_test, probabilities),
        "roc_auc": roc_auc_score(y_test, probabilities),
        # The floor each metric sits on for a model with no skill at all.
        "pr_floor": failure_rate,
        "roc_floor": 0.5,
    }

    precision_curve, recall_curve, _ = precision_recall_curve(y_test, probabilities)
    fpr, tpr, _ = roc_curve(y_test, probabilities)
    result["pr_curve"] = (recall_curve, precision_curve)
    result["roc_curve"] = (fpr, tpr)

    return result


def print_result(result):
    print(f"\n{RULE}")
    print(f"REGIME: {result['regime']}")
    print(RULE)
    print(
        f"  test set {result['test_rows']:,} rows | {result['test_failures']} failures "
        f"| failure rate {result['failure_rate']:.3%}"
    )
    print(f"\n  at the default threshold {DEFAULT_THRESHOLD}:")
    print(
        f"    TN {result['tn']:>5}   FP {result['fp']:>4}   "
        f"FN {result['fn']:>4}   TP {result['tp']:>4}"
    )
    print(
        f"    of {result['tp'] + result['fn']} real failures, "
        f"{result['tp']} caught and {result['fn']} missed"
    )
    print(
        f"    precision {result['precision']:.4f}   recall {result['recall']:.4f}   "
        f"F1 {result['f1']:.4f}"
    )

    print("\n  threshold-free:")
    print(
        f"    PR-AUC   {result['pr_auc']:.4f}   "
        f"(no-skill floor {result['pr_floor']:.4f}  ->  "
        f"{result['pr_auc'] / result['pr_floor']:.0f}x the floor)"
    )
    print(
        f"    ROC-AUC  {result['roc_auc']:.4f}   "
        f"(no-skill floor {result['roc_floor']:.4f}  ->  "
        f"{result['roc_auc'] / result['roc_floor']:.1f}x the floor)"
    )


def print_headline(published, stress):
    print(f"\n{RULE}")
    print("THE HEADLINE")
    print(RULE)
    print(
        f"  failures get {published['failure_rate'] / stress['failure_rate']:.1f}x rarer:"
        f"  {published['failure_rate']:.3%} -> {stress['failure_rate']:.3%}"
    )
    print(
        f"    ROC-AUC  {published['roc_auc']:.4f} -> {stress['roc_auc']:.4f}   "
        f"({stress['roc_auc'] - published['roc_auc']:+.4f})  barely moves"
    )
    print(
        f"    PR-AUC   {published['pr_auc']:.4f} -> {stress['pr_auc']:.4f}   "
        f"({stress['pr_auc'] - published['pr_auc']:+.4f})"
    )
    print(
        f"    recall   {published['recall']:.4f} -> {stress['recall']:.4f}   "
        f"({stress['recall'] - published['recall']:+.4f})"
    )
    print(
        f"\n  In the stress regime the model misses {stress['fn']} of "
        f"{stress['fn'] + stress['tp']} failures"
    )
    print(f"  while ROC-AUC still reads {stress['roc_auc']:.2f}. Anyone reporting only")
    print("  ROC-AUC would call that a working system.")


def plot_curves(published, stress):
    """Two panels, one message: the same models, two very different stories."""
    fig, (pr_ax, roc_ax) = plt.subplots(1, 2, figsize=(12, 5))

    for result, color in [(published, "#4c72b0"), (stress, "#c44e52")]:
        recall_curve, precision_curve = result["pr_curve"]
        pr_ax.plot(
            recall_curve,
            precision_curve,
            color=color,
            label=f"{result['regime']} - PR-AUC {result['pr_auc']:.3f}",
        )
        pr_ax.axhline(
            result["pr_floor"],
            color=color,
            linestyle=":",
            linewidth=1,
            label=f"no-skill floor {result['pr_floor']:.3f}",
        )

        fpr, tpr = result["roc_curve"]
        roc_ax.plot(
            fpr,
            tpr,
            color=color,
            label=f"{result['regime']} - ROC-AUC {result['roc_auc']:.3f}",
        )

    pr_ax.set_xlabel("recall")
    pr_ax.set_ylabel("precision")
    pr_ax.set_title("Precision-Recall\nthe floor drops with the failure rate")
    pr_ax.set_xlim(0, 1)
    pr_ax.set_ylim(0, 1.02)
    pr_ax.legend(loc="lower left", fontsize=8)

    roc_ax.plot(
        [0, 1],
        [0, 1],
        color="#999999",
        linestyle=":",
        linewidth=1,
        label="no-skill floor 0.500",
    )
    roc_ax.set_xlabel("false positive rate")
    roc_ax.set_ylabel("true positive rate (recall)")
    roc_ax.set_title("ROC\nthe floor never moves, so neither does the verdict")
    roc_ax.set_xlim(0, 1)
    roc_ax.set_ylim(0, 1.02)
    roc_ax.legend(loc="lower right", fontsize=8)

    fig.suptitle(
        "Same model, same features, two failure rates. "
        "ROC-AUC hides what PR-AUC reports.",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "09_pr_vs_roc.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    set_global_seed()
    ensure_dirs()

    raw = load_raw()
    published = add_physical_features(raw)
    stress = add_physical_features(make_stress_variant(raw))

    published_result = evaluate_regime(published, "published (3.39%)")
    stress_result = evaluate_regime(stress, "stress (0.51%)")

    print_result(published_result)
    print_result(stress_result)
    print_headline(published_result, stress_result)

    plot_curves(published_result, stress_result)
    print(f"\n  Figure written to {FIGURES_DIR / '09_pr_vs_roc.png'}")


if __name__ == "__main__":
    main()
