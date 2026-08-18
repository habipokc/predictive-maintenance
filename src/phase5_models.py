"""Phase 5 - the model ladder, evaluated with cross-validation.

Everything up to here was measured on a single 80/20 split. That was fine
while the question was "does this change help?", because the split was held
constant and only one thing moved at a time. It is not fine for "which model
is better", because two models can differ by a couple of points purely
because of which 68 failures happened to land in the test set.

So this phase changes the measuring instrument as well as the model:

    StratifiedKFold(n_splits=5)   every row is tested exactly once, and every
                                  fold keeps the 3.39% failure ratio

Five folds, not ten: 339 failures over 5 folds leaves ~68 failures in each
held-out part, the same size as the test set used in Phases 2-4, so the
numbers stay comparable. Ten folds would leave ~34, and a single failure
would then be worth 3 points of recall.

The rungs, in increasing capacity:

    DecisionTree        the Phase 4 model, carried over as the reference
    LogisticRegression  a linear boundary; needs scaling, the trees do not
    RandomForest        many deep trees, averaged (bagging)
    XGBoost             trees fitted in sequence on each other's errors
    LightGBM            the same idea with a different growth strategy

This is a comparison of model FAMILIES at comparable capacity, not a
hyperparameter search. Every ensemble gets the same budget (300 trees) and the
boosters get the same depth as the reference tree, so that the table answers
"what does a different kind of model buy?" and not "who tuned harder?".
Tuning is deliberately out of scope; Phase 7 and Phase 8 change the decision
rule instead, which on this problem is worth far more.

Run with:  python -m src.phase5_models
"""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

from src.config import FIGURES_DIR, RANDOM_SEED, ensure_dirs, set_global_seed
from src.data import TARGET, add_physical_features, feature_columns, load_raw
from src.phase3_leakage import MAX_DEPTH, encode

RULE = "=" * 78
N_SPLITS = 5

# One budget for every ensemble. 300 trees is past the point where adding more
# stops moving the score and only costs time; using the same number everywhere
# is what makes the rows of the table comparable.
N_ESTIMATORS = 300
LEARNING_RATE = 0.1

SCORING = {
    "recall": "recall",
    "precision": "precision",
    # Threshold-free: the area under the precision-recall curve. It summarises
    # how well the model RANKS failures above healthy parts, independently of
    # where anyone decides to cut. Phase 6 takes this apart properly.
    "pr_auc": "average_precision",
}


def build_models():
    """The ladder. Every model gets the same seed; none gets class weighting.

    Imbalance handling is deliberately absent here. Phase 7 adds it and
    measures what it is worth; mixing it in now would confuse "a stronger
    model helped" with "a different decision rule helped".
    """
    return {
        "DecisionTree": DecisionTreeClassifier(
            max_depth=MAX_DEPTH, random_state=RANDOM_SEED
        ),
        # The only rung that needs scaling: a linear model compares raw
        # magnitudes, and wear_torque runs in the thousands while temp_diff
        # sits around 10. Without standardisation the fit is dominated by
        # whichever column happens to carry the largest numbers.
        "LogisticRegression": make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=1000, random_state=RANDOM_SEED),
        ),
        "RandomForest": RandomForestClassifier(
            n_estimators=N_ESTIMATORS,
            # Depth is left unlimited on purpose. A random forest controls
            # variance by averaging many decorrelated deep trees, so capping
            # depth would remove the mechanism it relies on.
            random_state=RANDOM_SEED,
            n_jobs=-1,
        ),
        "XGBoost": XGBClassifier(
            n_estimators=N_ESTIMATORS,
            max_depth=MAX_DEPTH,
            learning_rate=LEARNING_RATE,
            eval_metric="logloss",
            random_state=RANDOM_SEED,
            n_jobs=-1,
        ),
        "LightGBM": LGBMClassifier(
            n_estimators=N_ESTIMATORS,
            max_depth=MAX_DEPTH,
            learning_rate=LEARNING_RATE,
            random_state=RANDOM_SEED,
            n_jobs=-1,
            verbose=-1,
        ),
    }


def evaluate(models, X, y):
    """Run every model through the same folds and collect mean and spread."""
    folds = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_SEED)

    results = {}
    for name, model in models.items():
        scores = cross_validate(model, X, y, cv=folds, scoring=SCORING, n_jobs=1)
        results[name] = {
            metric: (scores[f"test_{metric}"].mean(), scores[f"test_{metric}"].std())
            for metric in SCORING
        }
        results[name]["fit_seconds"] = scores["fit_time"].mean()
    return results


def print_table(results):
    print(f"\n{RULE}")
    print(f"MODEL LADDER - {N_SPLITS}-fold stratified cross-validation")
    print(RULE)
    print(f"  {'model':<20}{'recall':>16}{'precision':>16}{'PR-AUC':>16}{'fit s':>9}")
    for name, scores in results.items():
        recall_mean, recall_std = scores["recall"]
        precision_mean, precision_std = scores["precision"]
        pr_mean, pr_std = scores["pr_auc"]
        print(
            f"  {name:<20}"
            f"{recall_mean:>9.3f}+-{recall_std:<5.3f}"
            f"{precision_mean:>9.3f}+-{precision_std:<5.3f}"
            f"{pr_mean:>9.3f}+-{pr_std:<5.3f}"
            f"{scores['fit_seconds']:>9.2f}"
        )

    print("\n  +- is the standard deviation across the five folds. A difference")
    print("  smaller than the spread is not a difference.")


def report_gaps(results):
    """Say plainly how much the ladder actually bought over its first rung."""
    reference = results["DecisionTree"]
    best_name = max(results, key=lambda name: results[name]["pr_auc"][0])
    best = results[best_name]

    print(f"\n{RULE}")
    print("WHAT DID THE LADDER BUY?")
    print(RULE)
    print(
        f"  best PR-AUC: {best_name} at {best['pr_auc'][0]:.3f} "
        f"against the reference tree's {reference['pr_auc'][0]:.3f} "
        f"({best['pr_auc'][0] - reference['pr_auc'][0]:+.3f})"
    )
    print(
        f"  recall     : {reference['recall'][0]:.3f} -> {best['recall'][0]:.3f} "
        f"({best['recall'][0] - reference['recall'][0]:+.3f}), "
        f"fold spread {best['recall'][1]:.3f}"
    )
    print(
        f"  cost       : {reference['fit_seconds']:.2f}s -> "
        f"{best['fit_seconds']:.2f}s per fit"
    )
    print(
        "\n  For comparison, Phase 4 moved recall from 0.294 to 0.824 on a single"
        "\n  split by adding three columns. Feature engineering and model choice"
        "\n  are not the same size of lever on this problem."
    )


def plot_ladder(results):
    """One figure: every rung, every metric, with the fold spread drawn on."""
    names = list(results)
    metrics = ["recall", "precision", "pr_auc"]
    colors = {"recall": "#c44e52", "precision": "#4c72b0", "pr_auc": "#55a868"}

    positions = np.arange(len(names))
    width = 0.26

    fig, ax = plt.subplots(figsize=(10, 5))
    for offset, metric in zip([-width, 0, width], metrics):
        means = [results[name][metric][0] for name in names]
        stds = [results[name][metric][1] for name in names]
        ax.bar(
            positions + offset,
            means,
            width,
            yerr=stds,
            capsize=4,
            label=metric,
            color=colors[metric],
        )

    ax.set_xticks(positions)
    ax.set_xticklabels(names, rotation=15, ha="right")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("score")
    ax.set_title(
        f"{N_SPLITS}-fold stratified CV, identical features everywhere.\n"
        "Error bars are the spread across folds."
    )
    ax.legend()

    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "08_model_ladder.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    set_global_seed()
    ensure_dirs()

    df = add_physical_features(load_raw())
    X = encode(df[feature_columns(include_derived=True)])
    y = df[TARGET]

    print(f"  rows {len(X):,} | failures {int(y.sum())} | features {X.shape[1]}")
    print(f"  every model sees exactly these columns: {', '.join(X.columns)}")

    results = evaluate(build_models(), X, y)
    print_table(results)
    report_gaps(results)

    plot_ladder(results)
    print(f"\n  Figure written to {FIGURES_DIR / '08_model_ladder.png'}")


if __name__ == "__main__":
    main()
