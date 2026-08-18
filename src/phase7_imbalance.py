"""Phase 7 - imbalance handling, and whether the probabilities mean anything.

Two questions, both of which Phase 8 depends on.

FIRST: does fighting the imbalance help? Three treatments, measured rather
than assumed:

    none                the Phase 5/6 model, untouched
    class_weight        every failure is counted as if it were many rows, so
                        the fit stops being able to ignore them
    SMOTE               invent synthetic failures by interpolating between
                        real ones until the classes are balanced

SMOTE deserves suspicion in this project specifically. It draws a straight
line between two real failure rows and picks a point on it. In a physical
feature space that point may be a machine state that cannot exist: this
dataset's rotational speed and torque are correlated at -0.88 because the
spindle runs at roughly constant power, so a row interpolated between a
high-torque/low-speed failure and a low-torque/high-speed failure lands on a
torque-speed combination the machine never produces. The metric improves on
paper; the model has learned a region of space the shop floor never visits.

SECOND: is 0.73 actually a 73% chance of failure? Phase 8 chooses a threshold
by cost, and a threshold is a cut on the probability axis. If the axis is not
a probability axis, the cut is arbitrary. A reliability diagram answers this:
bucket the predictions, and in each bucket compare the average predicted
probability against the fraction that actually failed. On the diagonal means
calibrated.

Everything is measured with the same 5-fold stratified CV as Phase 5, and the
resampling lives INSIDE the pipeline so it only ever touches training folds.
Applying SMOTE before splitting would put synthetic copies of test rows into
training - Phase 3's lesson, in a form that is easy to miss.

Run with:  python -m src.phase7_imbalance
"""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import brier_score_loss
from sklearn.model_selection import StratifiedKFold, cross_val_predict, cross_validate

from src.config import FIGURES_DIR, RANDOM_SEED, ensure_dirs, set_global_seed
from src.data import (
    TARGET,
    add_physical_features,
    feature_columns,
    load_raw,
    make_stress_variant,
)
from src.phase3_leakage import encode
from src.phase5_models import N_ESTIMATORS, N_SPLITS, SCORING

RULE = "=" * 78

# Ten buckets over [0, 1]. Fewer would hide the shape of the curve; more would
# leave single-digit counts per bucket on a dataset with 339 failures, and a
# bucket holding four rows cannot estimate a failure rate.
N_CALIBRATION_BINS = 10


def forest(**kwargs):
    return RandomForestClassifier(
        n_estimators=N_ESTIMATORS, random_state=RANDOM_SEED, n_jobs=-1, **kwargs
    )


def build_strategies():
    """The three treatments, all wrapping the same forest."""
    return {
        "none": forest(),
        # "balanced" sets each class's weight to n_samples / (n_classes * count),
        # so a failure carries the weight of many healthy rows. Nothing is
        # invented; the existing rows are simply counted differently when the
        # quality of a split is scored.
        "class_weight": forest(class_weight="balanced"),
        # SMOTE runs as a pipeline STEP, which is what keeps it honest: during
        # cross-validation it is fitted on the training part of each fold and
        # never sees the held-out rows.
        "SMOTE": ImbPipeline(
            [
                ("smote", SMOTE(random_state=RANDOM_SEED)),
                ("forest", forest()),
            ]
        ),
    }


def compare_strategies(X, y, regime_name):
    folds = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_SEED)

    print(f"\n{RULE}")
    print(f"IMBALANCE TREATMENTS - {regime_name}")
    print(RULE)
    print(f"  {len(X):,} rows | {int(y.sum())} failures | {y.mean():.3%}")
    print(f"\n  {'strategy':<16}{'recall':>16}{'precision':>16}{'PR-AUC':>16}")

    results = {}
    for name, estimator in build_strategies().items():
        scores = cross_validate(estimator, X, y, cv=folds, scoring=SCORING, n_jobs=1)
        results[name] = {
            metric: (scores[f"test_{metric}"].mean(), scores[f"test_{metric}"].std())
            for metric in SCORING
        }
        row = results[name]
        print(
            f"  {name:<16}"
            f"{row['recall'][0]:>9.3f}+-{row['recall'][1]:<5.3f}"
            f"{row['precision'][0]:>9.3f}+-{row['precision'][1]:<5.3f}"
            f"{row['pr_auc'][0]:>9.3f}+-{row['pr_auc'][1]:<5.3f}"
        )

    baseline = results["none"]
    for name in ("class_weight", "SMOTE"):
        recall_gap = results[name]["recall"][0] - baseline["recall"][0]
        pr_gap = results[name]["pr_auc"][0] - baseline["pr_auc"][0]
        spread = max(results[name]["recall"][1], baseline["recall"][1])
        verdict = "REAL" if abs(recall_gap) > spread else "inside the fold spread"
        print(
            f"\n  {name:<13} recall {recall_gap:+.3f} ({verdict}), "
            f"PR-AUC {pr_gap:+.3f}"
        )

    return results


def assess_calibration(X, y, regime_name):
    """Are the probabilities probabilities? Measured out-of-fold, never in-sample."""
    folds = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_SEED)

    candidates = {
        "raw forest": forest(),
        # Isotonic regression learns a monotonic mapping from the model's score
        # to an observed frequency. It needs a decent number of positives to fit
        # that mapping.
        "isotonic": CalibratedClassifierCV(forest(), method="isotonic", cv=3),
        # Platt scaling fits a single sigmoid - two parameters - so it survives
        # small positive counts far better than isotonic does.
        "sigmoid": CalibratedClassifierCV(forest(), method="sigmoid", cv=3),
    }

    print(f"\n{RULE}")
    print(f"CALIBRATION - {regime_name}")
    print(RULE)
    print("  Brier score: mean squared error of the probabilities. Lower is better;")
    print("  it rewards being both correct AND honest about confidence.")
    print(f"\n  {'model':<14}{'Brier':>12}{'worst gap':>12}")

    curves = {}
    for name, estimator in candidates.items():
        probabilities = cross_val_predict(
            estimator, X, y, cv=folds, method="predict_proba", n_jobs=1
        )[:, 1]

        observed, predicted = calibration_curve(
            y, probabilities, n_bins=N_CALIBRATION_BINS, strategy="uniform"
        )
        curves[name] = (predicted, observed)

        brier = brier_score_loss(y, probabilities)
        worst_gap = np.max(np.abs(observed - predicted))
        print(f"  {name:<14}{brier:>12.5f}{worst_gap:>12.3f}")

    return curves


def plot_results(strategy_results, curves, regime_name):
    fig, (strategy_ax, calibration_ax) = plt.subplots(1, 2, figsize=(12, 5))

    names = list(strategy_results)
    metrics = ["recall", "precision", "pr_auc"]
    colors = {"recall": "#c44e52", "precision": "#4c72b0", "pr_auc": "#55a868"}
    positions = np.arange(len(names))
    width = 0.26

    for offset, metric in zip([-width, 0, width], metrics):
        strategy_ax.bar(
            positions + offset,
            [strategy_results[name][metric][0] for name in names],
            width,
            yerr=[strategy_results[name][metric][1] for name in names],
            capsize=4,
            label=metric,
            color=colors[metric],
        )
    strategy_ax.set_xticks(positions)
    strategy_ax.set_xticklabels(names)
    strategy_ax.set_ylim(0, 1.05)
    strategy_ax.set_ylabel("score")
    strategy_ax.set_title(f"Imbalance treatments - {regime_name}\n{N_SPLITS}-fold CV")
    strategy_ax.legend()

    calibration_ax.plot(
        [0, 1], [0, 1], color="#999999", linestyle=":", label="perfectly calibrated"
    )
    for name, (predicted, observed) in curves.items():
        calibration_ax.plot(predicted, observed, marker="o", label=name)
    calibration_ax.set_xlabel("predicted probability")
    calibration_ax.set_ylabel("observed failure rate")
    calibration_ax.set_title(
        "Reliability diagram\nabove the line = under-confident, below = over-confident"
    )
    calibration_ax.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(
        FIGURES_DIR / "10_imbalance_calibration.png", dpi=150, bbox_inches="tight"
    )
    plt.close(fig)


def main():
    set_global_seed()
    ensure_dirs()

    raw = load_raw()
    columns = feature_columns(include_derived=True)

    published = add_physical_features(raw)
    stress = add_physical_features(make_stress_variant(raw))

    X_published = encode(published[columns])
    y_published = published[TARGET]
    X_stress = encode(stress[columns])
    y_stress = stress[TARGET]

    published_results = compare_strategies(X_published, y_published, "published (3.39%)")
    compare_strategies(X_stress, y_stress, "stress (0.51%)")

    curves = assess_calibration(X_published, y_published, "published (3.39%)")

    plot_results(published_results, curves, "published")
    print(f"\n  Figure written to {FIGURES_DIR / '10_imbalance_calibration.png'}")


if __name__ == "__main__":
    main()
