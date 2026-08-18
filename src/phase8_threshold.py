"""Phase 8 - choosing the threshold from the cost of being wrong.

Every phase so far has used 0.5 without ever defending it. Nothing about this
problem makes 0.5 correct; it is simply what `predict()` does when nobody
says otherwise. This phase replaces it with a number that can be defended in
a room with the maintenance manager in it.

The cost model, stated explicitly so that anyone can disagree with it:

    C_FN   a failure we did not warn about. The line stops without warning,
           the part in the machine is scrapped, the tool or spindle may be
           damaged, and everything downstream waits.
    C_FP   an alarm on a healthy machine. One planned stop, a technician
           walks over, checks, finds nothing, restarts.

The ABSOLUTE numbers do not matter and are not claimed to be accurate. Only
the RATIO enters the optimisation. The default here is 50:1, which says one
unplanned stoppage costs as much as fifty unnecessary inspections - roughly an
eight-hour line stop against a ten-minute check. A plant that disagrees plugs
in its own ratio, which is exactly the point: this is a business input, not a
modelling constant. The sweep is repeated at 10, 50 and 200 to show how much
the answer moves when the assumption does.

Probabilities come from `cross_val_predict`, so every one of the 10,000 rows
is scored by a model that never saw it. Choosing a threshold on in-sample
probabilities would pick the point that best fits rows the model has already
memorised, which is Phase 3's mistake wearing a different hat.

Run with:  python -m src.phase8_threshold
"""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedKFold, cross_val_predict

from src.config import FIGURES_DIR, RANDOM_SEED, ensure_dirs, set_global_seed
from src.data import TARGET, add_physical_features, feature_columns, load_raw
from src.phase3_leakage import encode
from src.phase5_models import N_ESTIMATORS, N_SPLITS

RULE = "=" * 78

COST_FP = 1
COST_FN = 50
SENSITIVITY_RATIOS = [10, 50, 200]

# 0.01 to 0.99 in 99 steps. Finer than the resolution the probabilities
# themselves carry (300 trees voting gives steps of 1/300), so nothing is lost
# by not going finer.
THRESHOLDS = np.linspace(0.01, 0.99, 99)

DEFAULT_THRESHOLD = 0.5


def out_of_fold_probabilities(X, y):
    """Score every row with a model that never saw it."""
    folds = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_SEED)
    model = RandomForestClassifier(
        n_estimators=N_ESTIMATORS, random_state=RANDOM_SEED, n_jobs=-1
    )
    return cross_val_predict(model, X, y, cv=folds, method="predict_proba", n_jobs=1)[
        :, 1
    ]


def sweep(y, probabilities, cost_fn, cost_fp):
    """Total cost at every candidate threshold."""
    costs, recalls, precisions, counts = [], [], [], []

    for threshold in THRESHOLDS:
        predictions = (probabilities >= threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(y, predictions, labels=[0, 1]).ravel()

        costs.append(fn * cost_fn + fp * cost_fp)
        recalls.append(tp / (tp + fn) if tp + fn else 0.0)
        precisions.append(tp / (tp + fp) if tp + fp else 0.0)
        counts.append((int(tn), int(fp), int(fn), int(tp)))

    return {
        "cost": np.array(costs),
        "recall": np.array(recalls),
        "precision": np.array(precisions),
        "counts": counts,
    }


def describe_point(label, index, result, cost_fn, cost_fp):
    tn, fp, fn, tp = result["counts"][index]
    threshold = THRESHOLDS[index]
    print(f"\n  {label}")
    print(f"    threshold  {threshold:.2f}")
    print(f"    TN {tn:>5}   FP {fp:>4}   FN {fn:>4}   TP {tp:>4}")
    print(
        f"    recall {result['recall'][index]:.4f}   "
        f"precision {result['precision'][index]:.4f}"
    )
    print(
        f"    cost = {fn} missed x {cost_fn} + {fp} false alarms x {cost_fp} "
        f"= {int(result['cost'][index]):,}"
    )


def report_default_ratio(y, probabilities):
    result = sweep(y, probabilities, COST_FN, COST_FP)

    best = int(np.argmin(result["cost"]))
    default = int(np.argmin(np.abs(THRESHOLDS - DEFAULT_THRESHOLD)))

    print(f"\n{RULE}")
    print(f"COST-OPTIMAL THRESHOLD  (C_FN = {COST_FN}, C_FP = {COST_FP})")
    print(RULE)
    print(f"  one missed failure is assumed to cost as much as {COST_FN // COST_FP}")
    print("  unnecessary inspections. Change that number, change the answer.")

    describe_point("the conventional 0.50:", default, result, COST_FN, COST_FP)
    describe_point("the cost minimum:", best, result, COST_FN, COST_FP)

    saved = result["cost"][default] - result["cost"][best]
    caught = result["counts"][best][3] - result["counts"][default][3]
    extra_alarms = result["counts"][best][1] - result["counts"][default][1]

    print(
        f"\n  moving the threshold {THRESHOLDS[default]:.2f} -> {THRESHOLDS[best]:.2f} "
        f"catches {caught:+d} more failures"
    )
    print(
        f"  at the price of {extra_alarms:+d} extra false alarms, for a net saving "
        f"of {int(saved):,} cost units ({saved / result['cost'][default]:.1%})."
    )
    print("  Nothing about the model changed. Only the decision rule did.")

    # The threshold that minimises expected cost for a perfectly calibrated
    # model. Comparing it to the empirical minimum is a calibration check by
    # another route: if the two agree, the probability axis is behaving.
    analytic = COST_FP / (COST_FP + COST_FN)
    print(
        f"\n  analytic optimum for a calibrated model: "
        f"C_FP / (C_FP + C_FN) = {analytic:.4f}"
    )
    print(
        f"  empirical optimum measured out of fold : {THRESHOLDS[best]:.4f}"
        f"   (gap {THRESHOLDS[best] - analytic:+.4f})"
    )

    return result, best, default


def report_sensitivity(y, probabilities):
    """How far does the answer move when the cost assumption moves?"""
    print(f"\n{RULE}")
    print("SENSITIVITY - the ratio is an assumption, so vary it")
    print(RULE)
    print(
        f"  {'C_FN:C_FP':>10}{'threshold':>12}{'recall':>10}{'precision':>12}"
        f"{'missed':>9}{'false alarms':>15}"
    )

    curves = {}
    for ratio in SENSITIVITY_RATIOS:
        result = sweep(y, probabilities, ratio, COST_FP)
        best = int(np.argmin(result["cost"]))
        tn, fp, fn, tp = result["counts"][best]

        curves[ratio] = (result, best)
        print(
            f"  {ratio:>8}:1{THRESHOLDS[best]:>12.2f}{result['recall'][best]:>10.4f}"
            f"{result['precision'][best]:>12.4f}{fn:>9}{fp:>15}"
        )

    print("\n  The more a stoppage costs relative to an inspection, the lower the")
    print("  threshold and the more willing the system becomes to cry wolf.")
    return curves


def plot_cost_curves(default_result, best_index, default_index, sensitivity):
    fig, (cost_ax, ratio_ax) = plt.subplots(1, 2, figsize=(13, 5))

    cost_ax.plot(THRESHOLDS, default_result["cost"], color="#4c72b0", linewidth=2)
    cost_ax.axvline(
        THRESHOLDS[best_index],
        color="#55a868",
        linestyle="--",
        label=f"cost minimum {THRESHOLDS[best_index]:.2f} "
        f"({int(default_result['cost'][best_index]):,})",
    )
    cost_ax.axvline(
        THRESHOLDS[default_index],
        color="#c44e52",
        linestyle=":",
        label=f"default 0.50 ({int(default_result['cost'][default_index]):,})",
    )
    cost_ax.set_xlabel("threshold")
    cost_ax.set_ylabel(f"total cost  (FN x {COST_FN} + FP x {COST_FP})")
    cost_ax.set_title(
        f"Cost of the decision rule, C_FN:C_FP = {COST_FN}:{COST_FP}\n"
        "the model is identical at every point on this line"
    )
    cost_ax.legend()

    for ratio, (result, best) in sensitivity.items():
        normalised = result["cost"] / result["cost"].max()
        ratio_ax.plot(THRESHOLDS, normalised, label=f"{ratio}:1")
        ratio_ax.axvline(THRESHOLDS[best], linestyle="--", linewidth=1, alpha=0.6)
    ratio_ax.set_xlabel("threshold")
    ratio_ax.set_ylabel("cost, scaled to its own maximum")
    ratio_ax.set_title(
        "Sensitivity to the cost assumption\ndashed lines mark each ratio's optimum"
    )
    ratio_ax.legend(title="C_FN : C_FP")

    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "11_cost_threshold.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    set_global_seed()
    ensure_dirs()

    df = add_physical_features(load_raw())
    X = encode(df[feature_columns(include_derived=True)])
    y = df[TARGET]

    print(f"  {len(X):,} rows | {int(y.sum())} failures | out-of-fold probabilities")

    probabilities = out_of_fold_probabilities(X, y)
    result, best, default = report_default_ratio(y, probabilities)
    sensitivity = report_sensitivity(y, probabilities)

    plot_cost_curves(result, best, default, sensitivity)
    print(f"\n  Figure written to {FIGURES_DIR / '11_cost_threshold.png'}")


if __name__ == "__main__":
    main()
