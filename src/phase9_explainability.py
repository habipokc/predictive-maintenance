"""Phase 9 - SHAP, so the output stops being a verdict and becomes an instruction.

`feature_importances_` answers one question: across the whole dataset, which
columns did the trees find useful? That is a property of the MODEL. It cannot
tell a maintenance crew anything about the machine in front of them, because
it is the same answer for every row.

SHAP answers a different question, one row at a time: for THIS part, how much
did each column push the prediction up or down, relative to the average
prediction? Those per-row contributions add up exactly to the prediction
itself, which is what makes them usable as an explanation rather than a
ranking.

    global (beeswarm)   every row plotted, so the shape of each column's
                        influence is visible - not just how much it matters
                        but in which direction and at which values
    local (waterfall)   one prediction taken apart into its contributions

The framing that matters: this output does not say "machine 4212 will fail".
It says "torque and tool wear together pushed this part over the line - check
the cutting tool". A prediction is something to be believed or disbelieved; an
instruction is something to be acted on.

Run with:  python -m src.phase9_explainability
"""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import shap
from sklearn.ensemble import RandomForestClassifier

from src.config import FIGURES_DIR, RANDOM_SEED, ensure_dirs, set_global_seed
from src.data import add_physical_features, feature_columns, load_raw
from src.phase3_leakage import random_split
from src.phase5_models import N_ESTIMATORS

RULE = "=" * 74

# Exact SHAP values for a 300-tree forest cost time proportional to the number
# of rows explained. 1500 rows is enough for the beeswarm to show the shape of
# every column and still finishes in seconds. The waterfall uses a single row,
# so it is exact regardless.
N_EXPLAIN = 1500


def fit_model(df):
    X_train, X_test, y_train, y_test = random_split(
        df, feature_columns(include_derived=True)
    )
    model = RandomForestClassifier(
        n_estimators=N_ESTIMATORS, random_state=RANDOM_SEED, n_jobs=-1
    )
    model.fit(X_train, y_train)
    return model, X_test, y_test


def explain(model, X):
    """SHAP values for the failure class."""
    sample = X.sample(n=min(N_EXPLAIN, len(X)), random_state=RANDOM_SEED)

    explainer = shap.TreeExplainer(model)
    explanation = explainer(sample)

    # A binary sklearn forest returns one column of values per class. We only
    # ever reason about the probability of failure, so keep class 1.
    if explanation.values.ndim == 3:
        explanation = explanation[..., 1]

    return explanation, sample


def report_global(explanation):
    """Rank the columns by how much they move predictions, on average."""
    mean_impact = np.abs(explanation.values).mean(axis=0)
    order = np.argsort(mean_impact)[::-1]

    print(f"\n{RULE}")
    print("GLOBAL - mean |SHAP| per column")
    print(RULE)
    print("  How much each column moves a prediction, averaged over rows.")
    print("  Unlike feature_importances_, this is measured in the units of the")
    print("  model's output, so the numbers are comparable to each other.\n")

    for index in order:
        name = explanation.feature_names[index]
        print(f"    {name:<22}{mean_impact[index]:>8.4f}")

    return [explanation.feature_names[i] for i in order]


def report_local(model, explanation, sample, y_test):
    """Take one real failure apart, contribution by contribution."""
    probabilities = model.predict_proba(sample)[:, 1]

    # Pick a genuine failure the model was confident about - a clear case
    # makes the anatomy of the explanation readable.
    labels = y_test.loc[sample.index]
    failures = np.where(labels.to_numpy() == 1)[0]
    chosen = int(failures[np.argmax(probabilities[failures])])

    row_id = sample.index[chosen]
    print(f"\n{RULE}")
    print(f"LOCAL - one prediction, row {row_id}")
    print(RULE)
    print(
        f"  true label: FAILURE | model probability: {probabilities[chosen]:.3f}"
        f" | base rate: {explanation.base_values[chosen]:.3f}"
    )
    print("\n  what the sensors read:")
    for name in sample.columns:
        print(f"    {name:<22}{sample.iloc[chosen][name]:>12.2f}")

    print("\n  what pushed the prediction, largest first:")
    contributions = explanation.values[chosen]
    for index in np.argsort(np.abs(contributions))[::-1]:
        name = sample.columns[index]
        direction = "towards FAILURE" if contributions[index] > 0 else "towards healthy"
        print(f"    {name:<22}{contributions[index]:>+9.4f}   {direction}")

    total = explanation.base_values[chosen] + contributions.sum()
    print(
        f"\n  base {explanation.base_values[chosen]:.3f} + contributions "
        f"{contributions.sum():+.3f} = {total:.3f}  (the model's output)"
    )
    print("  The parts add up to the whole. That is the property that makes")
    print("  SHAP an explanation rather than an opinion.")

    return chosen, row_id


def plot_global(explanation):
    plt.figure()
    shap.plots.beeswarm(
        explanation, show=False, max_display=len(explanation.feature_names)
    )
    plt.title("Every row explained\ncolour = the value of that column for that row")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "12_shap_summary.png", dpi=150, bbox_inches="tight")
    plt.close()


def plot_local(explanation, chosen, row_id):
    plt.figure()
    shap.plots.waterfall(explanation[chosen], show=False)
    plt.title(f"One failure prediction taken apart - row {row_id}")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "13_shap_waterfall.png", dpi=150, bbox_inches="tight")
    plt.close()


def main():
    set_global_seed()
    ensure_dirs()

    df = add_physical_features(load_raw())
    model, X_test, y_test = fit_model(df)

    explanation, sample = explain(model, X_test)
    report_global(explanation)
    chosen, row_id = report_local(model, explanation, sample, y_test)

    plot_global(explanation)
    plot_local(explanation, chosen, row_id)
    print(f"\n  Figures written to {FIGURES_DIR / '12_shap_summary.png'}")
    print(f"                and {FIGURES_DIR / '13_shap_waterfall.png'}")


if __name__ == "__main__":
    main()
