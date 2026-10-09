"""Part 2: predict whether a product will be ordered in the next quarter.

Run from the project root:
    python src/reorder_model.py
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, recall_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

from data import CUTOFF, build_reorder_table, load_orders
from style import BLUE, INK, MUTED, comparison_bars

ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = ROOT / "reports" / "figures"
SEED = 42
ORIGINAL = "Decision Tree (original features)"

CATEGORICAL = ["category", "sub_category", "main_ship_mode"]
NUMERIC = [
    "n_orders", "n_customers", "n_regions", "total_quantity", "total_sales",
    "total_profit", "avg_discount", "avg_unit_price", "days_since_last_order",
    "days_since_first_order", "orders_last_365_days",
]


def build_models() -> dict:
    full = ColumnTransformer(
        [
            ("num", StandardScaler(), NUMERIC),
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),
        ]
    )
    # In the RapidMiner version, sales, profit and discount were collapsed to
    # true/false and became almost constant, so the tree in practice only had
    # ship mode and sub-category to work with. This replica uses just those two.
    original = ColumnTransformer(
        [("cat", OneHotEncoder(handle_unknown="ignore"), ["main_ship_mode", "sub_category"])]
    )
    return {
        "Always predict majority": DummyClassifier(strategy="most_frequent"),
        ORIGINAL: Pipeline(
            [("prep", original),
             ("model", DecisionTreeClassifier(max_depth=5, min_samples_leaf=10, random_state=SEED))]
        ),
        "Decision Tree": Pipeline(
            [("prep", full),
             ("model", DecisionTreeClassifier(max_depth=5, min_samples_leaf=10, random_state=SEED))]
        ),
        "Logistic Regression": Pipeline(
            [("prep", full), ("model", LogisticRegression(max_iter=2000))]
        ),
        "Random Forest": Pipeline(
            [("prep", full),
             ("model", RandomForestClassifier(n_estimators=400, min_samples_leaf=5,
                                              random_state=SEED, n_jobs=-1))]
        ),
    }


def plot_accuracy(results: dict) -> None:
    names = list(results)
    acc = [results[n]["test_accuracy"] * 100 for n in names]
    fig, ax = plt.subplots(figsize=(6.8, 3.6))
    comparison_bars(ax, names, acc, ORIGINAL, "%.1f%%")
    ax.set_xlim(0, 100)
    ax.set_xlabel("Test accuracy (%)")
    ax.set_title("Will the product sell next quarter?")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "reorder_model_comparison.png")
    plt.close(fig)


def plot_rate_by_history(table: pd.DataFrame) -> None:
    """The one pattern in the data: products that sold more often before are a bit likelier to sell again."""
    bands = pd.cut(table["n_orders"], [0, 2, 4, 6, 8, 100],
                   labels=["1-2", "3-4", "5-6", "7-8", "9+"])
    rate = table.groupby(bands, observed=True)["reordered"].agg(["mean", "size"])
    ticks = [f"{band}\n({int(n)} products)" for band, n in zip(rate.index, rate["size"])]
    fig, ax = plt.subplots(figsize=(6.8, 3.9))
    bars = ax.bar(ticks, rate["mean"] * 100, color=BLUE, width=0.55)
    ax.bar_label(bars, fmt="%.0f%%", padding=3, color=INK)
    overall = table["reordered"].mean() * 100
    ax.axhline(overall, color=MUTED, lw=1, ls="--", zorder=0)
    ax.set_ylim(0, 75)
    ax.set_xlabel(f"Orders before {CUTOFF:%b %Y}  ·  dashed line = all products ({overall:.0f}%)")
    ax.set_ylabel("Ordered next quarter (%)")
    ax.set_title("Past popularity barely predicts the next quarter")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "reorder_rate_by_history.png")
    plt.close(fig)


def main() -> dict:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    table = build_reorder_table(load_orders())
    X, y = table[NUMERIC + CATEGORICAL], table["reordered"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=SEED
    )

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    results, fitted = {}, {}
    for name, model in build_models().items():
        print(f"Training {name} ...")
        cv_res = cross_validate(model, X_train, y_train, cv=cv, scoring=["roc_auc", "accuracy"])
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        proba = model.predict_proba(X_test)[:, 1]
        results[name] = {
            "cv_roc_auc_mean": round(float(cv_res["test_roc_auc"].mean()), 4),
            "cv_roc_auc_std": round(float(cv_res["test_roc_auc"].std()), 4),
            "test_accuracy": round(float(accuracy_score(y_test, pred)), 4),
            "test_roc_auc": round(float(roc_auc_score(y_test, proba)), 4),
            "test_recall": round(float(recall_score(y_test, pred)), 4),
            "test_f1": round(float(f1_score(y_test, pred, zero_division=0)), 4),
        }
        fitted[name] = model

    trained = [n for n in results if n != "Always predict majority"]
    best_name = max(trained, key=lambda n: results[n]["cv_roc_auc_mean"])

    plot_accuracy(results)
    plot_rate_by_history(table)

    print()
    print(pd.DataFrame(results).T.to_string())
    print(f"\nBest model (by cross-validated ROC-AUC): {best_name}")
    return {
        "best_model": best_name,
        "cutoff_date": str(CUTOFF.date()),
        "n_products": len(table),
        "reorder_rate": round(float(y.mean()), 4),
        "n_train": len(X_train),
        "n_test": len(X_test),
        "models": results,
    }


if __name__ == "__main__":
    out = main()
    (ROOT / "reports" / "metrics_reorder.json").write_text(json.dumps(out, indent=2))
