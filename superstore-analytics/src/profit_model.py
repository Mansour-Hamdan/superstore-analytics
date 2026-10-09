"""Part 1: predict the profit of an order line.

Run from the project root:
    python src/profit_model.py
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OrdinalEncoder

from data import load_orders
from style import BLUE, GRAY, INK, MUTED, RED, comparison_bars

ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = ROOT / "reports" / "figures"
SEED = 42
ORIGINAL = "Linear Regression (original)"

NUMERIC = ["Sales", "Quantity", "Discount"]
CATEGORICAL = ["Category", "Sub-Category", "Region", "Segment", "Ship Mode"]


def original_features(X: pd.DataFrame) -> pd.DataFrame:
    """The four inputs the RapidMiner model used."""
    return pd.DataFrame(
        {
            "Sales": X["Sales"],
            "Discount": X["Discount"],
            "Sales_x_Discount": X["Sales"] * X["Discount"],
            "Is_Office_Supplies": (X["Category"] == "Office Supplies").astype(int),
        }
    )


class MarginRegressor(BaseEstimator, RegressorMixin):
    """Predict profit margin (profit / sales), then multiply by sales.

    Margin depends mostly on the product type and the discount, while the dollar
    profit also scales with order size. Learning the margin lets the model reuse
    the same pattern for a $10 order and a $10,000 order.
    """

    def __init__(self, base=None):
        self.base = base

    def fit(self, X, y):
        self.base_ = clone(self.base).fit(X, y / X["Sales"])
        return self

    def predict(self, X):
        return self.base_.predict(X) * X["Sales"].to_numpy()


def build_models() -> dict:
    def forest():
        ordinal = ColumnTransformer(
            [
                ("num", "passthrough", NUMERIC),
                ("cat", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1),
                 CATEGORICAL),
            ]
        )
        return Pipeline(
            [
                ("prep", ordinal),
                ("model", RandomForestRegressor(n_estimators=300, min_samples_leaf=2,
                                                random_state=SEED, n_jobs=-1)),
            ]
        )

    return {
        "Predict the mean": DummyRegressor(),
        ORIGINAL: Pipeline(
            [("features", FunctionTransformer(original_features)), ("model", LinearRegression())]
        ),
        "Random Forest (profit)": forest(),
        "Random Forest (margin)": MarginRegressor(forest()),
    }


def plot_discount_margin(df: pd.DataFrame) -> None:
    bands = pd.cut(df["Discount"], [-0.01, 0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
                   labels=["0%", "≤10%", "≤20%", "≤30%", "≤40%", "≤50%", "≤60%", "≤70%", "≤80%"])
    totals = df.groupby(bands, observed=True)[["Profit", "Sales"]].sum()
    margin = totals["Profit"] / totals["Sales"] * 100
    fig, ax = plt.subplots(figsize=(7, 3.8))
    bars = ax.bar(margin.index.astype(str), margin.values,
                  color=[BLUE if m >= 0 else RED for m in margin.values], width=0.6)
    ax.bar_label(bars, fmt="%.0f%%", padding=3, color=INK)
    ax.axhline(0, color=MUTED, lw=1)
    ax.set_ylabel("Profit margin (%)")
    ax.set_xlabel("Discount on the order line")
    ax.set_title("Profit margin by discount level")
    ax.set_ylim(margin.min() * 1.2, margin.max() * 2)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "profit_by_discount.png")
    plt.close(fig)


def plot_mae(results: dict) -> None:
    names = list(results)
    mae = [results[n]["test_mae"] for n in names]
    fig, ax = plt.subplots(figsize=(6.5, 3.2))
    comparison_bars(ax, names, mae, ORIGINAL, "$%.2f")
    ax.set_xlabel("Average error per order line, in dollars (lower is better)")
    ax.set_title("Profit prediction error on the test set")
    ax.set_xlim(0, max(mae) * 1.2)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "profit_model_comparison.png")
    plt.close(fig)


def plot_predicted_vs_actual(y_true, preds: dict) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(9, 4.3), sharex=True, sharey=True)
    lim = (-1500, 1500)
    for ax, (name, pred), color in zip(axes, preds.items(), [GRAY, BLUE]):
        ax.scatter(y_true, pred, s=8, alpha=0.45, color=color, linewidths=0)
        ax.plot(lim, lim, color=MUTED, lw=1, ls="--")
        ax.set_xlim(lim)
        ax.set_ylim(lim)
        ax.set_title(name, fontsize=11)
        ax.set_xlabel("Actual profit ($)")
    axes[0].set_ylabel("Predicted profit ($)")
    fig.suptitle("Predicted vs actual profit on the test set (dashed line = perfect)",
                 x=0.02, ha="left", fontweight="bold", fontsize=12)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "profit_predicted_vs_actual.png")
    plt.close(fig)


def main() -> dict:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    df = load_orders()
    plot_discount_margin(df)

    X, y = df[NUMERIC + CATEGORICAL], df["Profit"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=SEED)

    cv = KFold(n_splits=5, shuffle=True, random_state=SEED)
    results, preds = {}, {}
    for name, model in build_models().items():
        print(f"Training {name} ...")
        cv_res = cross_validate(model, X_train, y_train, cv=cv,
                                scoring=["neg_mean_absolute_error", "neg_root_mean_squared_error", "r2"])
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        preds[name] = pred
        results[name] = {
            "cv_mae_mean": round(float(-cv_res["test_neg_mean_absolute_error"].mean()), 2),
            "cv_mae_std": round(float(cv_res["test_neg_mean_absolute_error"].std()), 2),
            "cv_rmse_mean": round(float(-cv_res["test_neg_root_mean_squared_error"].mean()), 2),
            "cv_rmse_std": round(float(cv_res["test_neg_root_mean_squared_error"].std()), 2),
            "cv_r2_mean": round(float(cv_res["test_r2"].mean()), 4),
            "test_rmse": round(float(np.sqrt(mean_squared_error(y_test, pred))), 2),
            "test_mae": round(float(mean_absolute_error(y_test, pred)), 2),
            "test_r2": round(float(r2_score(y_test, pred)), 4),
        }

    trained = [n for n in results if n != "Predict the mean"]
    best_name = min(trained, key=lambda n: results[n]["cv_mae_mean"])

    plot_mae(results)
    plot_predicted_vs_actual(y_test, {ORIGINAL: preds[ORIGINAL], best_name: preds[best_name]})

    print()
    print(pd.DataFrame(results).T.to_string())
    print(f"\nBest model (by cross-validated MAE): {best_name}")
    return {"best_model": best_name, "n_train": len(X_train), "n_test": len(X_test),
            "models": results}


if __name__ == "__main__":
    out = main()
    path = ROOT / "reports" / "metrics_profit.json"
    path.write_text(json.dumps(out, indent=2))
