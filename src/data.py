"""Loading the Superstore dataset and building the product-level reorder table."""

from pathlib import Path

import pandas as pd

DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "superstore.csv"

# Reorder task: describe each product using its sales history before CUTOFF,
# then predict whether it is ordered again between CUTOFF and END.
CUTOFF = pd.Timestamp("2017-10-01")
END = pd.Timestamp("2018-01-01")


def load_orders(path: Path = DATA_PATH) -> pd.DataFrame:
    """One row per order line (9,994 rows, Jan 2014 to Dec 2017)."""
    df = pd.read_csv(path, encoding="latin-1")
    df["Order Date"] = pd.to_datetime(df["Order Date"], format="%m/%d/%Y")
    df["Ship Date"] = pd.to_datetime(df["Ship Date"], format="%m/%d/%Y")
    return df


def build_reorder_table(orders: pd.DataFrame) -> pd.DataFrame:
    """One row per product with history-only features and a next-quarter label.

    Every feature is computed from orders before CUTOFF and the label only from
    orders after it, so the model never sees the period it is predicting.
    """
    history = orders[orders["Order Date"] < CUTOFF]
    future = orders[(orders["Order Date"] >= CUTOFF) & (orders["Order Date"] < END)]

    g = history.groupby("Product ID")
    table = pd.DataFrame(
        {
            "category": g["Category"].first(),
            "sub_category": g["Sub-Category"].first(),
            "n_orders": g["Order ID"].nunique(),
            "n_customers": g["Customer ID"].nunique(),
            "n_regions": g["Region"].nunique(),
            "total_quantity": g["Quantity"].sum(),
            "total_sales": g["Sales"].sum(),
            "total_profit": g["Profit"].sum(),
            "avg_discount": g["Discount"].mean(),
            "avg_unit_price": g["Sales"].sum() / g["Quantity"].sum(),
            "days_since_last_order": (CUTOFF - g["Order Date"].max()).dt.days,
            "days_since_first_order": (CUTOFF - g["Order Date"].min()).dt.days,
            "orders_last_365_days": history[history["Order Date"] >= CUTOFF - pd.Timedelta(days=365)]
            .groupby("Product ID")["Order ID"]
            .nunique(),
            "main_ship_mode": g["Ship Mode"].agg(lambda s: s.mode().iloc[0]),
        }
    )
    table["orders_last_365_days"] = table["orders_last_365_days"].fillna(0).astype(int)
    table["reordered"] = table.index.isin(future["Product ID"]).astype(int)
    return table
