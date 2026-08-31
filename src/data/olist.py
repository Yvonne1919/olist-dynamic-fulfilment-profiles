"""Load Olist CSVs and construct one row per delivered order."""
from __future__ import annotations

import os
from pathlib import Path

import pandas as pd

REQUIRED_FILES = {
    "orders": "olist_orders_dataset.csv",
    "customers": "olist_customers_dataset.csv",
    "items": "olist_order_items_dataset.csv",
    "payments": "olist_order_payments_dataset.csv",
    "reviews": "olist_order_reviews_dataset.csv",
    "products": "olist_products_dataset.csv",
    "sellers": "olist_sellers_dataset.csv",
    "categories": "product_category_name_translation.csv",
}


def resolve_data_dir(explicit: str | Path | None = None) -> Path:
    """Resolve data without committing a user-specific absolute path."""
    candidates = []
    if explicit:
        candidates.append(Path(explicit))
    if os.environ.get("OLIST_DATA_DIR"):
        candidates.append(Path(os.environ["OLIST_DATA_DIR"]))
    candidates.extend([Path("data/olist_data"), Path("olist_data")])
    for candidate in candidates:
        if all((candidate / filename).exists() for filename in REQUIRED_FILES.values()):
            return candidate.resolve()
    raise FileNotFoundError(
        "Olist CSVs not found. Set OLIST_DATA_DIR or pass --data-dir."
    )


def _mode(series: pd.Series):
    values = series.dropna().mode()
    return values.iloc[0] if len(values) else pd.NA


def load_order_level_data(data_dir: str | Path | None = None) -> pd.DataFrame:
    """Return a unique-order delivered sample while retaining diagnostic columns."""
    root = resolve_data_dir(data_dir)
    tables = {key: pd.read_csv(root / name) for key, name in REQUIRED_FILES.items()}
    orders = tables["orders"]
    for column in [
        "order_purchase_timestamp", "order_approved_at",
        "order_delivered_carrier_date", "order_delivered_customer_date",
        "order_estimated_delivery_date",
    ]:
        orders[column] = pd.to_datetime(orders[column], errors="coerce")

    products = tables["products"].merge(tables["categories"], on="product_category_name", how="left")
    products["product_category_name_english"] = products[
        "product_category_name_english"
    ].fillna(products["product_category_name"])
    enriched_items = (
        tables["items"].merge(products, on="product_id", how="left")
        .merge(tables["sellers"], on="seller_id", how="left")
    )
    item_agg = enriched_items.groupby("order_id", as_index=False).agg(
        n_items=("order_item_id", "count"),
        n_unique_products=("product_id", "nunique"),
        n_unique_sellers=("seller_id", "nunique"),
        total_price=("price", "sum"),
        total_freight_value=("freight_value", "sum"),
        main_seller_id=("seller_id", _mode),
        main_product_category=("product_category_name_english", _mode),
        seller_state=("seller_state", _mode),
        avg_product_weight_g=("product_weight_g", "mean"),
        avg_product_length_cm=("product_length_cm", "mean"),
        avg_product_height_cm=("product_height_cm", "mean"),
        avg_product_width_cm=("product_width_cm", "mean"),
    )
    payment_agg = tables["payments"].groupby("order_id", as_index=False).agg(
        total_payment_value=("payment_value", "sum"),
        n_payment_installments=("payment_installments", "sum"),
        main_payment_type=("payment_type", _mode),
    )
    review_agg = tables["reviews"].groupby("order_id", as_index=False).agg(
        review_score=("review_score", "mean"), n_reviews=("review_id", "count")
    )
    frame = (
        orders.merge(tables["customers"], on="customer_id", how="left")
        .merge(item_agg, on="order_id", how="left")
        .merge(payment_agg, on="order_id", how="left")
        .merge(review_agg, on="order_id", how="left")
    )
    frame = frame.loc[
        frame["order_status"].eq("delivered")
        & frame["order_delivered_customer_date"].notna()
        & frame["order_estimated_delivery_date"].notna()
    ].copy()
    if frame["order_id"].duplicated().any():
        raise AssertionError("Order-level assembly produced duplicate order_id values.")
    return frame.reset_index(drop=True)
