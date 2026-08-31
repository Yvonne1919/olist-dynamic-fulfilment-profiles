"""Feature-availability audit underlying the Phase 1 information sets."""
from __future__ import annotations

import pandas as pd


def feature_availability_audit() -> pd.DataFrame:
    rows = []
    def add(features, source, availability, available, risk, block, keep, notes):
        for feature in features:
            rows.append(dict(feature=feature, raw_source=source,
                timestamp_availability=availability, available_at_promise=available,
                leakage_risk=risk, information_block=block, keep=keep, notes=notes))
    from src.features.blocks import M0_NUMERIC, M1_NUMERIC, M1_CATEGORICAL, M2_NUMERIC
    add(M0_NUMERIC, "orders", "issued estimated date and purchase timestamp", "yes", "low", "M0", "yes", "Direct descriptor of issued promise.")
    add(M1_NUMERIC, "order_items/products/sellers", "checkout/order record or catalog", "yes", "low", "M1", "yes", "Aggregated without post-promise timestamps.")
    add(M1_CATEGORICAL, "customers/order_items/products/sellers", "checkout/order record or catalog", "yes", "low", "M1", "yes", "Observed structural descriptors.")
    add(["main_seller_id"], "order_items", "checkout/order record", "yes", "low", "M1 hierarchy", "yes", "Training-only seller grouping for partial pooling; multi-seller orders use a dedicated group.")
    add(M2_NUMERIC, "purchase timestamp/issued promise/static calendar", "known at promise issuance", "yes", "low", "M2", "yes", "Calendar definitions are fixed ex ante.")
    add(["total_payment_value", "n_payment_installments", "main_payment_type"], "order_payments", "payment sequence timing not supplied", "uncertain", "medium", "M1", "no", "Excluded until availability relative to promise issuance is evidenced.")
    add(["order_approved_at"], "orders", "may occur after purchase/promise", "uncertain", "high", "ex-post", "no", "Operational timestamp excluded from ex-ante model.")
    add(["shipping_limit_date", "order_delivered_carrier_date", "order_delivered_customer_date"], "order_items/orders", "post-purchase operational/outcome", "no", "high", "ex-post", "no", "Post-promise information.")
    add(["review_score", "n_reviews"], "order_reviews", "after fulfilment", "no", "high", "RQ1 outcome", "no", "Outcome/diagnostic only.")
    add(["promise_error_days", "late_delivery", "severe_late_2d"], "derived outcome", "after delivery", "no", "certain leakage", "target", "no", "Targets only.")
    return pd.DataFrame(rows)
