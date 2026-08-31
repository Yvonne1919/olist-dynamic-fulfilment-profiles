# Olist data (download separately)

Raw Olist data are not redistributed in this repository. Obtain the original
[Brazilian E-Commerce Public Dataset by Olist](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce)
from its publisher's Kaggle page, review the publisher's terms and attribution,
then extract the CSV files into `data/olist_data/`.

Required for the complete released data/review construction:

- `olist_orders_dataset.csv`
- `olist_customers_dataset.csv`
- `olist_geolocation_dataset.csv`
- `olist_order_items_dataset.csv`
- `olist_products_dataset.csv`
- `olist_sellers_dataset.csv`
- `product_category_name_translation.csv`
- `olist_order_reviews_dataset.csv`
- `olist_order_payments_dataset.csv`

Payment records are read by the legacy data-reconciliation dependency but are
not purchase-time predictors in the final models. Review text is not a model
input. Do not commit downloaded files, Kaggle credentials, or regenerated
order/entity-level tables.

The canonical assembler accepts `--data-dir` or `OLIST_DATA_DIR`; the public RQ1
wrapper accepts `--data-dir`. Defaults resolve to `data/olist_data/` from the
repository root. Run commands from that root.

The RQ1 implementation validates frozen raw-file hashes and selected-review
counts. A different dataset version, modified line endings, or re-saved CSVs
can fail these checks. Do not silently replace the expected hashes: resolve
the input-version difference first. The public extract is not established to
be a random or representative sample of the complete platform.
