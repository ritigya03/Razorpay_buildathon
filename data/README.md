# Data

## Dataset: IEEE-CIS Fraud Detection

Real e-commerce transaction fraud, ~590k labelled transactions over ~6 months,
with rich shared-entity columns (`card1–6`, `addr1–2`, `P/R_emaildomain`,
`DeviceInfo`, `id_*`) that make coordinated-ring structure recoverable.

Download from Kaggle (requires accepting the competition rules):
<https://www.kaggle.com/competitions/ieee-fraud-detection/data>

Files used: `train_transaction.csv`, `train_identity.csv`.
Point the pipeline at the folder containing them:

```bash
export SENTINEL_DATA_DIR=/path/to/ieee-fraud-detection
```

The raw CSVs are **not** committed (`.gitignore`d — ~1.4 GB).

### Why the Kaggle `test_*.csv` files are unused

`test_transaction.csv` has no `isFraud` column (Kaggle withholds those labels),
so it cannot be used to measure precision/recall. Our held-out test set is the
**final 15% of `train_transaction.csv` by `TransactionDT`** — a true forward-in-time
holdout, produced by `prepare_splits.py`.

### Label semantics (matters for feature engineering)

A transaction is `isFraud = 1` if a chargeback was reported on the card, **and**
later transactions on the same card/account within 120 days are also labelled
fraud. The label therefore carries account/ring structure — which is the signal
Sentinel exploits, and also a leakage trap: any ring/uid aggregate feature
(fraud-rate, size, counts) must be computed only from transactions at or before
the training cutoff.

## Generated splits

`data/splits/{train,val,test}.parquet` — produced by `make splits`, git-ignored.

| Split | Rows | Fraud % | Day range |
|---|---|---|---|
| train | 413,378 | 3.52% | 0.0 – 119.8 |
| val | 88,581 | 3.43% | 119.8 – 151.2 |
| test | 88,581 | 3.48% | 151.2 – 182.0 |
