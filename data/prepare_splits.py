"""
Temporal train/val/test split for Project Sentinel.

Reads IEEE-CIS `train_transaction.csv` (+ `train_identity.csv`), joins on
TransactionID, sorts by TransactionDT, and cuts into:

    train : earliest 70%   (by TransactionDT value)
    val   : next 15%
    test  : final 15%       <- held-out, evaluated exactly once

The cut is on the TransactionDT *value* (via quantiles), not row position, so
there is zero temporal overlap between splits even when many transactions share
the same timestamp.

Output: data/splits/{train,val,test}.parquet

The IEEE-CIS `test_transaction.csv` file is NOT used anywhere in this project:
it has no `isFraud` column (Kaggle holds those labels), so it cannot be used to
measure precision/recall. Our held-out test set is the final 15% of the labelled
`train_transaction.csv` by time.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import pandas as pd

DATA_DIR = Path(os.environ.get("SENTINEL_DATA_DIR", "data/raw/ieee-fraud-detection"))
OUT_DIR = Path(__file__).resolve().parent / "splits"

TRAIN_FRAC = 0.70
VAL_FRAC = 0.15  # test gets the remaining 0.15


def main() -> None:
    tx_path = DATA_DIR / "train_transaction.csv"
    id_path = DATA_DIR / "train_identity.csv"
    if not tx_path.exists():
        sys.exit(f"Not found: {tx_path}\nSet SENTINEL_DATA_DIR to the ieee-fraud-detection folder.")

    print(f"Reading {tx_path.name} ...")
    tx = pd.read_csv(tx_path)
    print(f"Reading {id_path.name} ...")
    idf = pd.read_csv(id_path)

    df = tx.merge(idf, on="TransactionID", how="left")
    print(f"Joined: {df.shape[0]:,} rows x {df.shape[1]} cols "
          f"({df['TransactionID'].isin(idf['TransactionID']).mean():.1%} have identity)")

    df = df.sort_values(["TransactionDT", "TransactionID"]).reset_index(drop=True)

    t_train_end = df["TransactionDT"].quantile(TRAIN_FRAC)
    t_val_end = df["TransactionDT"].quantile(TRAIN_FRAC + VAL_FRAC)

    train = df[df["TransactionDT"] <= t_train_end]
    val = df[(df["TransactionDT"] > t_train_end) & (df["TransactionDT"] <= t_val_end)]
    test = df[df["TransactionDT"] > t_val_end]

    # temporal integrity: no split may start before the previous one ends
    assert train["TransactionDT"].max() <= val["TransactionDT"].min()
    assert val["TransactionDT"].max() <= test["TransactionDT"].min()
    assert len(train) + len(val) + len(test) == len(df)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    day0 = df["TransactionDT"].min()
    print("\n{:<6} {:>10} {:>9} {:>9}   {:>20}".format("split", "rows", "frauds", "fraud%", "day range"))
    print("-" * 62)
    for name, part in [("train", train), ("val", val), ("test", test)]:
        d_lo = (part["TransactionDT"].min() - day0) / 86400
        d_hi = (part["TransactionDT"].max() - day0) / 86400
        print("{:<6} {:>10,} {:>9,} {:>8.3f}%   {:>8.1f} .. {:<8.1f}".format(
            name, len(part), int(part["isFraud"].sum()),
            100 * part["isFraud"].mean(), d_lo, d_hi))
        part.to_parquet(OUT_DIR / f"{name}.parquet", index=False)

    print(f"\nWrote {OUT_DIR}/{{train,val,test}}.parquet")


if __name__ == "__main__":
    main()
