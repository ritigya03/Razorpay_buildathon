"""Shared data loading + feature prep for Project Sentinel models.

Phase 1 (this file's current scope): raw transaction features only.
Phase 2 will add ring / entity-graph features via a separate module; the
`prepare_features` signature is kept stable so the baseline stays comparable.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

SPLITS_DIR = Path(__file__).resolve().parents[1] / "data" / "splits"

TARGET = "isFraud"
ID_COL = "TransactionID"
TIME_COL = "TransactionDT"

# Columns dropped from the feature matrix.
#  - TARGET: label
#  - ID_COL: identifier
#  - TIME_COL: raw seconds-from-reference is monotonic with the split boundary;
#    feeding it in lets the model key on absolute position. We derive cyclical
#    hour/weekday features from it instead (see below).
DROP_COLS = [TARGET, ID_COL, TIME_COL]


def load_splits() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    parts = []
    for name in ("train", "val", "test"):
        p = SPLITS_DIR / f"{name}.parquet"
        if not p.exists():
            raise FileNotFoundError(f"{p} missing — run `make splits` first.")
        parts.append(pd.read_parquet(p))
    return tuple(parts)  # type: ignore[return-value]


def _add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    dt = df[TIME_COL].to_numpy()
    df = df.copy()
    df["_hour"] = (dt // 3600) % 24
    df["_weekday"] = (dt // 86400) % 7
    return df


def prepare_features(
    train: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame
) -> dict:
    """Returns dict with X*/y* frames, feature name list, and categorical feature list.

    Categorical columns are cast to pandas `category` dtype with the category set
    fixed from TRAIN only — values unseen in training become NaN in val/test,
    which LightGBM handles natively. This prevents any leakage of val/test-only
    category values into the encoding.
    """
    train, val, test = (_add_time_features(d) for d in (train, val, test))

    feat_cols = [c for c in train.columns if c not in DROP_COLS]

    # non-numeric, non-bool columns -> categorical, category set fixed from train.
    # (pandas >=3 reads string columns as the `str` dtype, not `object`, so we
    #  test numeric-ness rather than checking for `object`.)
    cat_cols = [
        c for c in feat_cols
        if not pd.api.types.is_numeric_dtype(train[c])
        and not pd.api.types.is_bool_dtype(train[c])
    ]
    for c in cat_cols:
        cats = pd.Index(pd.unique(train[c].dropna()))
        for d in (train, val, test):
            d[c] = pd.Categorical(d[c], categories=cats)

    out = {
        "X_train": train[feat_cols], "y_train": train[TARGET].to_numpy(),
        "X_val": val[feat_cols], "y_val": val[TARGET].to_numpy(),
        "X_test": test[feat_cols], "y_test": test[TARGET].to_numpy(),
        "amount_val": val["TransactionAmt"].to_numpy(),
        "amount_test": test["TransactionAmt"].to_numpy(),
        "feature_names": feat_cols,
        "cat_features": cat_cols,
    }
    return out
