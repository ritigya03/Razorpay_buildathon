"""Shared data loading + feature prep for Project Sentinel models.

The feature space (which columns, and the category set for each categorical
column — fixed from TRAIN only, so no val/test leakage) is captured in a
`spec` dict that can be saved to JSON and reloaded by the backend to score
arbitrary transactions with the trained model.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

SPLITS_DIR = Path(__file__).resolve().parents[1] / "data" / "splits"

TARGET = "isFraud"
ID_COL = "TransactionID"
TIME_COL = "TransactionDT"

# Dropped from the feature matrix:
#  - TARGET / ID_COL: label and identifier
#  - TIME_COL: raw seconds-from-reference is monotonic with the split boundary;
#    feeding it in lets the model key on absolute position. Cyclical hour/weekday
#    are derived from it instead.
DROP_COLS = [TARGET, ID_COL, TIME_COL]


def load_splits() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    parts = []
    for name in ("train", "val", "test"):
        p = SPLITS_DIR / f"{name}.parquet"
        if not p.exists():
            raise FileNotFoundError(f"{p} missing — run `make splits` first.")
        parts.append(pd.read_parquet(p))
    return tuple(parts)  # type: ignore[return-value]


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    dt = df[TIME_COL].to_numpy()
    df = df.copy()
    df["_hour"] = (dt // 3600) % 24
    df["_weekday"] = (dt // 86400) % 7
    return df


def build_feature_spec(train: pd.DataFrame) -> dict:
    """Feature column order + per-categorical category sets, fixed from train."""
    train = add_time_features(train)
    feat_cols = [c for c in train.columns if c not in DROP_COLS]
    cat_cols = [
        c for c in feat_cols
        if not pd.api.types.is_numeric_dtype(train[c])
        and not pd.api.types.is_bool_dtype(train[c])
    ]
    categories = {c: [str(v) for v in pd.unique(train[c].dropna())] for c in cat_cols}
    return {"feature_names": feat_cols, "cat_features": cat_cols, "categories": categories}


def save_spec(spec: dict, path: str | Path) -> None:
    Path(path).write_text(json.dumps(spec, indent=2))


def load_spec(path: str | Path) -> dict:
    return json.loads(Path(path).read_text())


def transform(df: pd.DataFrame, spec: dict) -> pd.DataFrame:
    """Apply the feature spec to `df` and return X in the trained column order.

    Missing feature columns are added as NaN; categorical values outside the
    train-time set become NaN (LightGBM handles both natively).
    """
    df = add_time_features(df) if TIME_COL in df.columns else df.copy()
    for c in spec["feature_names"]:
        if c not in df.columns:
            df[c] = np.nan
    for c in spec["cat_features"]:
        cats = spec["categories"][c]
        s = df[c].astype("string")
        s = s.where(s.isin(cats))  # unseen values -> NaN before constructing the Categorical
        df[c] = pd.Categorical(s, categories=cats)
    return df[spec["feature_names"]]


def prepare_features(
    train: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame
) -> dict:
    """Training-time convenience: build the spec from train, transform all three."""
    spec = build_feature_spec(train)
    Xtr, Xva, Xte = (transform(d, spec) for d in (train, val, test))
    return {
        "X_train": Xtr, "y_train": train[TARGET].to_numpy(),
        "X_val": Xva, "y_val": val[TARGET].to_numpy(),
        "X_test": Xte, "y_test": test[TARGET].to_numpy(),
        "amount_val": val["TransactionAmt"].to_numpy(),
        "amount_test": test["TransactionAmt"].to_numpy(),
        "feature_names": spec["feature_names"],
        "cat_features": spec["cat_features"],
        "spec": spec,
    }
