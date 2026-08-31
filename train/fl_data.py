"""
Phase 4 — dense feature matrices + synthetic merchant partition for the
federated-learning experiment.

DP-SGD and FedAvg both need a differentiable model on a fixed numeric matrix, so
this module turns the same IEEE-CIS splits Phase 1-3 use into plain float32
arrays:

  * numeric columns  -> median-impute (train stat) -> standardize (train stat)
  * categorical cols -> frequency encoding from the TRAIN value counts
    (deliberately NOT target encoding — DEVLOG Phase 2a documents target-encoding
    leakage collapsing a model on exactly this data)
  * engineered cols  -> the ~27 leakage-safe ring/entity features from
    `train/ring_features.py` (OOF-time-blocked reputation on the high-overlap
    entities, causal velocity, 7d union-find ring structure). MLPs can't derive
    these from the raw Vesta columns the way LightGBM's splits can, so they add
    fresh signal here even though Phase 2a found no lift for LightGBM.

The feature set / column order is taken from `common.build_feature_spec` so the
category handling matches training-time Phase 1.

Merchant partition: there is no merchant column in IEEE-CIS, so we synthesise
`N_MERCHANTS` merchants by hashing `card1` (a card BIN/issuer profile). The split
is deliberately NON-IID — per-merchant row count and fraud rate differ and are
reported in `report/fl_metrics.json` rather than hidden.

Everything is cached to `data/splits/fl_cache.npz` (rebuild with `--rebuild` or
`build_arrays(rebuild=True)`).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (  # noqa: E402
    TARGET, add_time_features, build_feature_spec, load_splits,
)
from ring_features import build_entity_features  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "data" / "splits" / "fl_cache.npz"
CACHE_VERSION = 3
N_MERCHANTS = 8
SEED = 42
USE_ENGINEERED = True  # append the leakage-safe ring/entity features


# --------------------------------------------------------------------------- #
# encoding
# --------------------------------------------------------------------------- #
def _fit_encoders(train: pd.DataFrame, spec: dict) -> dict:
    """Median/mean/std per numeric column + frequency map per categorical, from train only."""
    train = add_time_features(train)
    num_cols = [c for c in spec["feature_names"] if c not in spec["cat_features"]]
    medians, means, stds = {}, {}, {}
    for c in num_cols:
        v = pd.to_numeric(train[c], errors="coerce").to_numpy(dtype="float64")
        med = np.nanmedian(v) if np.isfinite(v).any() else 0.0
        v = np.where(np.isfinite(v), v, med)
        mean = float(v.mean())
        std = float(v.std())
        medians[c], means[c], stds[c] = float(med), mean, std if std > 1e-9 else 1.0

    freq = {}
    for c in spec["cat_features"]:
        vc = train[c].astype("string").value_counts(normalize=True)
        freq[c] = {str(k): float(v) for k, v in vc.items()}

    return {"num_cols": num_cols, "medians": medians, "means": means,
            "stds": stds, "cat_cols": list(spec["cat_features"]), "freq": freq}


def _apply(df: pd.DataFrame, enc: dict) -> tuple[np.ndarray, list[str]]:
    df = add_time_features(df)
    cols = enc["num_cols"] + enc["cat_cols"]
    X = np.zeros((len(df), len(cols)), dtype=np.float32)
    for j, c in enumerate(enc["num_cols"]):
        v = pd.to_numeric(df.get(c), errors="coerce").to_numpy(dtype="float64")
        v = np.where(np.isfinite(v), v, enc["medians"][c])
        X[:, j] = (v - enc["means"][c]) / enc["stds"][c]
    off = len(enc["num_cols"])
    for j, c in enumerate(enc["cat_cols"]):
        m = enc["freq"][c]
        s = df.get(c)
        if s is None:
            continue
        X[:, off + j] = s.astype("string").map(m).fillna(0.0).to_numpy(dtype="float32")
    # guard against any stray non-finite value reaching the MLP
    np.nan_to_num(X, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
    return X, cols


def _merchant_ids(train: pd.DataFrame) -> np.ndarray:
    """Stable hash of card1 -> [0, N_MERCHANTS). NaN card1 hashes into its own bucket."""
    key = train["card1"] if "card1" in train.columns else train.iloc[:, 0]
    h = pd.util.hash_pandas_object(key.fillna(-1), index=False).to_numpy()
    return (h % N_MERCHANTS).astype(np.int64)


# --------------------------------------------------------------------------- #
# public
# --------------------------------------------------------------------------- #
def build_arrays(rebuild: bool = False) -> dict:
    if CACHE.exists() and not rebuild:
        z = np.load(CACHE, allow_pickle=True)
        if int(z["version"]) == CACHE_VERSION:
            return {k: z[k] for k in z.files}
        print(f"cache version {int(z['version'])} != {CACHE_VERSION}, rebuilding")

    print("Loading splits ...")
    train, val, test = load_splits()

    if USE_ENGINEERED:
        print("Building leakage-safe ring/entity features (train/ring_features.py) ...")
        train, val, test = build_entity_features(train, val, test, reputation=True)

    spec = build_feature_spec(train)
    n_eng = sum(c.startswith("r_") for c in spec["feature_names"])
    print(f"features: {len(spec['feature_names'])}  "
          f"(categorical {len(spec['cat_features'])}, engineered {n_eng})")

    enc = _fit_encoders(train, spec)
    Xtr, cols = _apply(train, enc)
    Xva, _ = _apply(val, enc)
    Xte, _ = _apply(test, enc)
    print(f"dense matrix: {Xtr.shape} train | {Xva.shape} val | {Xte.shape} test")

    mids = _merchant_ids(train)
    ytr = train[TARGET].to_numpy(dtype=np.float32)
    out = {
        "version": np.int64(CACHE_VERSION),
        "X_train": Xtr, "y_train": ytr,
        "X_val": Xva, "y_val": val[TARGET].to_numpy(dtype=np.float32),
        "X_test": Xte, "y_test": test[TARGET].to_numpy(dtype=np.float32),
        "merchant_ids": mids,
        "feature_names": np.array(cols, dtype=object),
        "n_merchants": np.int64(N_MERCHANTS),
    }
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(CACHE, **out)
    print(f"cached -> {CACHE}")
    return out


def merchant_summary(y_train: np.ndarray, merchant_ids: np.ndarray) -> list[dict]:
    rows = []
    for m in range(int(merchant_ids.max()) + 1):
        mask = merchant_ids == m
        n = int(mask.sum())
        fr = float(y_train[mask].mean()) if n else 0.0
        rows.append({"merchant": m, "rows": n, "fraud_rows": int(y_train[mask].sum()),
                     "fraud_rate": fr})
    return rows


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--rebuild", action="store_true")
    args = ap.parse_args()
    d = build_arrays(rebuild=args.rebuild)
    print(f"\nmerchant partition (non-IID, hash(card1) % {int(d['n_merchants'])}):")
    for r in merchant_summary(d["y_train"], d["merchant_ids"]):
        print(f"  merchant {r['merchant']}: {r['rows']:>7,} rows  "
              f"{r['fraud_rows']:>5,} fraud  ({r['fraud_rate']:.2%})")
