"""
Phase 2: ring / entity-graph features for Project Sentinel.

Three families, all built to be leakage-safe under the temporal split:

  A. Reputation (uses the label) -> OUT-OF-FOLD target encoding on train
     (5 sequential time blocks; each block encoded from the others), and encoded
     from ALL of train for val/test. OOF is essential: without it the feature is
     a perfect label proxy on train and near-constant on val/test, which makes
     gradient boosting over-rely on it and collapse out-of-sample. Restricted to
     entities with high train<->later overlap (card_id ~98%, addr1 ~90%,
     P_emaildomain ~83%); _uid (33%) and _device_id (8%) are NOT reputation-encoded.

  B. Velocity / recency (no label) -> causal over the full timeline: for a
     transaction at time t only earlier transactions count (rolling closed='left',
     or cumulative-distinct minus the current row). Available in production live.

  C. Ring / connected component (structure, no label) -> union-find over
     (card_id <-> device_id) within a trailing 7-day window, rebuilt per day so
     it stays causal. Only structural stats (size, distinct cards / devices /
     accounts, transactions-per-hour) — no label-derived ring rate.

IEEE-CIS has no merchant id, so "spread" is measured over shipping address,
device and distinct card-accounts rather than merchants.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

TARGET = "isFraud"
TIME = "TransactionDT"
BASE_RATE = 0.035
SMOOTH_ALPHA = 20.0
RING_WINDOW_DAYS = 7
N_OOF_FOLDS = 5

KEY_COLS = ["_uid", "_card_id", "_device_id", "_emailaddr_id"]
REPUTATION_KEYS = ["_card_id", "addr1", "P_emaildomain"]  # high overlap only


# --------------------------------------------------------------------------- #
# entity keys
# --------------------------------------------------------------------------- #
def _sjoin(df: pd.DataFrame, cols: list[str]) -> pd.Series:
    out = df[cols[0]].astype("string").fillna("n")
    for c in cols[1:]:
        out = out.str.cat(df[c].astype("string").fillna("n"), sep="|")
    return out


def _add_keys(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    day = (df[TIME] // 86400).astype("int64")
    d1n = (df["D1"] - day).round().astype("string").fillna("n")
    df["_uid"] = _sjoin(df, ["card1", "card2", "card3", "card5", "addr1"]).str.cat(d1n, sep="|")
    df["_card_id"] = _sjoin(df, ["card1", "card2", "card3", "card5"])
    dev = _sjoin(df, ["DeviceInfo", "id_31"])
    df["_device_id"] = dev.where(df["DeviceInfo"].notna(), other=pd.NA)
    df["_emailaddr_id"] = _sjoin(df, ["P_emaildomain", "addr1"])
    return df


# --------------------------------------------------------------------------- #
# A. reputation — out-of-fold target encoding
# --------------------------------------------------------------------------- #
def _smooth_rate(cnt: pd.Series, fr: pd.Series) -> pd.Series:
    return (fr + SMOOTH_ALPHA * BASE_RATE) / (cnt + SMOOTH_ALPHA)


def _encode(keyvals: pd.Series, cnt: pd.Series, fr: pd.Series):
    rate = _smooth_rate(cnt, fr)
    return (keyvals.map(rate).fillna(BASE_RATE).to_numpy(),
            np.log1p(keyvals.map(cnt).fillna(0).to_numpy()),
            keyvals.isin(cnt.index).to_numpy().astype("int8"))


def _reputation(all_df: pd.DataFrame, is_train: np.ndarray) -> pd.DataFrame:
    feats = pd.DataFrame(index=all_df.index)
    tr = all_df.loc[is_train]
    tr_pos = np.where(is_train)[0]
    # 5 sequential time blocks over the (time-sorted) train rows
    fold = np.floor(np.linspace(0, N_OOF_FOLDS, len(tr_pos), endpoint=False)).astype(int)

    for key in REPUTATION_KEYS:
        name = key.lstrip("_")
        rate = np.full(len(all_df), BASE_RATE)
        lcnt = np.zeros(len(all_df))
        seen = np.zeros(len(all_df), dtype="int8")

        # train rows: OOF
        for k in range(N_OOF_FOLDS):
            oof = tr.iloc[fold == k]
            rest = tr.iloc[fold != k]
            g = rest.groupby(key, dropna=True)[TARGET]
            r, c, s = _encode(oof[key], g.count(), g.sum())
            idx = tr_pos[fold == k]
            rate[idx], lcnt[idx], seen[idx] = r, c, s

        # val/test rows: encode from ALL train
        g = tr.groupby(key, dropna=True)[TARGET]
        other = ~is_train
        r, c, s = _encode(all_df.loc[other, key], g.count(), g.sum())
        rate[other], lcnt[other], seen[other] = r, c, s

        feats[f"r_{name}_rep_rate"] = rate
        feats[f"r_{name}_rep_lcnt"] = lcnt
        feats[f"r_{name}_rep_seen"] = seen
    return feats


# --------------------------------------------------------------------------- #
# B. velocity / recency (causal, label-free)
# --------------------------------------------------------------------------- #
def _velocity(all_df: pd.DataFrame) -> pd.DataFrame:
    n = len(all_df)
    base = all_df[[TIME, "_uid", "_card_id", "_device_id", "addr1", "TransactionAmt"]].copy()
    base["_dt"] = pd.to_datetime(base[TIME], unit="s")
    base["_pos"] = np.arange(n)
    feats = pd.DataFrame(index=all_df.index)

    def scatter(pos, vals, fill=np.nan):
        arr = np.full(n, fill, dtype="float64")
        arr[pos] = np.asarray(vals, dtype="float64")
        return arr

    for key in ("_uid", "_card_id"):
        name = key.lstrip("_")
        sub = base.sort_values([key, "_dt", "_pos"])
        pos = sub["_pos"].to_numpy()
        g = sub.groupby(key, sort=False)

        feats[f"r_{name}_gap_s"] = scatter(pos, g[TIME].diff().to_numpy())
        r24 = g.rolling("86400s", on="_dt", closed="left")["TransactionAmt"]
        r7 = g.rolling("604800s", on="_dt", closed="left")["TransactionAmt"]
        feats[f"r_{name}_cnt_24h"] = scatter(pos, r24.count().to_numpy(), fill=0.0)
        feats[f"r_{name}_cnt_7d"] = scatter(pos, r7.count().to_numpy(), fill=0.0)
        feats[f"r_{name}_amt_sum_24h"] = scatter(pos, r24.sum().to_numpy(), fill=0.0)
        feats[f"r_{name}_amt_mean_24h"] = scatter(pos, r24.mean().to_numpy())

    # cumulative distinct fan-out, current row excluded (label-free ring signal):
    #   one physical card touching many "accounts" (uids) / addresses, etc.
    for key, val, label in [("_card_id", "_uid", "card_uids"),
                            ("_card_id", "addr1", "card_addrs"),
                            ("_uid", "_device_id", "uid_devices")]:
        sub = base.sort_values([key, "_dt", "_pos"]).copy()
        sub["_if"] = (~sub.duplicated([key, val])).astype("int64")
        cumd = sub.groupby(key, sort=False)["_if"].cumsum().to_numpy()
        distinct_before = cumd - sub["_if"].to_numpy()
        feats[f"r_{label}_cumdistinct"] = scatter(sub["_pos"].to_numpy(), distinct_before, fill=0.0)

    return feats


# --------------------------------------------------------------------------- #
# C. ring / connected component (trailing 7d union-find, rebuilt per day)
# --------------------------------------------------------------------------- #
class _UF:
    def __init__(self):
        self.p: dict = {}

    def find(self, x):
        self.p.setdefault(x, x)
        root = x
        while self.p[root] != root:
            root = self.p[root]
        while self.p[x] != root:
            self.p[x], x = root, self.p[x]
        return root

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[ra] = rb


def _rings(all_df: pd.DataFrame) -> pd.DataFrame:
    df = all_df[[TIME, "_card_id", "_device_id", "_uid"]].copy()
    df["_day"] = (df[TIME] // 86400).astype("int64")
    win = RING_WINDOW_DAYS * 86400

    cols = ["r_ring_size", "r_ring_cards", "r_ring_devices", "r_ring_uids", "r_ring_txn_per_hr"]
    out = pd.DataFrame(np.nan, index=all_df.index, columns=cols)
    min_day = int(df["_day"].min())

    for day in range(min_day, int(df["_day"].max()) + 1):
        lo = day * 86400 - win
        w = df[(df[TIME] > lo) & (df[TIME] <= (day + 1) * 86400)]
        if w.empty:
            continue
        uf = _UF()
        has_dev = w["_device_id"].notna().to_numpy()
        cids = w["_card_id"].to_numpy()
        dids = w["_device_id"].to_numpy()
        for i in range(len(w)):
            uf.union(("c", cids[i]), ("d", dids[i]) if has_dev[i] else ("c", cids[i]))
        root_to_id: dict = {}
        comp_ids = np.fromiter(
            (root_to_id.setdefault(uf.find(("c", c)), len(root_to_id)) for c in cids),
            dtype="int64", count=len(cids),
        )
        wg = w.assign(_comp=comp_ids)
        agg = wg.groupby("_comp").agg(
            size=("_card_id", "size"),
            cards=("_card_id", "nunique"),
            devices=("_device_id", "nunique"),
            uids=("_uid", "nunique"),
            t0=(TIME, "min"),
            t1=(TIME, "max"),
        )
        span_hr = np.maximum((agg["t1"] - agg["t0"]) / 3600.0, 1.0 / 60)
        agg["txn_per_hr"] = agg["size"] / span_hr

        this = wg[wg["_day"] == day]
        m = agg.loc[this["_comp"].to_numpy()]
        idx = this.index
        out.loc[idx, "r_ring_size"] = m["size"].to_numpy()
        out.loc[idx, "r_ring_cards"] = m["cards"].to_numpy()
        out.loc[idx, "r_ring_devices"] = m["devices"].to_numpy()
        out.loc[idx, "r_ring_uids"] = m["uids"].to_numpy()
        out.loc[idx, "r_ring_txn_per_hr"] = m["txn_per_hr"].to_numpy()

    return out


# --------------------------------------------------------------------------- #
# public entry
# --------------------------------------------------------------------------- #
def build_entity_features(
    train: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame,
    *, reputation: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    all_df = pd.concat([train, val, test], ignore_index=True)
    all_df = all_df.sort_values([TIME, "TransactionID"]).reset_index(drop=True)
    all_df = _add_keys(all_df)

    is_train = all_df["TransactionID"].isin(set(train["TransactionID"])).to_numpy()
    is_val = all_df["TransactionID"].isin(set(val["TransactionID"])).to_numpy()
    is_test = ~(is_train | is_val)

    parts = []
    print("  [B] velocity / recency features (causal) ...")
    parts.append(_velocity(all_df))
    print("  [C] ring / connected-component features (7d union-find) ...")
    parts.append(_rings(all_df))
    if reputation:
        print("  [A] reputation features (out-of-fold target encoding) ...")
        parts.append(_reputation(all_df, is_train))

    feat = pd.concat([all_df.drop(columns=KEY_COLS)] + parts, axis=1)
    n_added = sum(p.shape[1] for p in parts)

    tr_out = feat[is_train].reset_index(drop=True)
    va_out = feat[is_val].reset_index(drop=True)
    te_out = feat[is_test].reset_index(drop=True)
    print(f"  added {n_added} ring features; "
          f"train {len(tr_out):,} val {len(va_out):,} test {len(te_out):,}")
    return tr_out, va_out, te_out
