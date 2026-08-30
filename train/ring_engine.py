"""
Phase 2 (pivoted): the ring engine — triage + explainability, measured at RING level.

Finding from `model_ring.py`: explicit ring/entity features do NOT lift held-out
PR-AUC (+0.003, within noise) — the pre-engineered Vesta columns + LightGBM's
categorical handling already capture that signal. So the ring layer is not a
feature set; it is a triage product that turns a flood of transaction alerts into
a short list of coordinated rings, each with a forensic summary.

Ring definitions (deliberately conservative — no transitive union-find, so no
"mega-blob" merging thousands of unrelated transactions):

  * device ring   : one *specific* device fingerprint (DeviceInfo|browser, must
                    contain a digit or space and not be a generic OS label) used
                    by 2..CAP distinct accounts in the window.
  * address ring  : one exact (addr1, addr2, P_emaildomain) tuple used by
                    2..CAP distinct card identities in the window.

CAP filters out shared-NAT / datacenter / retail-kiosk artifacts (a fingerprint
tied to hundreds of accounts is infrastructure, not a ring).

A "true fraud ring" (the recall denominator) has >= 2 fraudulent transactions.

Only ~24% of transactions carry device data, so a device-based ring engine can
only ever cover the fraud that actually travels in coordinated rings — the report
states that ceiling explicitly rather than hiding it.

Advisory only: a flagged ring produces a review item + summary, never an auto-block.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import load_splits, prepare_features  # noqa: E402
from evaluate import ROOT  # noqa: E402
from ring_features import _add_keys  # noqa: E402

MODEL_FILE = ROOT / "models" / "baseline_lgb.txt"
OUT = ROOT / "report" / "ring_metrics.json"

CAP = 25  # max distinct accounts/cards for a group to count as a "ring"
GENERIC_DEV = {"Windows", "iOS Device", "MacOS", "Linux", "Trident/7.0", "other"}


def _specific_device(s) -> bool:
    return isinstance(s, str) and s not in GENERIC_DEV and bool(re.search(r"[0-9 ]", s))


def _rings_from_key(df: pd.DataFrame, key: str, count_col: str) -> pd.DataFrame:
    g = df.groupby(key)
    t = g.agg(
        txns=("score", "size"),
        members=(count_col, "nunique"),
        cards=("_card_id", "nunique"),
        uids=("_uid", "nunique"),
        score_mean=("score", "mean"),
        score_max=("score", "max"),
        frac_hot=("_hot", "mean"),
        amt=("TransactionAmt", "sum"),
        t0=("TransactionDT", "min"),
        t1=("TransactionDT", "max"),
        n_fraud=("isFraud", "sum"),
    )
    t = t[(t["members"] >= 2) & (t["members"] <= CAP)]
    t["is_fraud_ring"] = t["n_fraud"] >= 2
    t["span_h"] = np.maximum((t["t1"] - t["t0"]) / 3600.0, 1 / 60)
    return t


def _evaluate(rings: pd.DataFrame, thr: float) -> dict:
    flagged = rings["score_mean"] >= thr
    fr = rings["is_fraud_ring"]
    tp = int((flagged & fr).sum()); fp = int((flagged & ~fr).sum()); fn = int((~flagged & fr).sum())
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * p * r / (p + r) if p + r else 0.0
    return dict(threshold=float(thr), rings_flagged=int(flagged.sum()),
                ring_tp=tp, ring_fp=fp, ring_fn=fn,
                ring_precision=p, ring_recall=r, ring_f1=f1)


def _prep(df: pd.DataFrame, p: np.ndarray, txn_thr: float) -> pd.DataFrame:
    f = _add_keys(df.copy())
    f["score"] = p
    f["_hot"] = (p >= txn_thr).astype(int)
    f["_dev_ring_key"] = f["_device_id"].where(f["DeviceInfo"].map(_specific_device))
    f["_addr_ring_key"] = (
        f["addr1"].astype("string").fillna("n") + "|" +
        f["addr2"].astype("string").fillna("n") + "|" +
        f["P_emaildomain"].astype("string").fillna("n")
    ).where(f["addr1"].notna() & f["P_emaildomain"].notna())
    return f


def _tune_and_eval(rv: pd.DataFrame, rt: pd.DataFrame) -> tuple[dict, list]:
    grid = np.unique(np.quantile(rv["score_mean"], np.linspace(0.3, 0.999, 300)))
    best = max((_evaluate(rv, t) for t in grid), key=lambda d: d["ring_f1"])
    thr = best["threshold"]
    test_eval = _evaluate(rt, thr)
    curve = [_evaluate(rt, t) for t in
             np.unique(np.quantile(rt["score_mean"], np.linspace(0.3, 0.999, 120)))]
    return {"val_tuning": best, "test": test_eval, "threshold": thr}, curve


def main() -> None:
    train, val, test = load_splits()
    data = prepare_features(train.copy(), val.copy(), test.copy())
    booster = lgb.Booster(model_file=str(MODEL_FILE))
    p_val = booster.predict(data["X_val"])
    p_test = booster.predict(data["X_test"])
    txn_thr = json.loads((ROOT / "report" / "metrics.json").read_text())["operating_point"]["threshold"]

    fv = _prep(val, p_val, txn_thr)
    ft = _prep(test, p_test, txn_thr)

    n_fraud_txn = int(ft["isFraud"].sum())
    txn_alerts = int((p_test >= txn_thr).sum())
    report = {
        "meta": {
            "model": "Phase-1 baseline scorer + conservative ring grouping (no union-find)",
            "cap_distinct_members": CAP,
            "fraud_ring_definition": ">= 2 fraudulent transactions in the group",
            "ring_flag_rule": "mean member risk >= threshold, tuned on validation for best ring-F1",
            "action": "advisory — flagged ring -> review item + forensic summary, never auto-block",
            "per_txn_threshold": txn_thr,
            "test_fraud_transactions": n_fraud_txn,
            "test_transaction_level_alerts": txn_alerts,
        },
        "rings": {},
    }

    all_flagged_mask = np.zeros(len(ft), dtype=bool)

    for kind, key, ccol in [("device", "_dev_ring_key", "_uid"),
                            ("address", "_addr_ring_key", "_card_id")]:
        rv = _rings_from_key(fv[fv[key].notna()], key, ccol)
        rt = _rings_from_key(ft[ft[key].notna()], key, ccol)
        res, curve = _tune_and_eval(rv, rt)
        thr = res["threshold"]

        flagged_ids = set(rt.index[rt["score_mean"] >= thr])
        member = ft[key].isin(flagged_ids).to_numpy()
        all_flagged_mask |= member
        covered = int(ft.loc[member, "isFraud"].sum())
        bonus = int(ft.loc[member & (ft["isFraud"] == 1) & (ft["score"] < txn_thr)].shape[0])
        fraud_in_kind = int(ft.loc[ft[key].isin(rt.index), "isFraud"].sum())

        report["rings"][kind] = {
            "groups": int(len(rt)),
            "true_fraud_rings": int(rt["is_fraud_ring"].sum()),
            "ring_threshold": thr,
            **res["test"],
            "val_tuning": res["val_tuning"],
            "fraud_txns_in_this_ring_type": fraud_in_kind,
            "fraud_txns_covered_by_flagged": covered,
            "fraud_coverage_pct_of_all_fraud": covered / n_fraud_txn,
            "bonus_fraud_below_txn_threshold": bonus,
            "curve": curve,
        }

    combined_cov = int(ft.loc[all_flagged_mask, "isFraud"].sum())
    total_ring_alerts = sum(r["rings_flagged"] for r in report["rings"].values())
    report["combined"] = {
        "ring_alerts_total": total_ring_alerts,
        "alert_volume_reduction_x": txn_alerts / max(total_ring_alerts, 1),
        "fraud_txns_covered": combined_cov,
        "fraud_coverage_pct_of_all_fraud": combined_cov / n_fraud_txn,
    }

    OUT.write_text(json.dumps(report, indent=2))

    print("\nRING ENGINE — held-out TEST")
    for kind, r in report["rings"].items():
        print(f"  [{kind}]  groups {r['groups']:4d}  fraud-rings {r['true_fraud_rings']:3d}  "
              f"| flagged {r['rings_flagged']:3d}  precision {r['ring_precision']:.2f}  "
              f"recall {r['ring_recall']:.2f}  F1 {r['ring_f1']:.2f}  "
              f"| fraud covered {r['fraud_txns_covered_by_flagged']:4d} "
              f"(+{r['bonus_fraud_below_txn_threshold']} below per-txn thr)")
    c = report["combined"]
    print(f"  [combined] {txn_alerts:,} txn alerts -> {c['ring_alerts_total']} ring alerts "
          f"({c['alert_volume_reduction_x']:.0f}x fewer);  "
          f"fraud coverage {c['fraud_txns_covered']:,}/{n_fraud_txn:,} "
          f"({c['fraud_coverage_pct_of_all_fraud']:.1%})")
    print(f"\nWrote {OUT}")


if __name__ == "__main__":
    main()
