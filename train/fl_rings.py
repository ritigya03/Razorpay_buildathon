"""
Phase 8 — federated cross-merchant ring detection.

The pitch: *the same cross-merchant fraud-ring signal Vulcan gets by centralising
raw data, but without any merchant sharing raw transactions* — via federated
aggregation and differential privacy.

This experiment measures exactly that. The per-transaction risk score is held
FIXED (the Phase-1 LightGBM) across both arms, so the comparison isolates the
cost of federating the ring-detection layer:

  * CENTRALISED arm — pool every transaction, compute per-fingerprint
    (txn count, mean risk, distinct merchants) exactly, flag the coordinated
    cross-merchant ones. This is the "sees everything" oracle.

  * FEDERATED arm — each of N synthetic merchants sees ONLY its own rows. Per
    fingerprint it releases a Gaussian-DP-noised (count, risk-sum) pair — never
    raw rows, emails, amounts or card numbers. Reports are committed
    (SHA-256) then revealed; the commitments are hashed into a Merkle root.
    A robust aggregator rejects a merchant whose reported volume is a gross
    outlier (flood poisoning) or whose reveal does not match its commitment.
    The surviving histograms are summed and the SAME flag rule is applied.

Entity = device / address fingerprint. A card's rows all hash to one merchant
(merchant = hash(card1) % N), so cards cannot span merchants on this partition;
device fingerprints and shipping addresses can — and the concept doc names
"shared device fingerprints / IP clusters" as the cross-merchant signal anyway.

Ground truth (built centrally with true labels — an eval oracle, not part of any
detector): a fingerprint is a *cross-merchant fraud ring* if 2 <= its txn count
<= CAP, it has >= 2 fraudulent transactions, and its transactions span >= 2
merchants. Single-merchant fraud rings are reported for context — a lone merchant
already catches those; federation is for the ones no single merchant sees.

Honest limitations (also in report/PHASE8.md):
  * IEEE-CIS has no merchant column -> synthetic merchants (hash(card1) % N),
    same partition as Phase 4.
  * The scorer is centralised LightGBM held fixed. A fully federated deployment
    would use the Phase-4 federated MLP (Phase 4: ~0.01 PR-AUC vs centralised);
    the two costs compose. LightGBM is used here so the ring numbers are not
    confounded by the trees-vs-nets tabular gap.
  * DP-noised histogram sharing is a simplified stand-in for a real
    private-set-intersection protocol (stated, like the Merkle root).

Run:  make fl-rings   ->  report/fl_ring_metrics.json   (core .venv; seed 42)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import load_splits, prepare_features  # noqa: E402
from fl_crypto import (  # noqa: E402
    BUCKET_EDGES, BUCKET_MID, DPHistogram, N_BUCKETS, bucketize,
    commit_histogram, fingerprint_hmac, merkle_root, risk_estimate,
)
from ring_features import _add_keys  # noqa: E402
from ring_engine import _specific_device  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
MODEL_FILE = ROOT / "models" / "baseline_lgb.txt"
OUT = ROOT / "report" / "fl_ring_metrics.json"
SALT = b"sentinel-fl-rings-v1"          # fixed experiment salt for the fingerprint HMAC

SEED = 42
N_MERCHANTS = 8
CAP = 40                # a fingerprint with > CAP txns in the window is shared infra
DELTA = 1e-5
NORM_RATIO = 5.0        # reject a merchant whose reported volume > NORM_RATIO x median
                        # (loose — merchant volumes are genuinely non-IID; this only
                        #  catches gross volume-inflation, not honest size spread)
RISK_RATIO = 2.0        # reject a merchant whose mean risk-estimate > this x median merchant
EPS_SWEEP = [float("inf"), 32.0, 16.0, 8.0, 4.0, 2.0, 1.0]
N_DP_REPEATS = 5        # average metrics over this many independent noise draws per epsilon

# risk buckets each merchant sorts its own transactions into (by the shared
# scorer). One txn -> exactly one bucket, so the vector has DP sensitivity 1.
# Bucket edges / midpoints / helpers live in fl_crypto (shared with the live
# detector). Local aliases keep the rest of this file terse.
_bucketize = bucketize
_risk_estimate = risk_estimate


# --------------------------------------------------------------------------- #
# data
# --------------------------------------------------------------------------- #
def _merchant_ids(df: pd.DataFrame) -> np.ndarray:
    key = df["card1"] if "card1" in df.columns else df.iloc[:, 0]
    h = pd.util.hash_pandas_object(key.fillna(-1), index=False).to_numpy()
    return (h % N_MERCHANTS).astype(np.int64)


def _prep(df: pd.DataFrame, scores: np.ndarray) -> pd.DataFrame:
    f = _add_keys(df.copy())
    f["score"] = scores
    f["merchant"] = _merchant_ids(df)
    f["dev_fp"] = f["_device_id"].where(f["DeviceInfo"].map(_specific_device))
    f["addr_fp"] = (
        f["addr1"].astype("string").fillna("n") + "|"
        + f["addr2"].astype("string").fillna("n") + "|"
        + f["P_emaildomain"].astype("string").fillna("n")
    ).where(f["addr1"].notna() & f["P_emaildomain"].notna())
    return f


# --------------------------------------------------------------------------- #
# ground truth + centralised aggregation (exact, from pooled data)
# --------------------------------------------------------------------------- #
def _exact_stats(f: pd.DataFrame, fp_col: str) -> pd.DataFrame:
    d = f[f[fp_col].notna()].copy()
    d["_h"] = d[fp_col].map(lambda x: fingerprint_hmac(SALT, str(x)))  # match the federated ids
    d["_b"] = _bucketize(d["score"].to_numpy())
    rows = []
    for h, grp in d.groupby("_h"):
        vec = np.bincount(grp["_b"].to_numpy(), minlength=N_BUCKETS).astype(float)
        rows.append((h, len(grp), grp["merchant"].nunique(), int(grp["isFraud"].sum()), vec))
    t = pd.DataFrame(rows, columns=["_h", "txns", "merchants", "n_fraud", "vec"]).set_index("_h")
    t["risk_est"] = _risk_estimate(np.stack(t["vec"].to_numpy()))
    t["is_fraud_ring"] = t["n_fraud"] >= 2
    t["cross_merchant"] = t["merchants"] >= 2
    t["in_window"] = (t["txns"] >= 2) & (t["txns"] <= CAP)
    return t


def _flag(stats: pd.DataFrame, thr: float) -> pd.Series:
    return stats["in_window"] & stats["cross_merchant"] & (stats["risk_est"] >= thr)


def _truth_fps(stats: pd.DataFrame) -> set:
    """The fingerprint ids that ARE cross-merchant fraud rings (eval oracle)."""
    m = stats["is_fraud_ring"] & stats["cross_merchant"] & stats["in_window"]
    return set(stats.index[m])


def _pr_against(flagged: pd.Series, truth: set) -> dict:
    """precision / recall / F1 of a flagged fingerprint set vs the truth set."""
    fl = set(flagged.index[flagged])
    tp = len(fl & truth); fp = len(fl - truth); fn = len(truth - fl)
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return {"flagged": len(fl), "tp": tp, "fp": fp, "fn": fn,
            "precision": round(p, 4), "recall": round(r, 4),
            "f1": round(2 * p * r / (p + r), 4) if p + r else 0.0}


# --------------------------------------------------------------------------- #
# federated aggregation (from per-merchant DP-noised histograms)
# --------------------------------------------------------------------------- #
def _merchant_histograms(f: pd.DataFrame, fp_col: str, epsilon: float,
                         poison: str | None = None, rep: int = 0) -> tuple[dict, dict]:
    """Each merchant sees only its own rows. Per salted-HMAC fingerprint it
    releases a Gaussian-DP-noised risk-bucket count vector. `rep` shifts the
    noise seed for independent draws. Returns {merchant: revealed_hist},
    {merchant: commitment}."""
    reveals, commits = {}, {}
    d = f[f[fp_col].notna()].copy()
    d["_h"] = d[fp_col].map(lambda x: fingerprint_hmac(SALT, str(x)))
    d["_b"] = _bucketize(d["score"].to_numpy())
    for m in range(N_MERCHANTS):
        rows = d[d["merchant"] == m]
        hist: dict[str, list[float]] = {}
        for h, grp in rows.groupby("_h"):
            hist[h] = np.bincount(grp["_b"].to_numpy(), minlength=N_BUCKETS).astype(float).tolist()
        hist = DPHistogram(epsilon, DELTA, seed=SEED + m + 100_000 * rep).release(hist)

        honest_commit = commit_histogram(hist)
        if poison == "hot_flood" and m == 0:
            # shove every fingerprint's whole mass into the top risk bucket
            hist = {k: [0.0] * (N_BUCKETS - 1) + [float(sum(v))] for k, v in hist.items()}
            commits[m] = commit_histogram(hist)
        elif poison == "commit_mismatch" and m == 0:
            hist = {k: [0.0] * (N_BUCKETS - 1) + [float(sum(v))] for k, v in hist.items()}
            commits[m] = honest_commit                       # stale commitment
        else:
            commits[m] = honest_commit
        reveals[m] = hist
    return reveals, commits


def _federated_aggregate(reveals: dict, commits: dict, defend: bool
                         ) -> tuple[pd.DataFrame, str, list]:
    audit: list[dict] = []
    root = merkle_root([commits[m] for m in sorted(commits)])

    # 1. commit / reveal verification
    verified = {}
    for m, hist in reveals.items():
        if commit_histogram(hist) != commits.get(m):
            audit.append({"merchant": m, "reason": "commitment_mismatch"})
        else:
            verified[m] = hist

    # 2. robust filters
    if defend and len(verified) >= 3:
        totals = {m: sum(sum(v) for v in h.values()) for m, h in verified.items()}
        med_t = float(np.median(list(totals.values())))
        risk = {}
        for m, h in verified.items():
            mat = np.array(list(h.values())) if h else np.zeros((1, N_BUCKETS))
            risk[m] = float(_risk_estimate(mat.sum(axis=0, keepdims=True))[0])
        med_r = float(np.median(list(risk.values())))
        for m in list(verified):
            if med_t > 0 and totals[m] > NORM_RATIO * med_t:
                audit.append({"merchant": m, "reason": "volume_outlier",
                              "ratio": round(totals[m] / med_t, 2)})
                verified.pop(m)
            elif med_r > 0 and risk[m] > RISK_RATIO * med_r:
                audit.append({"merchant": m, "reason": "risk_estimate_outlier",
                              "ratio": round(risk[m] / med_r, 2)})
                verified.pop(m)

    # 3. per-fingerprint sum of bucket vectors across surviving merchants
    fps = {k for h in verified.values() for k in h}
    rows = []
    for fp in fps:
        contribs = [np.asarray(h[fp], dtype=float) for h in verified.values() if fp in h]
        vec = np.sum(contribs, axis=0)
        rows.append((fp, float(vec.sum()), vec, sum(1 for c in contribs if c.sum() >= 1)))
    t = pd.DataFrame(rows, columns=["_h", "txns", "vec", "merchants"]).set_index("_h")
    t["risk_est"] = _risk_estimate(np.stack(t["vec"].to_numpy())) if len(t) else []
    t["in_window"] = (t["txns"] >= 2) & (t["txns"] <= CAP)
    t["cross_merchant"] = t["merchants"] >= 2
    return t, root, audit


# --------------------------------------------------------------------------- #
# run
# --------------------------------------------------------------------------- #
def _best_threshold(stats: pd.DataFrame) -> float:
    """risk-estimate threshold maximising F1 over val fraud rings (2<=txns<=CAP,
    >=2 fraud) — cross-merchant span not required here, for a larger tuning set."""
    truth = stats["is_fraud_ring"] & stats["in_window"]
    best_f1, best_t = -1.0, 0.5
    for thr in np.linspace(0.1, 0.9, 81):
        fl = stats["in_window"] & (stats["risk_est"] >= thr)
        tp = int((fl & truth).sum()); fp = int((fl & ~truth).sum())
        fn = int((~fl & truth).sum())
        p = tp / (tp + fp) if tp + fp else 0.0
        r = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * p * r / (p + r) if p + r else 0.0
        if f1 > best_f1:
            best_f1, best_t = f1, float(thr)
    return round(best_t, 4)


def _tune_threshold(fv: pd.DataFrame, epsilon: float = float("inf")) -> dict:
    """One risk-estimate threshold per ring kind. epsilon=inf -> tuned on the
    exact val stats (used for centralised + no-DP federated). A finite epsilon ->
    tuned on val AFTER the same DP noise + robust aggregation, averaged over a
    few independent noise draws so calibration luck isn't a confound (val only;
    never test)."""
    out = {}
    for name, col in (("device", "dev_fp"), ("address", "addr_fp")):
        if not np.isfinite(epsilon):
            out[name] = _best_threshold(_exact_stats(fv, col))
            continue
        gt = _exact_stats(fv, col)[["is_fraud_ring"]]
        ts = []
        for rep in range(3):
            reveals, commits = _merchant_histograms(fv, col, epsilon, rep=1000 + rep)
            s, _, _ = _federated_aggregate(reveals, commits, defend=True)
            s = s.join(gt, how="left")
            s["is_fraud_ring"] = s["is_fraud_ring"].fillna(False)
            ts.append(_best_threshold(s))
        out[name] = round(float(np.median(ts)), 4)
    return out


def _combined(dev: dict, addr: dict) -> dict:
    return {k: dev[k] + addr[k] for k in ("flagged", "tp", "fp", "fn")} | _pr(dev, addr)


def _pr(dev: dict, addr: dict) -> dict:
    tp = dev["tp"] + addr["tp"]; fp = dev["fp"] + addr["fp"]; fn = dev["fn"] + addr["fn"]
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return {"precision": round(p, 4), "recall": round(r, 4),
            "f1": round(2 * p * r / (p + r), 4) if p + r else 0.0}


def main() -> None:
    train, val, test = load_splits()
    data = prepare_features(train.copy(), val.copy(), test.copy())
    booster = lgb.Booster(model_file=str(MODEL_FILE))
    fv = _prep(val, booster.predict(data["X_val"]))
    ft = _prep(test, booster.predict(data["X_test"]))

    thr = _tune_threshold(fv)                       # exact-stats threshold
    KINDS = (("device", "dev_fp"), ("address", "addr_fp"))

    # ---- ground-truth census (test) --------------------------------------- #
    gt = {}
    for name, col in KINDS:
        s = _exact_stats(ft, col)
        gt[name] = {
            "fingerprints_in_window": int(s["in_window"].sum()),
            "fraud_rings": int((s["is_fraud_ring"] & s["in_window"]).sum()),
            "cross_merchant_fraud_rings": int(
                (s["is_fraud_ring"] & s["cross_merchant"] & s["in_window"]).sum()),
            "single_merchant_fraud_rings": int(
                (s["is_fraud_ring"] & ~s["cross_merchant"] & s["in_window"]).sum()),
        }

    truth_fps = {name: _truth_fps(_exact_stats(ft, col)) for name, col in KINDS}

    def eval_centralised() -> dict:
        per = {name: _pr_against(_flag(_exact_stats(ft, col), thr[name]), truth_fps[name])
               for name, col in KINDS}
        return per | {"combined": _combined(per["device"], per["address"])}

    def _one_run(epsilon, poison, defend, th, rep):
        per, roots, audits = {}, {}, []
        for name, col in KINDS:
            reveals, commits = _merchant_histograms(ft, col, epsilon, poison, rep)
            fstats, root, audit = _federated_aggregate(reveals, commits, defend)
            per[name] = _pr_against(_flag(fstats, th[name]), truth_fps[name])
            roots[name] = root
            audits.extend([{**a, "kind": name} for a in audit])
        return {**per, "combined": _combined(per["device"], per["address"]),
                "merkle_root": roots["device"], "rejections": audits}

    def eval_federated(epsilon: float, poison: str | None = None,
                       defend: bool = True, thresholds: dict | None = None) -> dict:
        th = thresholds or thr
        reps = 1 if not np.isfinite(epsilon) else N_DP_REPEATS
        runs = [_one_run(epsilon, poison, defend, th, r) for r in range(reps)]
        if reps == 1:
            return {**runs[0], "thresholds": th}
        # average P/R/F1 across independent noise draws; report the spread
        out = {"thresholds": th, "n_repeats": reps,
               "merkle_root": runs[0]["merkle_root"],
               "rejections": runs[0]["rejections"]}
        for key in ("device", "address", "combined"):
            for metric in ("precision", "recall", "f1"):
                vals = [r[key][metric] for r in runs]
                out.setdefault(key, {})[metric] = round(float(np.mean(vals)), 4)
                out[key][metric + "_std"] = round(float(np.std(vals)), 4)
        return out

    report = {
        "meta": {
            "phase": 8, "seed": SEED, "n_merchants": N_MERCHANTS, "cap_txns": CAP,
            "entity": "device fingerprint + shipping-address tuple",
            "cross_merchant_ring": "2<=txns<=CAP, >=2 fraud txns, spans >=2 merchants",
            "scorer": ("Phase-1 LightGBM; each merchant buckets its own transactions "
                       f"into {N_BUCKETS} risk buckets ({BUCKET_EDGES.tolist()}). "
                       "Identical scorer/bucketing across both arms."),
            "flag_rule": ("spans >=2 merchants AND 2<=txns<=CAP AND risk-estimate "
                          "(bucket-midpoint weighted mean) >= tuned threshold"),
            "threshold_tuned_on": "validation (best F1 over all val fraud rings), applied once to test",
            "released_per_merchant": ("salted-HMAC fingerprint -> Gaussian-DP-noised "
                                      "risk-bucket count vector (L2 sensitivity 1 -> one sigma, "
                                      "no budget split)"),
            "dp": {"mechanism": "Gaussian mechanism on the per-fingerprint bucket vector",
                   "delta": DELTA},
            "robust_aggregation": (f"reject a merchant whose total volume > {NORM_RATIO}x "
                                   f"the median merchant, or whose aggregate risk-estimate > "
                                   f"{RISK_RATIO}x the median; plus the commit/reveal digest check"),
            "thresholds": thr,
            "limitations": [
                "synthetic merchants (hash(card1) % N) — no merchant column in IEEE-CIS",
                "scorer is centralised LightGBM held fixed; a fully federated stack "
                "would use the Phase-4 federated MLP (Phase 4: ~0.01 PR-AUC vs "
                "centralised) — the two costs compose",
                "DP-noised histogram sharing is a simplified stand-in for a real "
                "private-set-intersection / secure-aggregation protocol",
                "silent (non-participation) poisoning is an incentive/audit problem, "
                "not defensible at the aggregation layer — only active (hot-flood) "
                "poisoning is shown defended",
            ],
        },
        "ground_truth": gt,
        "centralized": eval_centralised(),
        "federated_no_dp": eval_federated(float("inf")),
        "dp_sweep": [
            {"epsilon": (None if not np.isfinite(e) else e),
             **eval_federated(e, thresholds=(thr if not np.isfinite(e)
                                             else _tune_threshold(fv, e)))}
            for e in EPS_SWEEP
        ],
        "poison_demo": {
            "hot_flood_no_defense": eval_federated(float("inf"), poison="hot_flood", defend=False),
            "hot_flood_defended": eval_federated(float("inf"), poison="hot_flood", defend=True),
            "commit_mismatch_defended": eval_federated(float("inf"),
                                                       poison="commit_mismatch", defend=True),
        },
    }
    OUT.write_text(json.dumps(report, indent=2))

    c = report["centralized"]["combined"]
    f = report["federated_no_dp"]["combined"]
    print("\nFEDERATED CROSS-MERCHANT RING DETECTION — held-out test")
    print(f"  ground truth: "
          f"{sum(g['cross_merchant_fraud_rings'] for g in gt.values())} cross-merchant "
          f"fraud rings ({sum(g['single_merchant_fraud_rings'] for g in gt.values())} "
          f"single-merchant, for context)")
    print(f"  centralised (sees all)  : P {c['precision']:.2f}  R {c['recall']:.2f}  F1 {c['f1']:.2f}")
    print(f"  federated,  no DP       : P {f['precision']:.2f}  R {f['recall']:.2f}  F1 {f['f1']:.2f}")
    for row in report["dp_sweep"]:
        e = row["epsilon"]; cc = row["combined"]
        print(f"  federated,  eps={str(e):>4}    : P {cc['precision']:.2f}  R {cc['recall']:.2f}  "
              f"F1 {cc['f1']:.2f}  (rejected {len(row['rejections'])})")
    pn = report["poison_demo"]["hot_flood_no_defense"]["combined"]
    pd_ = report["poison_demo"]["hot_flood_defended"]["combined"]
    print(f"  poison hot-flood, no def : P {pn['precision']:.2f}  R {pn['recall']:.2f}  F1 {pn['f1']:.2f}")
    print(f"  poison hot-flood, defended: P {pd_['precision']:.2f}  R {pd_['recall']:.2f}  F1 {pd_['f1']:.2f}  "
          f"(rejected {sorted({r['merchant'] for r in report['poison_demo']['hot_flood_defended']['rejections']})})")
    print(f"\nWrote {OUT}")


if __name__ == "__main__":
    main()
