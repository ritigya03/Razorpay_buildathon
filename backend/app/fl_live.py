"""Live federated cross-merchant detection over the Razorpay test-mode payments.

The runtime counterpart of train/fl_rings.py. Each synthetic merchant node holds
only its own payments; it releases, per SALTED-HMAC card fingerprint, a
risk-bucket count vector (optionally Gaussian-DP-noised). The aggregator hashes
the commitments into a Merkle root, sums the vectors, and flags a fingerprint
that spans >= 2 merchants with an elevated risk estimate — a coordinated card no
single merchant could see alone. Nothing else (card numbers, emails, contacts,
amounts) leaves a node.

A centralised pass over the same payments is returned alongside for the
side-by-side "the central view would find this too — but it needs the raw data".
"""
from __future__ import annotations

import sys

import numpy as np
from sqlmodel import Session, select

from .config import REPO_ROOT
from .models import Transaction

sys.path.insert(0, str(REPO_ROOT / "train"))
from fl_crypto import (  # noqa: E402
    DPHistogram, N_BUCKETS, bucketize, commit_histogram, fingerprint_hmac,
    merkle_root, risk_estimate,
)

SALT = b"sentinel-fl-live-v1"
DELTA = 1e-5
CAP = 25                 # matches the runtime ring engine's ring_cap
LIVE_RISK_THR = 0.35     # bucket-weighted risk estimate to flag a live ring


def _bucket_vec(scores: list[float]) -> list[float]:
    v = np.bincount(bucketize(scores), minlength=N_BUCKETS).astype(float)
    return v.tolist()


def run_detection(session: Session, epsilon: float | None = None) -> dict:
    txns = session.exec(
        select(Transaction).where(Transaction.source == "razorpay")  # noqa: E712
    ).all()
    rows = [t for t in txns if t.merchant and t.card_id]

    # ---- per-merchant nodes: release only salted-HMAC fp -> risk-bucket vector
    nodes: dict[str, list[Transaction]] = {}
    for t in rows:
        nodes.setdefault(t.merchant, []).append(t)

    reveals: dict[str, dict[str, list[float]]] = {}
    commits: dict[str, str] = {}
    sketches: list[dict] = []
    for seed, (merchant, mrows) in enumerate(sorted(nodes.items())):
        by_fp: dict[str, list[float]] = {}
        for t in mrows:
            by_fp.setdefault(fingerprint_hmac(SALT, t.card_id), []).append(t.score)
        hist: dict[str, list[float]] = {h: _bucket_vec(s) for h, s in by_fp.items()}
        if epsilon:
            hist = DPHistogram(float(epsilon), DELTA, seed=seed).release(hist)
        commits[merchant] = commit_histogram(hist)
        reveals[merchant] = hist
        sketches.append({
            "merchant": merchant,
            "payments": len(mrows),
            "commitment": commits[merchant][:16],
            "entries": [{"fingerprint": k[:12], "buckets": [int(x) for x in v],
                         "count": int(sum(v))} for k, v in sorted(hist.items())],
        })

    root = merkle_root([commits[m] for m in sorted(commits)])

    # ---- aggregate the revealed vectors
    fps = {k for h in reveals.values() for k in h}
    fed_rings = []
    for fp in fps:
        contribs = [np.asarray(h[fp], float) for h in reveals.values() if fp in h]
        vec = np.sum(contribs, axis=0)
        n = float(vec.sum())
        merch = sum(1 for c in contribs if c.sum() >= 1)
        re = float(risk_estimate(vec[None, :])[0])
        flagged = bool(merch >= 2 and 2 <= n <= CAP and re >= LIVE_RISK_THR)
        if merch >= 2 or flagged:
            fed_rings.append({"fingerprint": fp[:12], "payments": int(round(n)),
                              "merchants": merch, "risk_estimate": round(re, 3),
                              "flagged": flagged})

    # ---- centralised comparison (needs every raw payment in one place)
    by_card: dict[str, list[Transaction]] = {}
    for t in rows:
        by_card.setdefault(t.card_id, []).append(t)
    cen_rings = []
    for card, crows in by_card.items():
        merch = len({r.merchant for r in crows})
        n = len(crows)
        re = float(risk_estimate(
            np.bincount(bucketize([r.score for r in crows]),
                        minlength=N_BUCKETS)[None, :])[0])
        flagged = bool(merch >= 2 and 2 <= n <= CAP and re >= LIVE_RISK_THR)
        if merch >= 2 or flagged:
            cen_rings.append({"card": card, "payments": n, "merchants": merch,
                              "risk_estimate": round(re, 3), "flagged": flagged})

    return {
        "epsilon": epsilon,
        "merchant_sketches": sketches,
        "merkle_root": root,
        "federated_rings": sorted(fed_rings, key=lambda r: -r["risk_estimate"]),
        "centralized_rings": sorted(cen_rings, key=lambda r: -r["risk_estimate"]),
        "flagged_federated": sum(1 for r in fed_rings if r["flagged"]),
        "flagged_centralized": sum(1 for r in cen_rings if r["flagged"]),
        "note": (
            "each merchant released only salted-HMAC card fingerprints + "
            "risk-bucket counts"
            + (f", Gaussian-DP-noised at ε={epsilon}" if epsilon else "")
            + " — no card numbers, emails, contacts or amounts left any node. "
            "The centralised pass needs every raw payment pooled in one place."
        ),
    }
