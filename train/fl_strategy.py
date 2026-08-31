"""
Phase 4 — SentinelFedAvg: a Byzantine-robust federated aggregation strategy.

Subclasses Flower's `FedAvg` and reuses `flwr.common` parameter/message types and
`flwr.server.strategy.aggregate` (the standard weighted-mean), but replaces plain
FedAvg's "trust every update, weight by row count" with the four defences from
`project_sentinel.md` section 4.1:

  1. Commit / reveal  — each round is two phases. Clients first send only a
     SHA-256 digest of their update (commit); once every digest is in, they send
     the update itself (reveal). A reveal whose digest != its commitment is
     rejected. This removes the last-mover advantage: a client cannot wait to see
     other updates and then craft a poisoned one, because its digest is already
     locked.
  2. Merkle root      — all of a round's commitment digests are hashed into a
     binary Merkle tree; the root is logged. In production it would be anchored
     to a timestamping service — here it is computed + printed as a tamper-evident
     record of what each client committed to.
  3. Anomaly filter   — updates are converted to deltas from the current global
     model, then screened in two stages:
       a. norm test   — a delta whose L2 norm exceeds NORM_RATIO x the median
          delta norm is rejected (catches scaled model-negation attacks).
       b. cosine test  — among norm survivors, a delta whose cosine distance from
          the coordinate-wise median delta is BOTH a strong outlier
          (> median + COS_MAD_K x MAD) AND past COS_FLOOR is rejected (catches
          direction-reversal attacks that keep an honest-looking magnitude).
     The relative rule is deliberate: `project_sentinel.md`'s fixed
     "cosine distance > 0.3" rejected every honest client under this non-IID
     partition (their updates genuinely diverge once the model is near a
     minimum), so the shipped filter flags outliers relative to the round's own
     spread instead.
  4. Balance-weighted aggregation — survivors are averaged with weight = their
     fraud (minority-class) row count, not their total row count, so a
     high-volume low-fraud merchant cannot dilute the signal.

The round loop is driven in-process by `train/fl_experiment.py` (Ray-backed
`flwr.simulation.run_simulation` is not used — no Ray wheel for Python 3.14, and
the two-phase protocol does not map cleanly onto one request/response per round).
"""
from __future__ import annotations

import hashlib

import numpy as np
from flwr.common import Parameters, ndarrays_to_parameters, parameters_to_ndarrays
from flwr.server.strategy import FedAvg
from flwr.server.strategy.aggregate import aggregate

COS_THRESHOLD = 0.3  # project_sentinel.md 4.1 value — kept as the reported design target
NORM_RATIO = 3.0     # reject delta with L2 norm > NORM_RATIO x median norm
COS_MAD_K = 3.0      # reject cosine distance > median + COS_MAD_K x MAD ...
COS_FLOOR = 0.85     # ... and only if it is also past this absolute floor


def digest_ndarrays(nds: list[np.ndarray]) -> str:
    """Deterministic SHA-256 over an ordered list of arrays (the commit value)."""
    h = hashlib.sha256()
    for a in nds:
        a = np.ascontiguousarray(a, dtype=np.float32)
        h.update(str(a.shape).encode())
        h.update(a.tobytes())
    return h.hexdigest()


def merkle_root(digests: list[str]) -> str:
    """Binary Merkle tree root over the (already order-fixed) leaf digests."""
    if not digests:
        return hashlib.sha256(b"").hexdigest()
    layer = [bytes.fromhex(d) for d in digests]
    while len(layer) > 1:
        if len(layer) % 2:
            layer.append(layer[-1])
        layer = [hashlib.sha256(layer[i] + layer[i + 1]).digest()
                 for i in range(0, len(layer), 2)]
    return layer[0].hex()


def _flat(nds: list[np.ndarray]) -> np.ndarray:
    return np.concatenate([a.ravel() for a in nds]).astype(np.float64)


def _cos_dist(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-12 or nb < 1e-12:
        return 1.0
    return float(1.0 - np.dot(a, b) / (na * nb))


class SentinelFedAvg(FedAvg):
    def __init__(self, n_merchants: int, norm_ratio: float = NORM_RATIO,
                 cos_mad_k: float = COS_MAD_K, cos_floor: float = COS_FLOOR, **kw):
        super().__init__(min_available_clients=n_merchants,
                         min_fit_clients=n_merchants, **kw)
        self.norm_ratio = norm_ratio
        self.cos_mad_k = cos_mad_k
        self.cos_floor = cos_floor
        self.commitments: dict[int, dict[int, str]] = {}
        self.merkle_roots: dict[int, str] = {}
        self.audit: list[dict] = []

    # ---- phase 1: commit ------------------------------------------------- #
    def receive_commitments(self, server_round: int, commits: dict[int, str]) -> str:
        self.commitments[server_round] = dict(commits)
        leaves = [commits[c] for c in sorted(commits)]  # order fixed by cid
        root = merkle_root(leaves)
        self.merkle_roots[server_round] = root
        return root

    # ---- phase 2: reveal + robust aggregate ---------------------------- #
    def aggregate_reveal(
        self,
        server_round: int,
        global_params: Parameters,
        reveals: dict[int, list[np.ndarray]],
        fraud_counts: dict[int, int],
    ) -> tuple[Parameters, dict]:
        g = parameters_to_ndarrays(global_params)
        committed = self.commitments.get(server_round, {})

        # 1. commitment verification
        verified: dict[int, list[np.ndarray]] = {}
        rejected: list[dict] = []
        for cid, upd in reveals.items():
            if digest_ndarrays(upd) != committed.get(cid):
                rejected.append({"round": server_round, "merchant": cid,
                                 "reason": "commitment_mismatch"})
            else:
                verified[cid] = upd

        # 2. anomaly filter on deltas from the current global model
        deltas = {cid: [u - w for u, w in zip(upd, g)] for cid, upd in verified.items()}
        flat = {cid: _flat(d) for cid, d in deltas.items()}
        survivors = list(deltas)

        if len(deltas) >= 3:
            # 2a. norm test
            norms = {cid: float(np.linalg.norm(fv)) for cid, fv in flat.items()}
            med_norm = float(np.median(list(norms.values())))
            survivors = []
            for cid in deltas:
                if med_norm > 1e-12 and norms[cid] > self.norm_ratio * med_norm:
                    rejected.append({"round": server_round, "merchant": cid,
                                     "reason": "norm_outlier",
                                     "norm_ratio": round(norms[cid] / med_norm, 3)})
                else:
                    survivors.append(cid)

            # 2b. relative cosine test among norm survivors
            if len(survivors) >= 3:
                med_vec = np.median(np.stack([flat[c] for c in survivors]), axis=0)
                dists = {c: _cos_dist(flat[c], med_vec) for c in survivors}
                dv = np.array(list(dists.values()))
                cut = float(np.median(dv) + self.cos_mad_k *
                            np.median(np.abs(dv - np.median(dv))))
                kept = []
                for c in survivors:
                    if dists[c] > max(cut, self.cos_floor):
                        rejected.append({"round": server_round, "merchant": c,
                                         "reason": "cosine_outlier",
                                         "cos_distance": round(dists[c], 4)})
                    else:
                        kept.append(c)
                survivors = kept

        # 3. balance-weighted aggregation of surviving deltas
        if len(survivors) < 2:
            new_params = global_params  # not enough trustworthy updates — hold
            applied = []
        else:
            weighted = [(deltas[cid], max(int(fraud_counts.get(cid, 1)), 1)) for cid in survivors]
            agg_delta = aggregate(weighted)
            new_g = [w + d for w, d in zip(g, agg_delta)]
            new_params = ndarrays_to_parameters(new_g)
            applied = survivors

        self.audit.extend(rejected)
        info = {
            "round": server_round,
            "merkle_root": self.merkle_roots.get(server_round, ""),
            "revealed": sorted(reveals),
            "rejected": rejected,
            "aggregated_merchants": sorted(applied),
            "weighting": "balance-weighted by fraud row count",
        }
        return new_params, info
