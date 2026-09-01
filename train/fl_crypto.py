"""Tamper-evidence + differential-privacy primitives shared by the federated
experiments (train/fl_rings.py) and the live federated detector
(backend/app/fl_live.py).

Pure stdlib + numpy — no torch / flwr — so it imports from the core `.venv`
and from the backend without pulling the Phase-4 stack.
"""
from __future__ import annotations

import hashlib
import hmac
import json

import numpy as np


def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def commit_histogram(hist: dict) -> str:
    """SHA-256 over a canonical serialization of a merchant's reported histogram
    (the commit value in the commit/reveal protocol)."""
    return sha256_hex(json.dumps(hist, sort_keys=True, separators=(",", ":")).encode())


def merkle_root(leaves: list[str]) -> str:
    """Binary Merkle-tree root over an ordered list of hex leaf digests.
    Odd layers duplicate the last node. Empty -> sha256(b'')."""
    if not leaves:
        return sha256_hex(b"")
    layer = [bytes.fromhex(x) for x in leaves]
    while len(layer) > 1:
        if len(layer) % 2:
            layer.append(layer[-1])
        layer = [hashlib.sha256(layer[i] + layer[i + 1]).digest()
                 for i in range(0, len(layer), 2)]
    return layer[0].hex()


def fingerprint_hmac(salt: bytes, value: str) -> str:
    """Salted HMAC-SHA256 of an entity identifier. What a merchant node shares
    instead of the raw card/device string — the aggregator can still intersect
    identical entities across merchants, but cannot read them back."""
    return hmac.new(salt, value.encode(), hashlib.sha256).hexdigest()[:16]


def gaussian_sigma(epsilon: float, delta: float, sensitivity: float) -> float:
    """Std-dev of the Gaussian mechanism for one (epsilon, delta) release of a
    query with the given L2 sensitivity. epsilon<=0 or inf -> 0 (no noise)."""
    if not np.isfinite(epsilon) or epsilon <= 0:
        return 0.0
    return float(np.sqrt(2.0 * np.log(1.25 / delta)) * sensitivity / epsilon)


class DPHistogram:
    """A merchant node's per-entity risk-bucket histogram with the Gaussian
    mechanism applied. For each salted-HMAC fingerprint the merchant reports a
    vector of transaction counts per risk bucket (its own scorer bucketed each
    txn on-device). Adding/removing one transaction changes exactly one bucket
    by 1 -> the whole vector has L2 sensitivity 1, so one sigma (from the full
    epsilon) is applied independently to every bucket -- no budget split, and
    more risk detail than a single count at the same privacy cost."""

    def __init__(self, epsilon: float, delta: float = 1e-5, seed: int = 0):
        self.epsilon = epsilon
        self.delta = delta
        self.rng = np.random.default_rng(seed)
        self._sigma = gaussian_sigma(epsilon, delta, sensitivity=1.0)

    def release(self, bucket_hist: dict[str, list[float]]) -> dict[str, list[float]]:
        out: dict[str, list[float]] = {}
        for k, vec in bucket_hist.items():
            noised = [max(0.0, round(c + (self.rng.normal(0, self._sigma)
                                          if self._sigma else 0.0))) for c in vec]
            if sum(noised) >= 1:
                out[k] = noised
        return out

    @property
    def sigma(self) -> float:
        return self._sigma
