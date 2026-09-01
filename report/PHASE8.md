# Phase 8 — federated cross-merchant ring detection

**The core claim, measured.** `project_sentinel.md`: *"the same cross-merchant
fraud-ring signal Vulcan gets by centralising raw data, but without any merchant
sharing raw transactions — through Federated Learning and Differential Privacy."*

This experiment builds that detector two ways and compares them on the held-out
temporal test set:

| arm | how the per-fingerprint stats are computed |
|---|---|
| **centralised** | pool every transaction; compute (txn count, risk estimate, distinct merchants) exactly. The "sees everything" oracle. |
| **federated** | each of 8 synthetic merchants sees only its own rows. Per **salted-HMAC** device / address fingerprint it releases a **Gaussian-DP-noised risk-bucket count vector** — never raw rows, emails, amounts or card numbers. Reports are committed (SHA-256) then revealed; commitments are hashed into a **Merkle root**. A robust aggregator rejects a merchant whose reported volume or aggregate risk is a gross outlier, or whose reveal ≠ its commitment. Surviving histograms are summed and the **same** flag rule is applied. |

The per-transaction risk **score is held identical** across both arms (the
Phase-1 LightGBM, each merchant bucketing its own transactions into 5 risk
buckets on-device), so the comparison isolates the cost of federating the
ring-detection layer — nothing else moves.

Regenerate: `make fl-rings` → `report/fl_ring_metrics.json` (core `.venv`, seed 42,
fully reproducible; ~1 min).

---

## What counts as a ring

Entity = **device fingerprint** (`DeviceInfo|id_31`, must look like a real device)
or **shipping-address tuple** (`addr1|addr2|P_emaildomain`). A card's rows all
hash to one merchant (merchant = `hash(card1) % 8`), so cards cannot span
merchants on this partition; devices and addresses can — and the concept doc
names "shared device fingerprints / IP clusters" as the cross-merchant signal.

**Ground truth** (built centrally with true labels — an eval oracle, not part of
any detector): a fingerprint is a *cross-merchant fraud ring* if `2 ≤ txns ≤ 40`,
it has `≥ 2` fraudulent transactions, and its transactions span `≥ 2` merchants.

Held-out test census: **118 cross-merchant fraud rings** (83 device + 35 address).
For context, only **20** fraud rings are single-merchant — a lone merchant
already catches those; federation is for the 85 % that no single merchant sees.

**Flag rule (identical in both arms):** `spans ≥ 2 merchants` **and**
`2 ≤ txns ≤ 40` **and** `risk-estimate ≥ threshold`, where risk-estimate is the
bucket-midpoint weighted mean (a bounded estimate of mean member risk). One
threshold per ring kind, tuned on **validation** for best F1 over all val fraud
rings, applied once to test.

---

## Result 1 — federating the computation is free

Held-out test, combined device + address:

| detector | precision | recall | F1 |
|---|---|---|---|
| centralised (sees every transaction) | **0.68** | **0.66** | **0.67** |
| federated, no DP (8 merchants, commit/reveal + Merkle + robust agg) | **0.68** | **0.66** | **0.67** |

**Identical** — same 78 true positives, 36 false positives, 40 misses. Moving the
computation onto the merchants, with the tamper-evidence machinery running,
costs *nothing*. Raw transactions never leave a merchant.

## Result 2 — differential privacy: where the trade-off bites

Each row is the mean of **5 independent noise draws**; the detector threshold is
re-calibrated on validation for each ε (never on test).

| target ε | precision | recall | F1 | Δ F1 vs no-DP |
|---|---|---|---|---|
| ∞ (no DP) | 0.68 | 0.66 | 0.67 | — |
| 32 | 0.67 | 0.66 | 0.66 | −0.01 |
| 16 | 0.95 | 0.27 | 0.43 | −0.24 |
| 8  | 0.77 | 0.26 | 0.39 | −0.28 |
| 4  | 0.40 | 0.26 | 0.32 | −0.35 |
| 2  | 0.23 | 0.21 | 0.22 | −0.45 |
| 1  | 0.09 | 0.07 | 0.08 | −0.59 |

**DP is free down to ε ≈ 32; below ε ≈ 16 the trade-off is real.** The cost is
concentrated in *small* cross-merchant rings: a ring of 3–4 transactions spread
across merchants means each merchant contributes a per-fingerprint count of ~1,
and the Gaussian noise floor (σ ≈ 0.30 at ε = 16, σ ≈ 4.8 at ε = 1) erases those
single-transaction contributions — the ring loses its `≥ 2 merchants` status and
drops out. What survives at tight ε is high-precision (ε = 16: P 0.95) but only a
quarter of the rings.

**This regime widens with per-merchant volume.** The knee sits at ε ≈ 16–32 on
88 k test transactions across 8 merchants; a merchant with Vulcan-scale history
(billions of transactions) has per-fingerprint counts orders of magnitude larger,
pushing the same knee far below ε = 1. We report the honest cost at *this* scale
rather than claiming DP is free.

## Result 3 — Byzantine robustness

One malicious merchant (0 of 8), held-out test, combined:

| | no defense | SentinelFedAvg-style aggregator |
|---|---|---|
| **hot-flood** (merchant 0 reports every fingerprint as maximum-risk) | P 0.22 · R 0.81 · **F1 0.34** | P 0.65 · R 0.57 · **F1 0.61** — merchant 0 rejected as a risk-estimate outlier (both ring kinds) |
| **commit-mismatch** (commits an honest digest, reveals the flood) | — | **F1 0.61** — rejected by the SHA-256 commitment check before aggregation |

A single unchecked merchant marking everything high-risk drags precision to 0.22
(false cross-merchant rings everywhere). The robust aggregator rejects that
merchant every run and recovers to ≈ the honest federated number. Commit/reveal
specifically stops the *adaptive* attacker — the digest is locked before any
other report is seen.

**Merkle root** of each round's 8 commitments is logged
(`fl_ring_metrics.json → federated_no_dp.merkle_root`). In production it would be
anchored to a public timestamping service; here it is computed and logged as a
tamper-evident record, stated as a proposal.

---

## Honest limitations

- **Synthetic merchants** — IEEE-CIS has no merchant column, so merchants are
  `hash(card1) % 8` (the Phase-4 partition). Non-IID by construction.
- **The scorer is centralised LightGBM, held fixed.** A fully federated stack
  would score with the Phase-4 federated MLP (Phase 4: federated ≈ centralised,
  ~0.01 PR-AUC gap). The two costs compose; this experiment measures only the
  ring-layer cost, with the scorer removed as a confound. LightGBM is used
  because it is the stronger scorer and the ring numbers should not be dragged
  down by the known trees-beat-nets-on-tabular gap.
- **DP-noised histogram sharing is a simplified stand-in** for a real
  private-set-intersection / secure-aggregation protocol — the aggregator still
  sees per-merchant per-fingerprint noised counts, where a production system
  would reveal only the final intersection. Stated, like the Merkle root.
- **Silent (non-participation) poisoning** — a merchant that simply withholds
  reports — is an incentive/audit problem, not defensible at the aggregation
  layer. Only active (hot-flood) poisoning is shown defended.
- **Recall ceiling.** Device data is present on ~24 % of transactions, so a
  device/address-fingerprint ring detector — federated or not — can only ever
  cover the coordinated slice of fraud that carries those identifiers.

## Reading

The strong claims are **federated == centralised with no DP** (federating the
computation is free) and the **poison before/after** (a malicious merchant is
caught and the signal recovers). The DP curve is an honest cost, not a triumph:
at this data scale meaningful privacy (ε ≤ 8) halves ring-detection F1, and that
cost shrinks as per-merchant volume grows.
