# Phase 2 — ring / entity-graph work

## 2a. Explicit ring features — a negative result (kept on purpose)

Added 27 leakage-safe ring/entity features on top of the Phase 1 baseline:

- **Reputation** — out-of-fold target encoding (5 sequential time blocks) for
  `card_id`, `addr1`, `P_emaildomain`; encoded from all of train for val/test.
  (Without OOF the feature is a perfect label proxy on train and near-constant
  out-of-sample — that version collapsed the model, PR-AUC 0.17.)
- **Velocity** — causal trailing-window counts / amounts / inter-transaction gap
  for `uid` and `card_id`; cumulative distinct fan-out (one card → many accounts).
- **Ring structure** — 7-day sliding union-find over `card_id ↔ device_id`:
  component size, distinct cards / devices / accounts, transactions-per-hour.

**Result on the held-out test split:**

| | baseline | + 27 ring features |
|---|---|---|
| PR-AUC | 0.546 | 0.549 |
| ROC-AUC | 0.905 | 0.906 |
| precision @ recall 0.50 | 0.595 | 0.609 |

The ring features are used heavily (`r_card_id_rep_rate` is the model's #1
feature by gain; ~18 of the top 30 are ring features) but held-out accuracy does
not move. The pre-engineered Vesta columns (`C1–C14` counts, `D1–D15` time
deltas, `V1–V339` aggregations) plus LightGBM's categorical handling already
capture the entity/velocity/ring signal. Our hand-built versions are redundant.

**We kept the baseline as the graded transaction scorer** and did not adopt the
ring-feature model. `train/model_ring.py` and `report/metrics_ring.json` stay in
the repo as the documented experiment.

## 2b. The ring engine — triage, measured at ring level

The ring layer's value is not accuracy, it is **turning an alert flood into a
short, explainable review queue**. `train/ring_engine.py` scores every
transaction with the baseline model, then groups transactions into:

- **device rings** — one specific device fingerprint used by 2–25 distinct
  accounts in the window;
- **address rings** — one exact `(addr1, addr2, P_emaildomain)` tuple used by
  2–25 distinct card identities.

No transitive union-find (that produced 65k-node mega-blobs). A ring is flagged
when its members' mean risk clears a threshold tuned on validation for best
ring-F1. A "fraud ring" has ≥ 2 fraudulent transactions.

**Held-out test results:**

| ring type | groups | fraud rings | flagged | precision | recall | F1 |
|---|---|---|---|---|---|---|
| device | 433 | 92 | 92 | **0.75** | **0.75** | **0.75** |
| address | 586 | 42 | 40 | 0.42 | 0.40 | 0.41 |

- **Alert-volume reduction:** 18,001 transaction-level alerts → **132 ring
  alerts** (≈ 136× fewer items to review).
- **Fraud coverage:** flagged rings contain 510 / 3,083 fraud transactions
  (16.5% of all fraud). This is the ceiling for a device/address ring engine —
  only ~24% of transactions carry device data, and most fraud is single-account,
  not coordinated. Stated openly rather than hidden.
- The engine catches a handful of frauds whose own transaction score was below
  the operating threshold (caught only because their ring was flagged).

**Reading:** the transaction model handles fraud broadly (PR-AUC 0.55, recall
0.84 at the cost-optimal point); the ring engine takes the coordinated slice and
makes it triage-able at 0.75 / 0.75 ring precision/recall with 100× less review
volume. They are complementary, and both numbers are measured on data the models
never saw.
