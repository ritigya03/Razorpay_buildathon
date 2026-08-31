# Project Sentinel

**Razorpay AI Buildathon — Track 02: AI Risk Manager**

A privacy-preserving detector for **coordinated fraud rings** — the class of loss
where one actor spreads many small transactions across cards, devices and
merchants so that no single merchant sees enough to react. Sentinel builds a
network-level view from shared-entity structure, scores each transaction, and
reports its accuracy honestly: **precision / recall on a held-out temporal test
set, with the false-positive cost quantified.**

Concept & positioning: [`project_sentinel.md`](project_sentinel.md).
Running engineering journal (decisions, dead ends, bugs): [`DEVLOG.md`](DEVLOG.md).

> **Defense-only.** Sentinel is advisory. It emits a 0–100 risk score; flagged
> transactions are routed to a human reviewer. Nothing is auto-blocked, and the
> repo contains no offense-capable code. Ring/entity features are always computed
> from data at or before the training cutoff — never from the test period.

---

## Status

| Phase | What | State |
|---|---|---|
| **1** | Temporal split + LightGBM baseline (raw features), held-out metrics + cost curve | ✅ done |
| **2** | Ring / entity work: feature experiment (negative) + ring engine (triage) | ✅ done |
| **3** | FastAPI backend: replay feed + Razorpay test-mode Payments/Disputes | ✅ done |
| **6** | React dashboard (Vite + Recharts + SVG ring graph) | ✅ done |
| **4** | Federated learning + DP + Byzantine robustness — side experiment ([details](report/PHASE4.md)) | ✅ done |
| 5 | Agent layer (Claude Agent SDK): explain ring, suggest mitigation | — |

### Phase 1 — transaction scorer (held-out test = final 15% of the timeline, by date)

| Metric | Value |
|---|---|
| PR-AUC (test) | 0.546 |
| ROC-AUC (test) | 0.905 |
| Balanced point (recall ≈ 0.47) | precision **0.60** |
| Cost-optimal point (₹3 / review) | recall **0.84**, precision 0.14, **76% lower** expected loss than doing nothing |

### Phase 2 — ring work ([details](report/PHASE2.md))

- **Explicit ring features: no lift.** +0.003 PR-AUC on held-out — the
  pre-engineered Vesta columns already capture entity/velocity/ring signal.
  Kept as a documented negative result (`model_ring.py`).
- **Ring engine (triage):** groups transactions into coordinated device/address
  rings. Held-out **device rings: precision 0.75, recall 0.75**. Collapses
  18,001 transaction alerts into **132 ring alerts (~136× fewer)**; covers 16.5%
  of all fraud (the coordinated slice — ceiling stated openly).

### Phase 3 — backend ([backend/README.md](backend/README.md))

FastAPI service. A **replay feed** streams the held-out split through the real
model + ring engine at compressed time; **Razorpay test-mode** supplies real
orders and payment/dispute webhooks alongside it. `POST /api/simulate/dispute`
closes the money loop — it disputes a flagged fraudulent transaction and reports
how many hours earlier Sentinel flagged it (~40h in a typical run).
`make backend` runs it; `make test` runs the end-to-end pipeline test.

### Phase 6 — dashboard ([frontend/README.md](frontend/README.md))

React + Vite. Seven tabs: Overview (replay + live metrics), Feed (transaction
stream), Rings (flagged rings + SVG member graph), Disputes (chargeback loop +
lead time), Metrics (held-out PR / cost curves), Federated (Phase 4), Razorpay
(real test-mode checkout). `cd frontend && npm install && npm run dev` →
http://localhost:5173.

### Phase 4 — federated learning + DP ([details](report/PHASE4.md))

**Secondary experiment. Does not touch the graded numbers above.** Turns
`project_sentinel.md` §4 into running code: 8 non-IID synthetic merchants, a
`SentinelFedAvg` strategy built on Flower 1.35 (commit/reveal + a norm/cosine
poison filter + balance-weighted aggregation + a logged Merkle root), and Opacus
DP-SGD per merchant. The model is an **MLP, not the graded LightGBM** (DP-SGD and
FedAvg need a differentiable model). With a literature-guided pass (leakage-safe
engineered features, tanh, large DP batch, FedSWA — citations in
[`report/PHASE4.md`](report/PHASE4.md)) it reaches **PR-AUC 0.39**, in the range
of published *centralized* neural nets on this held-out split (LSTM 0.485,
Transformer 0.409). The point is the *relative* comparisons:

- **federated ≈ centralized** on held-out test (0.384 vs 0.395) — not pooling raw
  data costs ~nothing;
- an **accuracy-vs-ε curve** with a visible knee — ε ≥ 4 is indistinguishable
  from centralized; ε = 1 costs ~0.02 PR-AUC (~5%);
- **1/8 malicious merchants destroys plain FedAvg** (test PR-AUC 0.04, below the
  fraud base rate); `SentinelFedAvg` rejects the poisoned update every round and
  recovers to 0.39 — three attack types (`sign_flip` / `flip` / `last_mover`),
  each caught by a different defense layer (norm test / cosine test /
  commit-reveal hash).

`make fl-deps` (own `.venv-fl` — Flower pins conflict with the backend's
FastAPI), then `make fl` → `report/fl_metrics.json` +
`report/figures/fl_epsilon_curve.png`.

---

## Reproduce

```bash
brew install libomp                 # LightGBM's OpenMP runtime (macOS)
make venv                           # Python 3.14 venv + pinned deps
export SENTINEL_DATA_DIR=/path/to/ieee-fraud-detection
make reproduce                      # splits + baseline -> report/metrics.json
```

Every number in `report/metrics.json` is regenerated by `make reproduce`
(`seed = 42`). See [`data/README.md`](data/README.md) for the dataset.

The Phase 4 side experiment is separate (its own venv, not part of `reproduce`):

```bash
make fl-deps                        # .venv-fl (torch + opacus + flwr)
make fl                             # -> report/fl_metrics.json  (~20 min; --quick ~90s)
```

---

## Layout

```
data/prepare_splits.py    temporal 70/15/15 split (by TransactionDT, never shuffled)
train/common.py           shared feature prep (leakage-safe categorical encoding)
train/evaluate.py         shared training + cost model + report writer
train/baseline.py         Phase 1 scorer            -> report/metrics.json
train/ring_features.py    leakage-safe ring/entity feature builders
train/model_ring.py       Phase 2a experiment      -> report/metrics_ring.json  (negative result)
train/ring_engine.py      Phase 2b ring engine     -> report/ring_metrics.json
report/PHASE2.md          write-up of the ring work
backend/                  FastAPI service            (Phase 3)
frontend/                 React dashboard            (Phase 6)
train/fl_data.py          Phase 4: dense matrices + non-IID merchant partition
train/fl_model.py         Phase 4: MLP + focal loss + centralized trainer
train/fl_strategy.py      Phase 4: SentinelFedAvg (commit/reveal, poison filter, Merkle)
train/fl_client.py        Phase 4: per-merchant client + attack modes
train/fl_experiment.py    Phase 4: runs it all      -> report/fl_metrics.json
report/PHASE4.md          write-up of the FL / DP / robustness experiment
```
