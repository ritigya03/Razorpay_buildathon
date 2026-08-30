# Project Sentinel — Dev Log

A running, chronological account of what was decided, tried, broken, and fixed —
including dead ends. The polished write-ups live in `report/` and `README.md`;
this file is the raw journal. Newest entries at the bottom of each phase.

Dates are absolute. "Held-out test" always means the final 15% of
`train_transaction.csv` by `TransactionDT` (see Phase 1).

---

## Phase 0 — Scoping & research (2026-08-30)

**Track:** Razorpay AI Buildathon, Track 02 "AI Risk Manager". Bar: a working
detector/verifier/auto-responder for ONE class of loss, with measured
precision/recall on a held-out test set, honest false-positive cost,
strictly defense-only.

**Vulcan research (Razorpay's foundation model, launched 18 Aug 2026):**
- Transformer model, ~3T data points / ~4B payments, ~3,000 signals/txn.
- Does routing + fraud + risk + checkout personalisation in one model.
- Network-level training → catches a compromised card the moment it appears at
  multiple unrelated merchants. Claims: 8x intl card fraud, 5x disputed txns.
- Public criticism (DQIndia): no disclosed baseline, no control group, no
  false-positive rates, no methodology, no whitepaper, no third-party audit; no
  explanation of merchant data isolation.
- Agent Studio (built on Anthropic Claude Agent SDK) already ships a production
  "Dispute Responder Agent" → a chargeback auto-responder would compete with an
  existing product. No named pre-transaction fraud-ring agent exists.

**Decisions (locked with the user):**
1. Loss class = **coordinated fraud-ring / abuse-ring detection**.
2. FL scope = centralized model is graded; federated learning (Flower + Opacus
   DP) is a **secondary experiment** (utility-vs-ε), never on the critical path.
3. Razorpay = test-mode Payments API + Disputes webhooks feeding a React
   dashboard; disputes act as ground-truth labels. No auto-responder unless
   time allows.
4. Frontend = **React** (not Streamlit, user's call).
5. Agent = **Claude Agent SDK** (same as Razorpay Agent Studio), not LangGraph.
6. Dataset = **IEEE-CIS Fraud Detection** (Kaggle).

**Dataset study (`/Users/ritigya/Downloads/ieee-fraud-detection/`):**
- `train_transaction.csv` 590,540 × 394; `train_identity.csv` 144,233 × 41,
  joins on `TransactionID`, only 24.4% of txns have identity.
- `test_*.csv` have **no `isFraud`** (Kaggle withholds) → unusable for
  precision/recall. Held-out test must be a temporal slice of train.
- Fraud prevalence 3.50%. `TransactionDT` spans 182 days, fraud rate stable
  (no drift).
- Label propagates: chargeback on a card ⇒ later txns on that card/account
  within 120 days also `isFraud=1` → the label carries account/ring structure
  (the signal we exploit, and a leakage trap).
- Ring signal verified: accounts (uid = card1+card2+card3+card5+addr1+(D1−day))
  with ≥3 txns hold 43.8% of all fraud; 945 accounts that are ≥3 txns & ≥80%
  fraud hold 33.5% of all fraud. corr(early-period uid fraud rate,
  late-period) = 0.895.
- **No merchant column** — only `ProductCD` (5 categories). Cross-merchant
  narrative needs synthetic merchant ids (dashboard/FL only, not model features).

**Environment:** user's Mac has only Python 3.14. Verified working for
pandas 3.0.5 / numpy 2.5.2 / scikit-learn 1.9.0 / lightgbm 4.7.0 (needed
`brew install libomp`). PyTorch/Flower/Opacus for Phase 4 not yet checked on
3.14 — may need a separate 3.12 venv.

---

## Phase 1 — Temporal split + baseline scorer (2026-08-30) — commit `c5ea43b`

**`data/prepare_splits.py`** — join transaction+identity, sort by
`TransactionDT`, cut on the DT *value* (quantiles) into train 70% / val 15% /
test 15%. Value-based cut (not row position) → zero temporal overlap even with
same-second ties. Never shuffled.

| split | rows | fraud % | day range |
|---|---|---|---|
| train | 413,378 | 3.52% | 0.0–119.8 |
| val | 88,581 | 3.43% | 119.8–151.2 |
| test | 88,581 | 3.48% | 151.2–182.0 |

**`train/common.py`** — feature prep. Decisions:
- Drop raw `TransactionDT` (monotonic with the split boundary → model would key
  on position). Derive cyclical `_hour`, `_weekday` instead.
- Non-numeric columns → pandas `category` with the category set fixed from
  **train only** (unseen val/test values → NaN, handled natively by LightGBM).
  Note: pandas ≥3 reads strings as the `str` dtype, not `object` — first
  version tested `dtype == object` and found 0 categoricals; fixed to test
  `is_numeric_dtype`.

**`train/baseline.py` + `train/evaluate.py`** — LightGBM, advisory cost model:
- FP cost = `C_REVIEW = 3` currency units (analyst clears a legit flag).
- FN cost = full `TransactionAmt` of the missed fraud.
- TP cost = 0. "Do nothing" = every fraud is a loss.
- Threshold chosen on **val** (min expected cost), applied **once** to test.

**First run was bad:** `is_unbalance=True` + lr 0.02 + num_leaves 128 →
`best_iteration = 5`, PR-AUC 0.40. Overfit in 5 trees. Fixed params: dropped
`is_unbalance` (only ranking quality matters, threshold comes from the cost
curve), lr 0.05, num_leaves 63, min_child_samples 100, `cat_smooth 20` +
`min_data_per_group 200` (high-cardinality categoricals like DeviceInfo ~1800
values overfit via categorical splits under the temporal shift).

**Held-out test result:**

| metric | value |
|---|---|
| PR-AUC | 0.546 |
| ROC-AUC | 0.905 |
| balanced point (recall ≈ 0.47) | precision 0.60 |
| cost-optimal point (C_REVIEW=3) | recall 0.84, precision 0.14, 76% lower expected loss than do-nothing |

`best_iteration ≈ 609`. Top features by gain: `card1`, `card2`, `addr1`,
`TransactionAmt`, `DeviceInfo` — shared-entity identifiers dominate before any
ring feature exists.

precision 0.14 at the cost-optimal point is by design: at ₹3/review the
optimizer floods the queue because reviews are cheap vs a ~₹150 average missed
fraud. `C_REVIEW` is tunable; the balanced point is reported alongside.

---

## Phase 2 — Ring / entity work (2026-08-30) — commit `aa91fc0`

User decision after the results below: **pivot the ring layer to triage +
explainability**, measured at ring level — not a PR-AUC claim.

### 2a. Explicit ring features — negative result (kept as documentation)

`train/ring_features.py`, three families, all meant to be leakage-safe:
- **A. Reputation** — entity fraud-rate / count / seen-flag.
- **B. Velocity** — trailing-window counts, amounts, inter-txn gap; cumulative
  distinct fan-out (one card → many accounts).
- **C. Ring structure** — 7-day sliding union-find over `card_id ↔ device_id`:
  component size, distinct cards / devices / accounts, txns-per-hour.

**Bugs fixed along the way:**
- pandas `cannot reindex on an axis with duplicate labels` —
  `groupby().rolling(on='_dt')` returns a result indexed by `_dt`, which has
  many duplicate timestamps. Rewrote to sort per key, take `.to_numpy()` in
  group order, and scatter back by original position.
- numpy `Buffer has wrong number of dimensions (expected 1, got 2)` — union-find
  component roots were tuples `("c", value)`; `np.array([...], dtype=object)`
  made a 2D array. Fixed by mapping roots → integer ids via a dict.

**Attempt 1 (reputation as plain train-period rate): model collapse.**
PR-AUC test 0.546 → **0.170**, ROC-AUC 0.57, `best_iteration = 14`.
Cause: `r_uid_rate_train` is a near-perfect label proxy on train, but `_uid`
only overlaps train↔test **33%** (D1n drifts over time), so on test 67% of rows
fall back to the prior 0.035 — strong on train, near-constant on test.
Gradient boosting over-relied on it and collapsed out-of-sample.
`_device_id` overlap was only 8%.

Entity overlap diagnostic (train → test):
`_uid` 33% · `_card_id` 98% · `_device_id` 8% · `_emailaddr_id` 99.9% ·
`addr1` 90% · `P_emaildomain` 82%.

**Attempt 2 — fix with out-of-fold (OOF) target encoding.**
Reputation restricted to high-overlap entities (`_card_id`, `addr1`,
`P_emaildomain`); `_uid` and `_device_id` used only for velocity/structure.
OOF = 5 sequential time blocks on train; each block encoded from the other
four; val/test encoded from all of train. This makes the train-time feature
realistically noisy so the model learns a sane weight. Dropped
`r_ring_rate_train` entirely (same leakage risk). Set
`early_stopping(first_metric_only=True)` (it had been tracking `binary_logloss`,
which behaved oddly).

**Attempt 3 — B + C only (no reputation):** PR-AUC test **0.545** vs 0.546.
Flat. Velocity + ring structure alone add nothing.

**Attempt 4 — full A + B + C with OOF reputation:** PR-AUC test **0.549**
vs 0.546 (val 0.622 vs 0.612 — slight val gain, test flat).
`r_card_id_rep_rate` is the model's #1 feature by gain and ~18 of the top 30
are ring features — **the model uses them heavily but held-out accuracy does
not move.** They are redundant: the pre-engineered Vesta columns (`C1–C14`
counts, `D1–D15` time deltas, `V1–V339` aggregations) + LightGBM's categorical
handling already carry this signal.

→ Kept the baseline as the graded scorer. `train/model_ring.py` +
`report/metrics_ring.json` retained as the documented experiment.

### 2b. The ring engine — `train/ring_engine.py`

**Attempt 5 — one union-find graph over the test window: mega-blob.**
Edges = shared `_card_id` OR `_device_id`. Result: 2,665 components but the
largest had **65,444 of 88,581 test rows**. Ring precision 0.067.
Cause: `_card_id` = `card1|card2|card3|card5` is a card BIN/type profile, not a
unique card — thousands of unrelated cards share the tuple, so transitive
union-find merges nearly everything. (`D1n = "n"` when `D1` is missing — 50–90%
missing — also collapses many `_uid`s.)

Component-size diagnostic by edge rule: card_id+device → max 65,444;
uid+device → 18,165; uid+specific-device → 5,295. Every variant blobbed.
**Union-find was the wrong approach for this data.**

Device-fingerprint diagnostic: "specific device" = `DeviceInfo` contains a
digit or space and is not a generic OS label. Group by exact `_device_id`,
keep 2–25 distinct accounts → **433 groups, 92 with ≥2 frauds.** Samples were
textbook coordination: `CRO-L03 … chrome 52.0 for android` = 48 txns / 3
accounts / 3 cards / all 48 fraud; `MotoG3 … chrome 66.0` = 27 txns / 22
accounts / 17 cards / 19 fraud.

**Attempt 6 — conservative grouping, no union-find:**
- **device ring** = one specific device fingerprint used by 2–25 distinct
  accounts in the window.
- **address ring** = one exact `(addr1, addr2, P_emaildomain)` tuple used by
  2–25 distinct card identities.
- CAP = 25 filters shared-NAT / kiosk artifacts.
- Ring flagged when members' mean risk ≥ threshold tuned on validation for best
  ring-F1. "Fraud ring" (recall denominator) = ≥ 2 fraudulent transactions.
- Advisory only — a flagged ring is a review item + forensic summary.

**Held-out test result:**

| ring type | groups | fraud rings | flagged | precision | recall | F1 |
|---|---|---|---|---|---|---|
| device | 433 | 92 | 92 | **0.75** | **0.75** | **0.75** |
| address | 586 | 42 | 40 | 0.42 | 0.40 | 0.41 |

- Alert-volume reduction: 18,001 transaction alerts → **132 ring alerts**
  (≈ 136× fewer to review).
- Fraud coverage: flagged rings hold 510 / 3,083 fraud txns (**16.5%** of all
  fraud) — the coordinated slice. Only ~24% of transactions carry device data;
  most fraud is single-account. Ceiling stated openly.
- A few frauds are caught only because their ring was flagged (own score below
  the per-txn threshold).

**Reading:** transaction model handles fraud broadly (PR-AUC 0.55, recall 0.84
at the cost-optimal point); the ring engine takes the coordinated slice and
makes it triage-able at 0.75/0.75 with ~100× less review volume. Complementary;
both measured on unseen data.

---

## Housekeeping

- **2026-08-30** — user asked that commits carry no `Co-Authored-By: Claude` /
  `Claude-Session:` trailers and be authored under their name only. The two
  existing local commits were rebased to strip the trailers (nothing had been
  pushed). Rule recorded; applies to all future commits.

---

## Phase 3 — FastAPI backend (2026-08-30)

**Design constraint discovered:** a Razorpay payment webhook carries ~8 fields
(amount, email, card network, method, contact, created_at). The trained model
needs ~430 IEEE-CIS features. So the "live feed" can't just be real payments.

**Resolution (told the user, proceeded):**
- **Replay feed** — `data/splits/test.parquet` streamed in `TransactionDT`
  order at N days/real-second. Each row scored by the real LightGBM model +
  grouped by the real ring engine. Ground truth is known, so dispute simulation
  and lead-time claims are truth-backed. This is the dashboard's main data.
- **Razorpay test-mode** — real orders + payment/dispute webhooks flow in
  alongside. Live payments get a transparent **rules** score (`app/rules.py`,
  built from the data study's email-domain fraud rates), labelled
  `scorer="rules"` vs the replay's `scorer="model"`.
- **Money loop** — `POST /api/simulate/dispute` raises a dispute against a
  flagged fraudulent replay txn in a flagged ring; response reports the
  lead time (hours we flagged it before the "chargeback").

**Refactor:** `train/common.py` now exposes `build_feature_spec` /
`save_spec` / `load_spec` / `transform`, and training writes
`models/baseline_lgb_feature_spec.json`, so the backend applies the exact
training-time feature transform. `_add_time_features` → public
`add_time_features`. Re-ran `make baseline` (best_iter 1144, PR-AUC test 0.546,
ROC-AUC 0.903 — same as before within noise; the longer run is from
`first_metric_only=True` added in Phase 2).

**Components:** `app/{config,models,db,scoring,rings,replay,rules,razorpay_client,events,main}.py`.
SQLite event store: `Transaction`, `Ring`, `Dispute`, `Alert`. Runtime ring
engine (`app/rings.py`) reuses the Phase-2b device/address-ring definition
(no union-find); default flag threshold read from
`report/ring_metrics.json` (~0.082).

**Bugs fixed during the smoke test:**
- `sqlmodel` has no `func` — import from `sqlalchemy`.
- Sync endpoint calling `asyncio.create_task` fails (no running loop in the
  threadpool) — made `/api/replay/{action}` `async`.
- **Replay virtual clock ran away** — it was free-running on wall-time ×
  days_per_sec while ingestion was capped at `MAX_PER_TICK` rows/tick. The
  clock reached "day 842" while only 30k of 88k rows were in; the ring window
  query `ts >= virtual_now − 7d` then matched nothing → zero rings. Fix: the
  clock now tracks the **data frontier** — `virtual_now = TransactionDT of the
  last ingested row` — so windows always align with real data.
- `session.exec(delete(Ring))` unreliable via SQLModel — switched to
  iterate-and-delete (few hundred rows/rebuild).

**Smoke result (`make test`, 2 tests pass in ~8s):** replay ingests, model
scores (live precision ≈ 0.15, matches the cost-optimal operating point),
device rings form (e.g. `SM-J105B … chrome 65.0 for android`, 3 accounts /
3 cards / mean risk 0.99 / all fraud), `simulate/dispute` returns
`was_flagged=True`, `lead_time_hours ≈ 40`, linked to a flagged ring.

---

## Phase 6 — React dashboard (2026-08-30, done before Phase 4 at user's request)

Vite + React + TypeScript, Recharts for the two curves, hand-drawn SVG for the
ring graph, one `styles.css` (dark). Dev server proxies `/api` + `/webhook` to
:8000. Six tabs: Overview, Feed, Rings, Disputes, Metrics, Razorpay. All poll
every 2–3 s.

Backend additions: `GET /api/report` (serves `metrics.json` + `ring_metrics.json`
for the Metrics tab); `/api/health` now returns the public `razorpay_key_id` for
Checkout.js; `/api/alerts?kind=`.

**Bugs fixed during wiring:**
- `pandas 4` deprecation: `pd.Categorical(values, categories=…)` with
  out-of-category values. `common.transform` now nulls unseen values with
  `.where(isin(cats))` before constructing the Categorical.
- **Stale replay on restart** — `uvicorn --reload` (or any restart) spawned a
  fresh process with `replay.cursor = 0` while the SQLite DB still held the
  previous run's rows; `_ingest_due` then hit primary-key conflicts and the loop
  stalled at `ingested = 0` even though the DB looked full. Fix: lifespan now
  does `replay.load(); replay.reset(); replay.start()` (each service start = a
  clean replay), and `_ingest_due` skips ids already present as a guard.
- Left a stray uvicorn holding :8000 from an earlier manual run — killed via
  `lsof -ti :8000`.

Verified: `npm run build` (tsc + vite) clean; backend + `vite dev` up together;
`/api/report` shape matches the Metrics component; simulated dispute through the
vite proxy returns `was_flagged=true`, `lead_time_hours ≈ 114`. Not screenshot-
verified (no browser tool this session) — user to eyeball at :5173.

---

## Phase 4 — Federated-learning side experiment — NOT STARTED

Planned: partition train into 8–10 "merchants", run Flower + Opacus DP-SGD,
produce (a) federated vs centralized accuracy, (b) accuracy-vs-ε curve.
Secondary — does not touch the graded numbers. May need a separate Python 3.12
venv if PyTorch has no 3.14 wheel.
