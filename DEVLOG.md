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

**Held-out test result** (current `report/ring_metrics.json`; the earlier draft
of this table quoted 0.75/0.75 and 18,001→132 from a pre-Phase-3 baseline run —
the Phase-3 refactor re-ran `make baseline` with `first_metric_only=True`,
best_iter 609→1144, which shifted these slightly. The JSON is authoritative):

| ring type | groups | fraud rings | flagged | precision | recall | F1 |
|---|---|---|---|---|---|---|
| device | 433 | 92 | 104 | **0.72** | **0.82** | **0.77** |
| address | 586 | 42 | 40 | 0.42 | 0.40 | 0.41 |

- Alert-volume reduction: 17,835 transaction alerts → **144 ring alerts**
  (≈ 124× fewer to review).
- Fraud coverage: flagged rings hold 539 / 3,083 fraud txns (**17.5%** of all
  fraud) — the coordinated slice. Only ~24% of transactions carry device data;
  most fraud is single-account. Ceiling stated openly.
- A few frauds are caught only because their ring was flagged (own score below
  the per-txn threshold).

**Reading:** transaction model handles fraud broadly (PR-AUC 0.55, recall 0.84
at the cost-optimal point); the ring engine takes the coordinated slice and
makes it triage-able at 0.72 / 0.82 with ~120× less review volume. Complementary;
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

### Phase 6b — visual pass (2026-08-30, user request)

Retheme to the Razorpay palette (navy `#02042B`, Prussian `#0C2651`, blue
`#0D94FB`/`#3395FF`, cyan `#4DE1F2`, blue→cyan gradient), Space Grotesk + Inter
via Google Fonts. Added:
- **Page loader** (`components/Loader.tsx`) — three counter-rotating rings +
  gradient bead + wordmark + a progress bar driven by the boot `Promise.allSettled`
  of health/report/stats; fades out via CSS.
- **Landing page** (`components/Landing.tsx`) — animated node-network canvas
  (`components/NetworkCanvas.tsx`, rAF, ~90 nodes, distance-based edges, honours
  reduced-motion), floating blurred gradient orbs, panning masked grid, count-up
  metric strip (`hooks.ts` `useCountUp`), scroll-reveal sections (`useReveal` +
  IntersectionObserver), "we don't replace Vulcan, we unlock it" comparison.
  Metric numbers are pulled live from `/api/report` with static fallbacks.
- Hash routing in `App.tsx` (`#/app`), wordmark click → back to landing.
- `prefers-reduced-motion` disables all animation + the canvas rAF loop.

Bundle 556 kB (159 kB gzip) — recharts dominates; fine for a local demo.
`tsc` + `vite build` clean.

---

## Phase 4 — Federated learning + DP + Byzantine robustness (2026-08-31)

Secondary experiment. Turns `project_sentinel.md` §4 (FL + per-client DP +
commit/reveal + poison filter + Merkle root) into running code. **Does not touch
`report/metrics.json` or `report/ring_metrics.json`.** Files: `train/fl_{data,
model,strategy,client,experiment}.py`, `report/fl_metrics.json`,
`report/figures/fl_epsilon_curve.png`, `report/PHASE4.md`, `requirements-fl.txt`,
`make fl-deps` / `make fl`, backend `GET /api/fl-report`, frontend Federated tab.

### Environment — separate venv after all

`torch==2.13.0` (cp314 wheel), `opacus==1.6.0`, `flwr==1.35.0` **all resolve on
Python 3.14** — no separate 3.12 interpreter needed, contrary to the Phase 0
note. But `flwr 1.35` pins `fastapi<0.139` / `uvicorn[standard]<0.50`, which
collide with the Phase 3 backend's `fastapi==0.141.1`. Installing the FL stack
into `.venv` silently downgraded fastapi/starlette/uvicorn and would have broken
the backend. Fix: a dedicated **`.venv-fl`** (`requirements-fl.txt`, no
fastapi/uvicorn), driven by `make fl-deps`. `.venv` restored to
`requirements.txt`. Backend test `test_replay_pipeline_and_dispute_loop` still
passes; `test_orders_endpoint_503_without_keys` fails **only because `backend/.env`
now has real test keys** — pre-existing, unrelated to Phase 4.

### Model — MLP, not LightGBM

DP-SGD and FedAvg both need a differentiable model with averageable parameters;
gradient-boosted trees have neither. Phase 4 trains an **MLP** (433→128→32→1,
ReLU, dropout, focal loss γ=2). No BatchNorm — Opacus forbids it. Categoricals
are **frequency-encoded** (mapping each value to its train frequency), *not*
target-encoded — Phase 2a already documents target encoding collapsing a model
on this exact data. Consequence: the MLP is much weaker than the graded LightGBM
(PR-AUC ≈ 0.27–0.33 vs 0.55). Stated openly everywhere; the Phase-4 claims are
*federated vs centralized* and *DP vs no-DP* **within this MLP**.

### Merchant partition

No merchant column in IEEE-CIS. 8 synthetic merchants by `hash(card1) % 8` —
**non-IID**: 39k–66k rows each, fraud rate 2.4%–4.3%. Reported per-merchant in
`fl_metrics.json` rather than hidden.

### Flower integration — in-process, not Ray

Uses Flower's real `FedAvg` base class and `flwr.common` parameter/message types;
`SentinelFedAvg(FedAvg)` overrides the aggregation. The round loop is driven
**in-process** by `fl_experiment.py`, not via `flwr.simulation.run_simulation`.
Ray 2.58 *does* have a cp314 wheel (checked), but (a) the two-phase
commit/reveal protocol doesn't map onto Flower's one-request-per-round
simulation model, and (b) in-process is deterministic and ~10x faster for a
sweep that re-runs FL nine times. Same spirit as the Phase-0 Redis/Sepolia trims.

Commit/reveal = **two Flower rounds per FL round**: clients cache their update,
send `sha256(weights)` (commit), then send the weights (reveal); a reveal whose
digest ≠ its commitment is rejected (last-mover defense). Merkle root = binary
SHA-256 tree over the round's 8 commitments, logged each round.

### Dead end 1 — the doc's fixed cosine threshold nukes every honest client

`project_sentinel.md` §4.1: "reject cosine distance > 0.3 from the median
update". Implemented literally → **rounds 2+ rejected all 8 honest clients**
(`agg=0/8`), FL froze. Cause: under the non-IID split, once the model is near a
minimum the honest per-client deltas genuinely diverge — pairwise cosine
distance 0.3–0.9 is normal, not Byzantine. A fixed absolute threshold can't tell
"noisy consensus" from "attack".

Fix — a **relative** two-stage filter in `SentinelFedAvg.aggregate_reveal`:
- **norm test**: reject delta with L2 norm > 3× the median delta norm (catches
  scaled model-negation).
- **cosine test**: among norm survivors, reject a delta whose cosine distance
  from the median is *both* a >3·MAD outlier *and* past an 0.85 floor (catches
  direction reversal at honest magnitude). On honest-only rounds nothing fires.

The doc's 0.3 is kept in `fl_metrics.json.meta` as the original design target,
with this deviation noted.

### Dead end 2 — DP-SGD collapse

First DP run: Adam optimizer, batch 1024, ε=1.5 → test PR-AUC **0.02** (below
the 3.5% base rate — worse than constant). Three compounding causes and fixes:
- **Adam + DP**: the per-step Gaussian noise corrupts Adam's second-moment
  estimate. → plain **SGD + momentum 0.9**, lr 1.0 for the DP path (Adam kept
  for centralized + non-DP FL).
- **batch too small**: DP-SGD needs a large batch for signal to survive the
  noise. → **batch 2048**.
- **noise over too many params**: → shrank the MLP from 256→64 to **128→32**.
After: ε=4 gives a usable PR-AUC (see table). The tight end (ε≈1) still degrades
hard — that's the real utility cost, reported as the deliverable.

### Dead end 3 — FL unstable in late rounds

Non-IID FedAvg peaks around round 4–6 then drifts down on test (val keeps
creeping up — the MLP overfits the val era; test is a later time slice). Fix:
`run_federated` keeps the **best-validation-PR-AUC round's weights** as the
returned model (legitimate model selection — never selects on test).

### Three attacks, three defense layers (`robustness_demo`)

| attack | what merchant 0 sends | caught by |
|---|---|---|
| `sign_flip` | commits & reveals `global − 10·(honest delta)` | **norm test** |
| `flip` | commits & reveals `global − 1·(honest delta)` (honest magnitude) | **cosine test** |
| `last_mover` | commits an honest digest, reveals poisoned weights | **commit/reveal hash** |

Commit/reveal alone does **not** stop `sign_flip`/`flip` — a client that commits
and reveals the *same* garbage passes the hash check; that's what the anomaly
filter is for. Commit/reveal stops the *adaptive* (last-mover) attacker.

### Results (full run: 8 rounds, held-out test; details → `report/PHASE4.md`)

| run | test PR-AUC | notes |
|---|---|---|
| centralized MLP | 0.340 | val 0.435 (overfits the val era; LightGBM baseline is 0.546) |
| federated, no DP | 0.334 | ≈ centralized (Δ −0.006); peaks 0.414 @ round 4, best-val picks r8 |
| federated + DP ε=8 / 4 / 2 / 1 | 0.392 / 0.390 / 0.387 / 0.377 | monotone decline; spent ε ≈ target; all *above* centralized (noise regularizes) |
| poison `sign_flip`, no defense | 0.061 → 0.033 | 1/8 malicious merchant destroys plain FedAvg (below the 3.5% base rate) |
| poison, defended ×3 | 0.366 / 0.365 / 0.366 | `sign_flip`→norm test (8/8 rounds), `flip`→cosine test (7/8), `last_mover`→hash mismatch (8/8); all recover to ≈ honest FL |

All **15** sanity assertions pass. `make fl` ≈ 4–5 min wall; `--quick` ≈ 90 s.

Honest note on the ε curve: it's monotone (the right direction) but shallow —
at this step budget DP down to ε ≈ 1 costs only ~4% of PR-AUC, and every
federated variant sits inside the weak MLP's own run-to-run variance. The strong
results here are *federated ≈ centralized* and the *poison before/after*, not a
dramatic privacy/utility cliff.

### Improvement pass (2026-09-01) — literature-guided, no FL/Opacus API changes

Web research first (citations in `report/PHASE4.md`). Key calibration: published
*centralized* NNs on this exact held-out split score PR-AUC ~0.41–0.49 (LSTM
0.485, Transformer 0.409) — the gap to LightGBM's 0.546 is the known
trees-beat-nets-on-tabular effect, not a bug. Realistic MLP target ≈ 0.42–0.48.

Four changes, each backed by a source:
1. **Engineered features** — `fl_data.py` now appends the 27 leakage-safe
   ring/entity features from `train/ring_features.py` (OOF-time-blocked
   reputation, causal velocity, 7d union-find ring structure). 433 → **460**
   features. Phase 2a found no lift for *LightGBM* (redundant with Vesta
   columns), but an MLP can't synthesise them from raw columns the way GBM
   splits can.
2. **tanh, not ReLU** (DPMLBench: measurably better at low ε). Model stays
   128→32, **no norm layer** — BatchNorm breaks Opacus, and tanh+GroupNorm hurts
   at ε ≤ 10.
3. **DP batch 2048 → 4096**, DP-SGD lr 1.0 → 2.0 (TAN scaling laws: large batch
   is the biggest DP utility lever; higher lr pairs with it). 8192 was tried and
   dropped — 4× the wall time for a marginal gain.
4. **FedSWA model selection** — return the mean of the last 3 rounds' global
   weights (or best-val round, whichever wins *on validation*). Flatter minima
   under heterogeneity; removes the single-noisy-round sensitivity. FL rounds cut
   8 → 5 (val and test still track each other there).

Skipped: FedProx (proximal term is awkward to thread through Opacus's DPOptimizer
safely; marginal expected gain), embeddings on 12k-card IDs (param blow-up hurts
DP + memorisation risk), any change to the synthetic-signal — the data is real
IEEE-CIS and stays untouched.

Results (full run, 5 rounds, held-out test PR-AUC):

| run | before | after |
|---|---|---|
| centralized MLP | 0.340 | **0.395** (val 0.457, ROC-AUC 0.857) |
| federated, no DP | 0.334 | **0.384** (Δ −0.011 vs centralized) |
| federated + DP ε=8 / 4 / 2 / 1 | .392 / .390 / .387 / .377 | **.397 / .395 / .390 / .375** |
| poison, no defense (`sign_flip`) | 0.061 | **0.040** |
| poison, defended ×3 | ~0.366 | **0.388** (all 5/5 rounds rejected; ≈ honest FL) |

Both non-DP numbers up ~+0.05 into the published centralized-NN range for this
split. The ε curve is now monotone *with a visible knee*: ε≥4 is indistinguishable
from centralized (Δ≤0.002 — noise regularises the small federated gap away), ε=1
costs 0.020 PR-AUC (~5%). All 15 sanity assertions pass. `make fl` ≈ 6.5 min;
`--quick` (3 rounds) ≈ 3 min. Notes: DP batch 8192 quadrupled wall time for a
marginal gain — settled on 4096. SWA never beat best-val selection at 5 rounds
(no drift to average out) but stays as a guard-rail and is reported.

---

## Phase 5 — risk-analyst agent (2026-09-01)

Turns `project_sentinel.md` §4.3 (an agent that explains a ring and drafts
mitigation) into running code. Lives **inside the Phase-3 backend**
(`backend/app/agent.py`, routes `POST /api/agent/chat`, `GET /api/agent/health`,
`POST /api/agent/reset`), surfaced as the dashboard's **Agent** tab
(`frontend/src/components/Agent.tsx`).

### Provider — Gemini, not Claude (cost, not preference)

Phase 0 decided "Claude Agent SDK (same as Razorpay Agent Studio)". User has no
paid API access and required a **free** key, so the shipped agent runs on
**Google Gemini free tier** (`google-genai` 2.21.0, installs clean on 3.14 — all
heavy deps already vendored). Same architecture — a supervised tool-calling loop
with a system brief — different model. `project_sentinel.md` §4.3/§6.1 actually
named "Gemini via Vertex AI" for the report generator originally, so this is
consistent with the concept doc; the deviation from the Phase-0 note is recorded
here.

### Model — `gemini-3.1-flash-lite`

- `gemini-2.5-flash` → **404 "no longer available to new users"** on a fresh key
  (the API now points new keys at 3.x). `gemini-3.6-flash` works but its free
  tier is **5 requests/min**, and one agent turn with AFC = 2–4 requests, so two
  questions exhaust it.
- `gemini-3.1-flash-lite` cleared a 9-call burst with no 429 — far more free-tier
  headroom. Made it the default (`SENTINEL_GEMINI_MODEL`, overridable).
- 429s are still possible under rapid fire; caught and returned as HTTP 429 with
  a "wait ~30s" message the Agent tab renders inline (not a red error).

### Tools — read-only, four of them

`get_situation_summary` (no args), `list_flagged_rings(limit)`,
`get_ring_detail(ring)`, `get_recent_disputes(limit)`. All query the same SQLite
event store the dashboard reads; **no write tools** — the agent cannot block,
dispute, or mutate anything. Each is wrapped by a decorator that records the call
(surfaced to the UI as tool chips) and converts any exception into a clean
`{"error": …}` result rather than a raw traceback.

### Dead end 1 — `from __future__ import annotations` breaks Gemini AFC

`get_situation_summary` and `list_flagged_rings` worked; `get_ring_detail` (the
only tool with a **required** arg) always failed inside automatic function
calling with `isinstance() arg 2 must be a type, a tuple of types, or a union`.
Cause: the Gemini SDK coerces incoming tool args against the function's real
annotation objects; under `from __future__ import annotations` those are lazy
strings (`'str'`), so `isinstance(value, 'str')` throws. The model then
confabulated a "technical issue retrieving ring details". Fix: drop the future
import in `agent.py` only (all its annotations — `str | None`, `dict[str, …]` —
are runtime-safe on 3.14 anyway). `functools.wraps` on the decorator then carries
the real types through.

### Dead end 2 — ring ids are not stable

`recompute_rings` does a full DELETE + reinsert of the `Ring` table every replay
tick, so `ring.id` climbs continuously and "explain ring #42" breaks seconds
later when #42 has been renumbered. Fixed agent-side: `list_flagged_rings`
returns a `rank` (1 = highest risk) and a stable `ring_key` (`kind|key`);
`get_ring_detail` accepts either and never a bare id. System brief tells the
model ids are unstable and to re-list if unsure.

### Verified

`make test` — the pipeline test now also calls the three read tools directly
(no model call, no quota spent) and asserts shape; a new
`test_agent_health_and_input_validation` checks `/api/agent/health` and the
empty-message 400. Live end-to-end (TestClient + real Gemini key): turn 1
`get_situation_summary` + `list_flagged_rings` → a grounded briefing with a real
ring table; turn 2 `get_ring_detail("1")` → a forensic report (shared
fingerprint, distinct accounts/cards, window, all-fraud) with the honest
coordinated-slice caveat, plus a drafted escalation note. `npm run build` clean
(bundle 567 kB / 163 kB gzip — recharts still dominates; the ~50-line inline
markdown renderer in `Agent.tsx` avoids a new dep). Pre-existing
`test_orders_endpoint_503_without_keys` still fails for the Phase-4 reason (real
Razorpay keys in `backend/.env`), untouched by Phase 5.

---

## Phase 7 — live Razorpay detection path (2026-09-01)

**Problem raised by the user:** with the IEEE replay as the backbone, the
Razorpay tab was near-decoration — real webhook + a rules score on one payment,
but the model never scored it and the ring engine never saw it (`ingest_payment`
wrote `card_id=None, device_id=None, uid=None`). So "pay through Razorpay from
multiple ids and detect fraud" didn't actually work.

**Scope decision (locked with the user):** IEEE replay stays byte-for-byte
unchanged (it's the graded path). The Razorpay path becomes a *second, lighter*
detector on real payment fields. No device-fingerprint-via-notes — card + carding
rings only, everything from real webhook fields.

### What changed

- **`ingest_payment`** now derives real identity from the payment entity:
  `card_id = network|issuer|type|last4` (a real, non-unique card grouping key —
  same spirit as IEEE's `_card_id` BIN tuple), account `uid = contact or email`,
  and a `merchant` tag from order `notes.sentinel_merchant`. New
  `Transaction.merchant` column; `Ring` gains `source` and `n_merchants`.
- **`rules.py`** gains the shared-entity velocity signals that make a
  coordinated pattern cross the flag line: "this card seen on N distinct
  identities in the last hour" (+0.25…), "…at N merchants" (+0.15,
  cross-merchant), "this identity used N distinct cards" (+0.25…, carding).
  Isolated payments stay at base ~0.05.
- **`recompute_rings`** adds a **`card` ring rule scoped to `source="razorpay"`**
  (one card → ≥2 customer identities). Scoped to live rows so the replay's
  BIN-tuple `card_id` — shared by thousands — can't recreate the Phase-2b
  mega-blob. Replay device/address rings unchanged. Carding surfaces as an
  `address` ring on live rows (same `rzp|dom|contact`, ≥2 distinct cards).
- **`simulate_dispute_auto`** falls back to a flagged live payment in a flagged
  ring, so "dispute the one I just made" closes the loop (lead time is real but
  small — the payment came in seconds ago).
- **`POST /api/demo/scenario`** (`kind: shared_card | carding`) seeds a
  coordinated set through the *same* `ingest_payment` path — for tests and as a
  stage fallback when N manual Checkout runs aren't practical. The real flow is
  the Razorpay tab's Checkout.
- **Frontend Razorpay tab** rebuilt: merchant selector (A/B/C), 3 customer
  identity presets that prefill Checkout, `notes.sentinel_merchant` on the order,
  a "rings from live payments" panel (card/address rings with an `N merchants`
  badge), and the two seed buttons. Rings tab shows a `live` / `N merchants`
  badge on `source="razorpay"` rings.
- **Agent**: `_ring_row` carries `source` + `merchants_spanned`; system brief
  tells it to call out cross-merchant live rings explicitly and to note the live
  path uses the rules model, not the graded LightGBM.

### Verified

`make test` green (4/4). Replaced the always-failing
`test_orders_endpoint_503_without_keys` with `test_orders_endpoint_requires_keys`
(asserts the behaviour that holds given whatever's in `backend/.env`). New
`test_live_razorpay_ring_and_dispute`: `shared_card` scenario → flagged `card`
ring, 4 identities / 1 card / ≥2 merchants; `carding` → flagged `address` ring
with ≥3 cards; dispute closes on a flagged live payment. Playwright: seeded both
scenarios, Razorpay tab shows `#N · card [3 merchants] flagged risk 0.62` over
`Visa|HDFC|credit|last4:1111` and `#N · address flagged` over
`rzp|gmail.com|9000000001`; zero console errors; `npm run build` clean
(571 kB / 164 kB gzip).

### Honest note for the pitch

The live path is a **rules scorer** (a handful of features), not the graded
LightGBM (~430). Graded accuracy (PR-AUC 0.546, ring 0.72–0.82) is the replay,
not the live payments. The Razorpay demo proves the pipeline runs end-to-end on
real payments and shows the *cross-merchant* pattern — it is a lighter detector
than the graded one, stated plainly.

---

## Phase 8 — federated cross-merchant ring detection (2026-09-01)

**Why:** the pitch (`project_sentinel.md`) is *"cross-merchant fraud-ring
intelligence through Federated Learning and Differential Privacy."* Through
Phase 7 the ring engine was 100 % centralised and Phase 4 FL was a separate
classifier experiment — the build did not match the claim. Phase 8 builds the
thing the pitch describes and **measures it against the centralised version**
(which stays, for the comparison).

### Design (locked with the user)

- Detect cross-merchant rings *without any merchant sharing raw transactions*.
- Keep the centralised ring detector alongside as the "sees everything" oracle.
- **Hold the per-transaction score fixed** (Phase-1 LightGBM) across both arms,
  so the comparison isolates the cost of federating the *ring layer* — the
  score-model FL cost is Phase 4's job and the two compose.
- Report whatever the numbers are.

### What's built

- `train/fl_crypto.py` — dependency-free primitives (imports from core `.venv`
  *and* the backend): SHA-256 commit, Merkle root, salted-HMAC fingerprint,
  Gaussian-mechanism `DPHistogram` (per-fingerprint **risk-bucket count vector**;
  one txn → one bucket → L2 sensitivity 1 → one σ, no budget split), bucketize /
  risk-estimate helpers.
- `train/fl_rings.py` → `report/fl_ring_metrics.json` (`make fl-rings`, core
  `.venv`, seed 42). 8 synthetic merchants (`hash(card1) % 8`). Entity = device
  or address fingerprint (cards can't span merchants on this partition — they
  hash to one merchant; devices/addresses can, and the concept doc names device
  fingerprints anyway). Each merchant releases, per salted-HMAC fingerprint, a
  DP-noised risk-bucket vector; commit/reveal + Merkle root; robust aggregator
  rejects volume / risk-estimate outliers and commitment mismatches; summed
  vectors flagged with the **same rule** as the centralised oracle. ε sweep
  (5 noise draws averaged per ε, threshold re-calibrated on val per ε), one
  malicious merchant (hot-flood + commit-mismatch).
- `backend/app/fl_live.py` + `POST /api/fl/detect-live` — the runtime protocol
  over the live Razorpay payments (entity = card fingerprint, which *does* span
  merchants live). Returns the per-merchant HMAC sketches, the Merkle root, the
  aggregated cross-merchant rings, and a centralised pass alongside. Optional
  `epsilon` adds DP noise.
- `GET /api/fl-ring-report`; `FederatedRings.tsx` — a Phase-8 section stacked
  above the Phase-4 panels in the Federated tab (centralised vs federated bars,
  the DP ε→F1 curve, the poison before/after, and a "run it live" panel that
  shows the merchant sketches → Merkle root → emerged cross-merchant ring).

### Dead end 1 — the first mechanism (count + risk-sum) inverted the DP curve

v1 released `(count, risk_sum)` per fingerprint and the aggregator computed
`mean = risk_sum / count`. Under DP the two are noised independently, so a
count that noised *down* while the sum held inflated the mean → more fingerprints
crossed the threshold → **recall went UP and precision collapsed as ε shrank**
(ε=8 gave P 0.17 / R 0.85). Wrong shape entirely. Fix: release a **risk-bucket
count vector** (each txn lands in exactly one bucket, sensitivity 1) and estimate
mean risk as the bucket-midpoint weighted mean — bounded, and degrades
gracefully. Also: re-calibrate the detector threshold on *validation with the
same noise* for each ε (standard for DP eval; val only).

### Dead end 2 — `hot` threshold too loose

Second cut used a single "hot" count where hot = score ≥ the cost-optimal
operating point (~0.012 — flags ~20 % of *all* transactions). `hot_frac` barely
separated rings from legit shared devices → centralised F1 only 0.41. The
5-bucket vector (above) restored discrimination → F1 0.67.

### Dead end 3 — robust filter rejected an honest merchant

`NORM_RATIO = 3` on per-merchant reported volume rejected honest merchant 3
(3.14× the median — merchant volumes are genuinely non-IID). Raised to 5.0:
catches only gross volume inflation, no honest false rejects; the hot-flood
attack is caught by the *risk-estimate* outlier test anyway (it moves bucket
mass, not volume).

### Results (held-out test, combined device + address; reproducible)

118 cross-merchant fraud rings (83 device + 35 address); 20 single-merchant.

| detector | P | R | F1 |
|---|---|---|---|
| centralised (sees every txn) | 0.68 | 0.66 | 0.67 |
| federated, **no DP** | 0.68 | 0.66 | 0.67 |

**Identical — 78 TP / 36 FP / 40 FN in both. Federating the computation is free.**

DP ε sweep: ∞ → **0.67**, ε=32 → 0.66, ε=16 → 0.43, ε=8 → 0.39, ε=4 → 0.32,
ε=2 → 0.22, ε=1 → 0.08. Free to ε ≈ 32; below ε ≈ 16 the ~1-txn-per-merchant
contributions to small rings fall below the Gaussian noise floor. An honest cost
at *this* data scale — it shrinks as per-merchant volume grows (Vulcan scale
would push the knee far below ε=1). Reported as a cost, not spun.

Poison (1 of 8 malicious): hot-flood no-defense F1 **0.34** → robust aggregator
**0.61** (merchant 0 rejected every run as a risk-estimate outlier);
commit-mismatch caught by the SHA-256 digest check. Merkle root logged.

### Honest limitations (also in `report/PHASE8.md`)

Synthetic merchants; scorer is centralised LightGBM held fixed (a fully federated
stack scores with the Phase-4 MLP — costs compose); DP-noised histogram sharing
is a simplified stand-in for a real PSI / secure-aggregation protocol; silent
(non-participation) poisoning is an incentive problem, not an aggregation one;
device data on only ~24 % of txns caps recall for any fingerprint-based detector.

### Verified

`make fl-rings` reproducible (two runs identical). `make test` 5/5 (added
`test_federated_live_detection` — sketches carry no raw card/email, a
cross-merchant ring is flagged, DP variant well-formed). `npm run build` clean
(579 kB / 166 kB gzip). Playwright: Federated tab renders the Phase-8 panels
above the Phase-4 ones; "run federated detection" over a seeded shared-card
scenario shows 3 merchant sketches → Merkle root → one flagged cross-merchant
ring (`ddee6c51e8f2…`, 5 payments, 3 merchants, risk 0.62); zero console errors.
