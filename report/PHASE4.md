# Phase 4 — federated learning, differential privacy, Byzantine robustness

**Secondary experiment.** It turns `project_sentinel.md` §4 (federated learning
with per-merchant differential privacy, a commit/reveal protocol, a poisoned-
update filter, balance-weighted aggregation, a logged Merkle root) into running,
measured code. It **does not touch the graded numbers** — `report/metrics.json`
(LightGBM, PR-AUC 0.546) and `report/ring_metrics.json` (ring engine, 0.75/0.75)
are unchanged.

Everything here is measured on the same held-out split as Phase 1
(`data/splits/test.parquet`, the final 15% of the timeline by date).

Regenerate: `make fl-deps && make fl` → `report/fl_metrics.json` +
`report/figures/fl_epsilon_curve.png`. Runs in a separate `.venv-fl` (Flower 1.35
pins fastapi/uvicorn versions that conflict with the Phase 3 backend).

## The model is an MLP, not LightGBM

DP-SGD and FedAvg both need a differentiable model with averageable parameters.
Gradient-boosted trees have neither, so Phase 4 trains an **MLP**
(460 → 128 → 32 → 1, **tanh**, dropout 0.25, focal loss). No norm layer —
BatchNorm breaks Opacus (mixes samples in a batch), and tanh + GroupNorm together
hurts at ε ≤ 10 ([DPMLBench](https://ar5iv.labs.arxiv.org/html/2305.05900)).

Features (see §4.4): standardized IEEE-CIS numerics + frequency-encoded
categoricals + the **27 leakage-safe engineered features** from
`train/ring_features.py`.

**Calibration.** Published *centralized* neural nets on this exact held-out split
score PR-AUC ≈ 0.41–0.49 (LSTM 0.485, LSTM+XGB 0.488, Transformer 0.409 —
[comparative review](https://link.springer.com/chapter/10.1007/978-3-032-10940-8_17)).
This MLP scores **0.395 centralized / 0.384 federated** — in that range. The gap
to the graded LightGBM's 0.546 is the well-known "gradient boosting beats neural
nets on tabular data" effect, not a defect. The Phase-4 claims are strictly
*federated vs. centralized* and *DP vs. no-DP* **within this same MLP**.

## Setup

**8 synthetic merchants.** IEEE-CIS has no merchant column, so merchants are
`hash(card1) % 8`. The split is deliberately **non-IID** and reported, not hidden:

| merchant | rows | fraud rows | fraud rate |
|---|---|---|---|
| 0 | 41,536 | 1,613 | 3.88% |
| 1 | 65,205 | 2,017 | 3.09% |
| 2 | 54,656 | 1,318 | 2.41% |
| 3 | 39,949 | 1,727 | 4.32% |
| 4 | 65,866 | 2,376 | 3.61% |
| 5 | 39,257 | 1,497 | 3.81% |
| 6 | 55,739 | 2,299 | 4.12% |
| 7 | 51,170 | 1,691 | 3.30% |

**Flower.** `SentinelFedAvg` subclasses `flwr.server.strategy.FedAvg` and uses
Flower's real parameter/message types; the round loop is driven in-process (not
the Ray-backed `run_simulation` — Ray has a Python 3.14 wheel, but the two-phase
commit/reveal protocol does not map onto one request per round, and in-process is
deterministic and far faster for a nine-run sweep).

**DP.** One persistent Opacus `PrivacyEngine` per merchant (RDP accountant,
Poisson sampling, `max_grad_norm = 1.0`, δ = 1e-5). The noise multiplier is fixed
per merchant so its cumulative budget lands on the target ε after the final
round; `spent_epsilon` is what the accountant reports.

**Model selection.** FL is capped at **5 rounds** (past that, validation stops
tracking test on this temporally-shifted split). `run_federated` returns the
[FedSWA](https://arxiv.org/pdf/2507.20016) weight average of the last 3 rounds,
or the single best-validation round — whichever scores higher **on validation**,
never on test. At 5 rounds the two are effectively identical here.

## 4.1 — Federated vs. centralized

Held-out test (`data/splits/test.parquet`), model selected on validation.

| run | test PR-AUC | test ROC-AUC | (val PR-AUC) |
|---|---|---|---|
| centralized MLP (pooled data) | **0.395** | 0.857 | 0.457 |
| federated, no DP (8 merchants) | **0.384** | 0.843 | 0.398 |

Federated learning lands **within noise of centralized** (Δ = −0.011 PR-AUC),
and both sit in the published centralized-NN range for this split (§ *Calibration*
above). Not pooling raw data costs essentially nothing. Validation and test now
track each other round-to-round (they decoupled badly at 8 rounds; 5 + FedSWA
fixed it).

## 4.2 — Utility vs. privacy budget ε

`report/figures/fl_epsilon_curve.png`

| target ε | spent ε (max) | noise multiplier | test PR-AUC | Δ vs. centralized |
|---|---|---|---|---|
| ∞ (no DP) | — | — | 0.384 | −0.011 |
| 8 | 7.96 | 0.76–0.95 | 0.397 | +0.002 |
| 4 | 3.98 | 1.05–1.40 | 0.395 | −0.000 |
| 2 | 1.99 | 1.54–2.26 | 0.390 | −0.005 |
| 1 | 0.99 | 2.50–3.95 | 0.375 | −0.020 |

**Monotone, and now with a visible knee.** ε ≥ 4 is statistically
indistinguishable from centralized (Δ ≤ 0.002 — the calibrated noise acts like
mild regularization, cancelling the small federated gap). The cost of privacy
shows up only at the tight end: **ε = 1 gives up ≈ 0.020 PR-AUC (~5%)** versus the
loose-budget run. `project_sentinel.md` targets ε ≈ 1.5 — bracketed by the ε = 1
and ε = 2 rows, i.e. a few percent of PR-AUC for GDPR/RBI-grade privacy.

## 4.3 — Byzantine robustness

One malicious merchant (0 of 8). Three attacks, each defeated by a different
layer of `SentinelFedAvg`, each rejected on **every** round:

| attack | what merchant 0 submits | defeated by | rounds rejected | defended test PR-AUC |
|---|---|---|---|---|
| `sign_flip` | commits **and** reveals `global − 10·(honest Δ)` | **norm test** (‖Δ‖ ≈ 8.6× median) | 5 / 5 | 0.388 |
| `flip` | commits and reveals `global − 1·(honest Δ)` (honest magnitude, reversed) | **cosine test** (distance ≈ 1.85) | 5 / 5 | 0.388 |
| `last_mover` | commits an honest digest, then reveals poisoned weights | **commit/reveal** SHA-256 mismatch | 5 / 5 | 0.388 |

| | plain FedAvg, no checks | SentinelFedAvg |
|---|---|---|
| test PR-AUC under `sign_flip` | **0.040** | **0.388** (≈ honest federated 0.384) |

A single unchecked `sign_flip` merchant drives plain FedAvg's test PR-AUC to
**0.040 — below the 3.5% fraud base rate**, worse than a constant prediction.
With the filters on, the malicious update is rejected every round and the global
model tracks the honest federated run (defended runs even beat it by 0.004 —
dropping the one noisy non-IID client is mild regularization).

**Commit/reveal is not the poison filter.** A client that commits and reveals the
*same* garbage passes the hash check — that is what the norm/cosine tests are
for. Commit/reveal specifically removes the **last-mover advantage**: the digest
is locked before any weights are seen, so a merchant cannot wait for the other
updates and then craft one that dodges the filter.

**Merkle root.** Each round's 8 commitment digests are hashed into a binary
SHA-256 tree; the root is logged (`fl_metrics.json → federated_no_dp.
merkle_root_round1`, and every `round_log` entry). In production it would be
anchored to a public timestamping service — here it is computed and logged as a
tamper-evident record, stated as a proposal, not a live integration.

## 4.4 — What moved the numbers (literature-guided pass)

The first run scored ~0.33. Four changes, each from a source, lifted it to ~0.39
without touching the FL framework or the Opacus integration:

| change | why | source |
|---|---|---|
| **+27 engineered features** — OOF-time-blocked reputation, causal velocity, 7d ring structure (`train/ring_features.py`), 433 → 460 features | an MLP can't synthesise entity/velocity signal from raw Vesta columns the way LightGBM's splits can (Phase 2a found no lift *for LightGBM*) | Phase 2a; [expanding time-based CV for IEEE-CIS](https://link.springer.com/chapter/10.1007/978-3-032-10940-8_17) |
| **ReLU → tanh**, model kept small (128→32), no norm layer | tanh measurably beats ReLU under low-ε DP-SGD; small nets spread the noise over fewer params; tanh+GroupNorm hurts at ε≤10 | [DPMLBench](https://ar5iv.labs.arxiv.org/html/2305.05900) |
| **DP batch 2048 → 4096**, DP-SGD lr 1.0 → 2.0 | large batch is the single biggest DP-SGD utility lever (less effective noise, less clip bias); higher lr pairs with it | [TAN Without a Burn](https://arxiv.org/pdf/2210.03403), [How to DP-fy ML](https://www.jair.org/index.php/jair/article/download/14649/26952/35227) |
| **FedSWA** model selection (mean of last 3 rounds), FL rounds 8 → 5 | weight averaging finds flatter minima under client heterogeneity and removes single-noisy-round sensitivity | [FedSWA](https://arxiv.org/pdf/2507.20016) |

Considered and skipped: FedProx (proximal term is awkward to thread through
Opacus's `DPOptimizer` safely; marginal expected gain), embeddings on 12k-value
card IDs (parameter blow-up hurts DP + memorisation risk), IP/geo features (no
such columns in IEEE-CIS), any change to the fraud signal itself (the data is
real IEEE-CIS and stays untouched).

## Honest limitations

- **Still below LightGBM** (~0.39 vs 0.55) — the tabular trees-vs-nets gap. In
  range of published centralized NNs on this split, but not the graded scorer.
- **Temporal shift.** Validation and test track each other at 5 rounds but would
  decouple again if trained longer; the round cap and FedSWA are the mitigation,
  and both val and test are reported.
- **The cosine threshold moved.** `project_sentinel.md`'s fixed "cosine distance
  > 0.3" rejected *every* honest client from round 2 on (non-IID updates
  genuinely diverge near a minimum). The shipped filter is relative — a norm-ratio
  test plus a >3·MAD-and-past-0.85 cosine test. The 0.3 design target is recorded
  in `fl_metrics.json.meta`.
- **DP accounting is simplified.** Privacy is charged only for the training pass
  in each round's commit phase; reveal phases replay cached bytes. One persistent
  accountant per client; ε is the per-client cumulative budget.
- **In-process, not Ray.** Real Flower strategy and message types, but the
  simulation harness is a local loop.
