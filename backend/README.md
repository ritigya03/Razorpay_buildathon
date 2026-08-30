# Backend (Phase 3) — FastAPI service

Serves the dashboard. Two data sources into one event store (SQLite):

- **Replay feed** — the held-out test split (`data/splits/test.parquet`) streamed
  in `TransactionDT` order at `SENTINEL_REPLAY_DAYS_PER_SEC` days/real-second.
  Every row is scored by the real Phase-1 LightGBM model and grouped by the real
  ring engine; ground-truth labels are known, so lead-time / "flagged before the
  dispute" claims are backed by truth.
- **Razorpay test-mode** — real orders + payment/dispute webhooks. A live payment
  carries only a handful of fields, so it gets a transparent **rules** score
  (`app/rules.py`), clearly labelled `scorer="rules"` vs the replay's
  `scorer="model"`.

The **money loop**: `POST /api/simulate/dispute` raises a dispute against a
flagged fraudulent replay transaction that sits in a flagged ring, and the
response reports how many hours earlier Sentinel had flagged it.

## Run

```bash
# from repo root, with the venv and models already built (make venv && make baseline)
.venv/bin/uvicorn backend.app.main:app --reload --port 8000
# open http://localhost:8000/docs
```

Optional Razorpay: `cp backend/.env.example backend/.env` and fill in test keys.
Without them the replay feed still runs; only `/api/orders`, `/api/verify` and
webhook-signature verification need keys.

For local webhook testing: `ngrok http 8000`, then add
`https://<id>.ngrok.io/webhook/razorpay` in the Razorpay dashboard and subscribe
to `payment.captured`, `payment.dispute.created/won/lost/closed`.

## API

| Method | Path | Purpose |
|---|---|---|
| GET | `/api/health` | liveness, model + razorpay status |
| GET | `/api/stats` | counts, replay progress, live precision, model metrics |
| GET | `/api/transactions` | `?limit&flagged&source&ring_id` |
| GET | `/api/rings` | `?flagged&kind` |
| GET | `/api/rings/{id}` | ring + its member transactions |
| GET | `/api/disputes` | all disputes with lead-time |
| GET | `/api/alerts` | `?limit&kind=ring|txn` |
| POST | `/api/replay/{start\|pause\|reset}` | replay control |
| POST | `/api/orders` | create a test-mode Razorpay order (503 without keys) |
| POST | `/api/verify` | verify a payment signature, ingest the payment |
| POST | `/webhook/razorpay` | Razorpay webhook receiver (signature-checked) |
| POST | `/api/simulate/dispute` | `{}` = auto-pick, or `{"txn_id": "..."}` |

## Tests

```bash
.venv/bin/python -m pytest backend/tests/ -q
```

Boots the app, runs the replay for a few seconds, and asserts the pipeline
(scoring → rings → alerts → simulated dispute with a lead time) end to end.

## Layout

```
app/config.py         settings (env / backend/.env)
app/models.py         Transaction, Ring, Dispute, Alert
app/db.py             engine + session
app/scoring.py        LightGBM model + feature spec (reuses train/common.py)
app/rings.py          runtime ring engine (device / address rings, no union-find)
app/replay.py         time-compressed replay of the held-out split
app/rules.py          rules score for live Razorpay payments
app/razorpay_client.py order create + signature verification
app/events.py         ingest payments / disputes, dispute simulation
app/main.py           FastAPI app + routes
```
