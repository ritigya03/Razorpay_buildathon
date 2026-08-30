"""Project Sentinel backend — FastAPI service.

Serves the dashboard: a time-compressed replay of the held-out split scored by
the real model + ring engine, plus live Razorpay test-mode payments/disputes.
"""
from __future__ import annotations

import json
from contextlib import asynccontextmanager

from fastapi import Body, Depends, FastAPI, Header, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import func
from sqlmodel import Session, select

from .config import settings
from .db import engine, get_session, init_db
from .events import (
    ingest_dispute, ingest_payment, simulate_dispute_auto, update_dispute_status,
)
from .models import Alert, Dispute, Ring, Transaction
from .razorpay_client import create_order, verify_payment_signature, verify_webhook_signature
from .replay import replay
from .scoring import scorer


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    if settings.replay_autostart and scorer.ok:
        replay.load()
        replay.reset()   # each service start = a clean replay from day 0
        replay.start()
    yield
    replay.pause()


app = FastAPI(title="Project Sentinel", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
)


# --------------------------------------------------------------------------- #
# status / reads
# --------------------------------------------------------------------------- #
@app.get("/api/health")
def health() -> dict:
    return {
        "ok": True, "model_loaded": scorer.ok,
        "razorpay_ready": settings.razorpay_ready,
        "razorpay_key_id": settings.razorpay_key_id or None,  # public key, safe to expose
    }


@app.get("/api/stats")
def stats(s: Session = Depends(get_session)) -> dict:
    def count(model, *where):
        q = select(func.count()).select_from(model)
        for w in where:
            q = q.where(w)
        return s.exec(q).one()

    n_txn = count(Transaction)
    n_fraud = count(Transaction, Transaction.is_fraud == 1)
    n_flagged = count(Transaction, Transaction.flagged == True)  # noqa: E712
    tp = count(Transaction, Transaction.flagged == True, Transaction.is_fraud == 1)  # noqa: E712
    disputes = s.exec(select(Dispute)).all()
    metrics = {}
    if settings.metrics_file.exists():
        m = json.loads(settings.metrics_file.read_text())
        metrics = {"pr_auc_test": m["pr_auc"]["test"], "roc_auc_test": m["roc_auc"]["test"],
                   "operating_point": m["operating_point"]["test"]}
    return {
        "replay": replay.status(),
        "transactions": n_txn,
        "fraud_transactions": n_fraud,
        "flagged": n_flagged,
        "flagged_true_positive": tp,
        "live_precision_so_far": (tp / n_flagged) if n_flagged else None,
        "rings_total": count(Ring),
        "rings_flagged": count(Ring, Ring.flagged == True),  # noqa: E712
        "disputes": len(disputes),
        "disputes_flagged_in_advance": sum(1 for d in disputes if d.was_flagged),
        "avg_lead_time_hours": (
            sum(d.lead_time_hours or 0 for d in disputes if d.was_flagged)
            / max(sum(1 for d in disputes if d.was_flagged), 1)
        ),
        "model_metrics": metrics,
    }


@app.get("/api/transactions")
def list_transactions(
    limit: int = 100, flagged: bool | None = None, source: str | None = None,
    ring_id: int | None = None, s: Session = Depends(get_session),
) -> list[Transaction]:
    q = select(Transaction).order_by(Transaction.ts.desc()).limit(min(limit, 1000))
    if flagged is not None:
        q = q.where(Transaction.flagged == flagged)
    if source:
        q = q.where(Transaction.source == source)
    if ring_id is not None:
        q = q.where(Transaction.ring_id == ring_id)
    return s.exec(q).all()


@app.get("/api/rings")
def list_rings(
    flagged: bool | None = None, kind: str | None = None,
    limit: int = 100, s: Session = Depends(get_session),
) -> list[Ring]:
    q = select(Ring).order_by(Ring.score_mean.desc()).limit(min(limit, 500))
    if flagged is not None:
        q = q.where(Ring.flagged == flagged)
    if kind:
        q = q.where(Ring.kind == kind)
    return s.exec(q).all()


@app.get("/api/rings/{ring_id}")
def get_ring(ring_id: int, s: Session = Depends(get_session)) -> dict:
    ring = s.get(Ring, ring_id)
    if not ring:
        raise HTTPException(404, "ring not found")
    members = s.exec(select(Transaction).where(Transaction.ring_id == ring_id)
                     .order_by(Transaction.ts)).all()
    return {"ring": ring, "members": members}


@app.get("/api/disputes")
def list_disputes(s: Session = Depends(get_session)) -> list[Dispute]:
    return s.exec(select(Dispute).order_by(Dispute.created_at.desc())).all()


@app.get("/api/report")
def report() -> dict:
    out: dict = {}
    if settings.metrics_file.exists():
        out["metrics"] = json.loads(settings.metrics_file.read_text())
    rm = settings.metrics_file.parent / "ring_metrics.json"
    if rm.exists():
        out["ring_metrics"] = json.loads(rm.read_text())
    return out


@app.get("/api/alerts")
def list_alerts(
    limit: int = 50, kind: str | None = None, s: Session = Depends(get_session),
) -> list[Alert]:
    q = select(Alert).order_by(Alert.ts.desc()).limit(min(limit, 500))
    if kind:
        q = q.where(Alert.kind == kind)
    return s.exec(q).all()


# --------------------------------------------------------------------------- #
# replay control
# --------------------------------------------------------------------------- #
@app.post("/api/replay/{action}")
async def replay_control(action: str) -> dict:
    if action == "start":
        replay.start()
    elif action == "pause":
        replay.pause()
    elif action == "reset":
        replay.reset()
    else:
        raise HTTPException(400, "action must be start | pause | reset")
    return replay.status()


# --------------------------------------------------------------------------- #
# Razorpay: orders, verification, webhook, dispute simulation
# --------------------------------------------------------------------------- #
@app.post("/api/orders")
def api_create_order(payload: dict = Body(...)) -> dict:
    if not settings.razorpay_ready:
        raise HTTPException(503, "Razorpay keys not configured")
    amount = int(payload.get("amount_paise", 50000))
    try:
        return create_order(amount, payload.get("receipt", "sentinel_demo"),
                            payload.get("notes"))
    except Exception as e:  # noqa: BLE001
        raise HTTPException(502, f"razorpay error: {e}")


@app.post("/api/verify")
def api_verify(payload: dict = Body(...), s: Session = Depends(get_session)) -> dict:
    ok = verify_payment_signature(
        payload.get("razorpay_order_id", ""), payload.get("razorpay_payment_id", ""),
        payload.get("razorpay_signature", ""),
    )
    if not ok:
        raise HTTPException(400, "signature verification failed")
    if "payment" in payload:
        txn = ingest_payment(s, payload["payment"], source="razorpay")
        return {"verified": True, "transaction_id": txn.id, "score": txn.score}
    return {"verified": True}


@app.post("/webhook/razorpay")
async def razorpay_webhook(
    request: Request, x_razorpay_signature: str | None = Header(default=None),
    s: Session = Depends(get_session),
) -> dict:
    body = await request.body()
    if settings.razorpay_webhook_secret and not verify_webhook_signature(body, x_razorpay_signature):
        raise HTTPException(400, "invalid webhook signature")
    evt = json.loads(body)
    etype = evt.get("event", "")
    payload = evt.get("payload", {})

    def entity(*keys):
        for k in keys:
            node = payload.get(k)
            if isinstance(node, dict) and isinstance(node.get("entity"), dict):
                return node["entity"]
        return None

    handled = "ignored"
    try:
        if etype in ("payment.captured", "payment.authorized"):
            if (e := entity("payment")):
                ingest_payment(s, e, source="razorpay")
                handled = "payment"
        elif etype.startswith("payment.dispute."):
            d = entity("dispute", "payment.dispute")
            if d is not None:
                # dispute entity should carry payment_id; fall back to the payment node
                d.setdefault("payment_id", (entity("payment") or {}).get("id"))
                if etype == "payment.dispute.created":
                    ingest_dispute(s, d, source="razorpay")
                else:
                    update_dispute_status(s, d["id"], d.get("status", etype.rsplit(".", 1)[-1]))
                handled = etype
    except Exception as ex:  # noqa: BLE001 — never make Razorpay retry on our bug
        return {"event": etype, "handled": "error", "detail": str(ex)}
    return {"event": etype, "handled": handled}


@app.post("/api/simulate/dispute")
def simulate_dispute(payload: dict = Body(default={}), s: Session = Depends(get_session)) -> dict:
    txn_id = payload.get("txn_id")
    if txn_id:
        txn = s.get(Transaction, txn_id)
        if not txn:
            raise HTTPException(404, "transaction not found")
        disp = ingest_dispute(s, {
            "payment_id": txn_id, "phase": "fraud", "reason_code": "10.4",
            "amount": int(txn.amount * 100), "status": "open",
        }, source="simulated")
    else:
        disp = simulate_dispute_auto(s)
    if disp is None:
        raise HTTPException(404, "no eligible transaction to dispute yet")
    return {"dispute_id": disp.id, "payment_id": disp.payment_id,
            "was_flagged": disp.was_flagged, "lead_time_hours": disp.lead_time_hours,
            "ring_id": disp.ring_id}
