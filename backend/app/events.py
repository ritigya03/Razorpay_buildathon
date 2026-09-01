"""Ingest payments and disputes (Razorpay webhooks + demo simulation)."""
from __future__ import annotations

import time
import uuid

from sqlmodel import Session, select

from .models import Dispute, Ring, Transaction
from .replay import replay
from .rings import recompute_rings
from .rules import score_live_payment


# --------------------------------------------------------------------------- #
def _clean(v) -> str | None:
    s = str(v).strip() if v not in (None, "") else ""
    return s or None


def _card_identity(entity: dict) -> str | None:
    """A real (non-unique) card grouping key from the fields a Razorpay payment
    webhook actually carries: network|issuer|type|last4. Non-card methods fall
    back to the method bucket (upi / wallet / netbanking)."""
    card = entity.get("card") or {}
    if card.get("last4"):
        parts = [_clean(card.get("network")), _clean(card.get("issuer")),
                 _clean(card.get("type")), f"last4:{_clean(card.get('last4'))}"]
        return "|".join(p for p in parts if p)
    return _clean(entity.get("method"))


def ingest_payment(session: Session, entity: dict, source: str = "razorpay") -> Transaction:
    pid = entity["id"]
    existing = session.get(Transaction, pid)
    if existing:
        return existing
    amount = entity.get("amount", 0) / 100.0
    email = entity.get("email")
    contact = entity.get("contact")
    card = entity.get("card") or {}
    international = str(card.get("international", "")).lower() in ("true", "1")

    dom = (email or "").split("@")[-1].lower() or None
    card_id = _card_identity(entity)
    account = _clean(contact) or _clean(email)              # the "customer identity"
    merchant = _clean((entity.get("notes") or {}).get("sentinel_merchant"))

    score, reasons = score_live_payment(
        session, amount=amount, email=email, contact=contact,
        international=international, method=entity.get("method"),
        card_id=card_id, account=account,
    )
    txn = Transaction(
        id=pid, source=source, ts=float(entity.get("created_at", time.time())),
        amount=amount, email_domain=dom,
        card_id=card_id, device_id=None, uid=account, merchant=merchant,
        addr_key=(f"rzp|{dom}|{contact}" if dom and contact else None),
        score=score, scorer="rules", flagged=score >= 0.5,
        is_fraud=None,
    )
    session.add(txn)
    session.commit()
    session.refresh(txn)
    now = replay.virtual_now or time.time()
    recompute_rings(session, now - 7 * 86400)
    return txn


# --------------------------------------------------------------------------- #
def _lead_time_hours(txn: Transaction) -> float | None:
    if not txn.flagged and txn.ring_id is None:
        return None
    ref = replay.virtual_now if txn.source == "replay" and replay.virtual_now else time.time()
    return max((ref - txn.ts) / 3600.0, 0.0)


def ingest_dispute(session: Session, entity: dict, source: str = "razorpay") -> Dispute | None:
    payment_id = entity.get("payment_id")
    txn = session.get(Transaction, payment_id) if payment_id else None
    if txn is None:
        return None
    did = entity.get("id") or f"sim_{uuid.uuid4().hex[:12]}"
    if session.get(Dispute, did):
        return session.get(Dispute, did)

    txn.disputed = True
    txn.dispute_outcome = entity.get("status", "open")
    ring = session.get(Ring, txn.ring_id) if txn.ring_id else None
    if ring:
        ring.n_disputed = (ring.n_disputed or 0) + 1
        session.add(ring)

    disp = Dispute(
        id=did, payment_id=txn.id, source=source,
        phase=entity.get("phase"), reason_code=entity.get("reason_code"),
        amount=entity.get("amount", txn.amount * 100) / 100.0 if entity.get("amount") else txn.amount,
        status=entity.get("status", "open"),
        was_flagged=bool(txn.flagged or txn.ring_id),
        lead_time_hours=_lead_time_hours(txn),
        ring_id=txn.ring_id,
    )
    session.add(txn)
    session.add(disp)
    session.commit()
    session.refresh(disp)
    return disp


def update_dispute_status(session: Session, dispute_id: str, status: str) -> Dispute | None:
    disp = session.get(Dispute, dispute_id)
    if disp is None:
        return None
    disp.status = status
    txn = session.get(Transaction, disp.payment_id)
    if txn:
        txn.dispute_outcome = status
        session.add(txn)
    session.add(disp)
    session.commit()
    session.refresh(disp)
    return disp


# --------------------------------------------------------------------------- #
def simulate_dispute_auto(session: Session) -> Dispute | None:
    """Pick a flagged transaction sitting in a flagged ring and dispute it.
    Prefers a held-out replay fraud (truth-backed lead time); falls back to a
    flagged live Razorpay payment in a flagged ring."""
    for src in ("replay", "razorpay"):
        q = select(Transaction).where(
            Transaction.source == src, Transaction.disputed == False,  # noqa: E712
            Transaction.ring_id != None, Transaction.flagged == True,  # noqa: E711,E712
        )
        if src == "replay":
            q = q.where(Transaction.is_fraud == 1)
        for txn in session.exec(q.order_by(Transaction.score.desc())).all():
            ring = session.get(Ring, txn.ring_id)
            if ring and ring.flagged:
                return ingest_dispute(session, {
                    "payment_id": txn.id, "phase": "fraud", "reason_code": "10.4",
                    "amount": int(txn.amount * 100), "status": "open",
                }, source="simulated")
    return None
