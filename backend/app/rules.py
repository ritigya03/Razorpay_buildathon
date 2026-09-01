"""Rules-based advisory score for LIVE Razorpay payments.

A real payment webhook carries only a handful of fields — nowhere near the ~430
features the LightGBM model needs — so live payments get this transparent
rules score instead, clearly labelled `scorer="rules"` in the store. The ML
model runs on the full-feature replay stream.

Weights are rough and hand-set; the point is a sane, explainable 0..1 signal,
not a trained classifier.
"""
from __future__ import annotations

import time

from sqlmodel import Session, select

from .models import Transaction

# fraud rates by email domain from the IEEE-CIS training split (see data study)
HIGH_RISK_EMAIL = {
    "protonmail.com": 0.41, "mail.com": 0.19, "outlook.es": 0.13, "aim.com": 0.13,
    "outlook.com": 0.09, "hotmail.es": 0.07, "live.com.mx": 0.05,
}


def score_live_payment(
    session: Session, *, amount: float, email: str | None, contact: str | None,
    international: bool = False, method: str | None = None,
    card_id: str | None = None, account: str | None = None,
) -> tuple[float, list[str]]:
    reasons: list[str] = []
    risk = 0.05  # base

    dom = (email or "").split("@")[-1].lower()
    if dom in HIGH_RISK_EMAIL:
        bump = 0.15 + 0.25 * HIGH_RISK_EMAIL[dom]
        risk += bump
        reasons.append(f"email domain {dom} (hist. fraud {HIGH_RISK_EMAIL[dom]:.0%})")

    if international:
        risk += 0.20
        reasons.append("international card")

    if amount and amount < 3:
        risk += 0.15
        reasons.append("very low amount (card testing pattern)")
    elif amount and amount > 2000:
        risk += 0.10
        reasons.append("high amount")

    # --- shared-entity velocity from our own store (the coordinated-ring signal) ---
    cutoff = time.time() - 3600

    if card_id:
        peers = session.exec(
            select(Transaction).where(
                Transaction.ts >= cutoff, Transaction.source == "razorpay",
                Transaction.card_id == card_id,
            )
        ).all()
        accts = {t.uid for t in peers if t.uid} | ({account} if account else set())
        merchants = {t.merchant for t in peers if t.merchant}
        if len(accts) >= 2:
            risk += 0.25 + 0.08 * min(len(accts) - 2, 4)
            reasons.append(f"this card seen on {len(accts)} distinct customer identities in the last hour")
        if len(merchants) >= 2:
            risk += 0.15
            reasons.append(f"this card seen at {len(merchants)} merchants (cross-merchant pattern)")

    if account:
        peers = session.exec(
            select(Transaction).where(
                Transaction.ts >= cutoff, Transaction.source == "razorpay",
                Transaction.uid == account,
            )
        ).all()
        cards = {t.card_id for t in peers if t.card_id} | ({card_id} if card_id else set())
        if len(cards) >= 2:
            risk += 0.25 + 0.08 * min(len(cards) - 2, 4)
            reasons.append(f"this identity used {len(cards)} distinct cards in the last hour (carding)")

    return min(risk, 0.99), reasons
