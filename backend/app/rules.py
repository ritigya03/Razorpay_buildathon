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

    # velocity from our own store: same contact/email in the last hour
    cutoff = time.time() - 3600
    if contact or email:
        recent = session.exec(
            select(Transaction).where(
                Transaction.ts >= cutoff,
                (Transaction.email_domain == dom) if dom else (Transaction.id == "_none_"),
            )
        ).all()
        if len(recent) >= 3:
            risk += 0.15
            reasons.append(f"{len(recent)} payments from this email in the last hour")

    return min(risk, 0.99), reasons
