"""SQLModel tables — the event store the dashboard reads from."""
from __future__ import annotations

from datetime import datetime, timezone

from sqlmodel import Field, SQLModel


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class Transaction(SQLModel, table=True):
    id: str = Field(primary_key=True)          # TransactionID (replay) or Razorpay payment_id
    source: str = Field(index=True)            # "replay" | "razorpay"
    ts: float = Field(index=True)              # event time, unix seconds
    amount: float
    email_domain: str | None = None
    card_id: str | None = Field(default=None, index=True)     # card1|card2|card3|card5 (replay)
    device_id: str | None = Field(default=None, index=True)
    uid: str | None = Field(default=None, index=True)
    addr_key: str | None = Field(default=None, index=True)

    merchant: str | None = Field(default=None, index=True)    # synthetic merchant tag (razorpay demo)

    score: float = Field(index=True)          # 0..1 advisory risk
    scorer: str                               # "model" | "rules"
    flagged: bool = Field(default=False, index=True)
    ring_id: int | None = Field(default=None, index=True, foreign_key="ring.id")

    is_fraud: int | None = None               # ground truth (replay only), -1/None if unknown
    disputed: bool = Field(default=False, index=True)
    dispute_outcome: str | None = None        # open|under_review|won|lost|closed

    created_at: datetime = Field(default_factory=_utcnow)


class Ring(SQLModel, table=True):
    id: int | None = Field(default=None, primary_key=True)
    kind: str = Field(index=True)             # "device" | "address" | "card"
    key: str = Field(index=True)              # the shared fingerprint / addr tuple / card identity
    source: str = Field(default="replay", index=True)   # "replay" | "razorpay" | "mixed"
    size: int = 0
    distinct_members: int = 0                 # distinct accounts (device/card) or cards (address)
    distinct_cards: int = 0
    n_merchants: int = 0                      # distinct merchant tags among members (razorpay)
    score_mean: float = 0.0
    score_max: float = 0.0
    amount_total: float = 0.0
    first_ts: float = 0.0
    last_ts: float = 0.0
    flagged: bool = Field(default=False, index=True)
    n_fraud: int = 0                          # ground truth among members (replay)
    n_disputed: int = 0
    updated_at: datetime = Field(default_factory=_utcnow)


class Dispute(SQLModel, table=True):
    id: str = Field(primary_key=True)         # Razorpay dispute id, or "sim_..."
    payment_id: str = Field(index=True, foreign_key="transaction.id")
    source: str = "razorpay"                  # "razorpay" | "simulated"
    phase: str | None = None                  # fraud|retrieval|chargeback|...
    reason_code: str | None = None
    amount: float = 0.0
    status: str = "open"
    lead_time_hours: float | None = None      # how long before the dispute we flagged it
    was_flagged: bool = False
    ring_id: int | None = Field(default=None, foreign_key="ring.id")
    created_at: datetime = Field(default_factory=_utcnow)


class Alert(SQLModel, table=True):
    id: int | None = Field(default=None, primary_key=True)
    ts: float = Field(index=True)
    kind: str                                 # "ring" | "txn"
    ring_id: int | None = Field(default=None, foreign_key="ring.id")
    txn_id: str | None = Field(default=None, foreign_key="transaction.id")
    summary: str = ""
    score: float = 0.0
    created_at: datetime = Field(default_factory=_utcnow)
