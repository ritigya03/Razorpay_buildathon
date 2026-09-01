"""Ring engine at runtime.

Groups the transactions currently in the store (within a trailing window) into
coordinated device rings and address rings — the same conservative definition as
train/ring_engine.py (no transitive union-find). A ring is flagged when its
members' mean risk clears a threshold (default taken from the offline tuning in
report/ring_metrics.json). Advisory only.
"""
from __future__ import annotations

import json
import re

from sqlmodel import Session, select

from .config import REPO_ROOT, settings
from .models import Alert, Ring, Transaction

_GENERIC_DEV = {"Windows", "iOS Device", "MacOS", "Linux", "Trident/7.0", "other"}


def is_specific_device(s: str | None) -> bool:
    return bool(s) and s not in _GENERIC_DEV and bool(re.search(r"[0-9 ]", s))


def _default_threshold() -> float:
    try:
        j = json.loads((REPO_ROOT / "report" / "ring_metrics.json").read_text())
        return float(j["rings"]["device"]["ring_threshold"])
    except Exception:
        return 0.10


RING_THRESHOLD = _default_threshold()


def recompute_rings(session: Session, window_start_ts: float) -> list[Ring]:
    """Full rebuild of rings over [window_start_ts, now]. Returns newly-flagged rings."""
    rows = session.exec(
        select(Transaction).where(Transaction.ts >= window_start_ts)
    ).all()
    if not rows:
        return []

    prev_flagged = {
        r.kind + "|" + r.key
        for r in session.exec(select(Ring).where(Ring.flagged == True)).all()  # noqa: E712
    }

    # clear existing ring state for the window and rebuild from scratch
    for t in rows:
        t.ring_id = None
    for old in session.exec(select(Ring)).all():
        session.delete(old)
    session.flush()

    groups: dict[tuple[str, str], list[Transaction]] = {}
    for t in rows:
        if is_specific_device(t.device_id):
            groups.setdefault(("device", t.device_id), []).append(t)
        if t.addr_key:
            groups.setdefault(("address", t.addr_key), []).append(t)
        # card rings: one card identity across several customer identities.
        # Live Razorpay rows only — the replay's card_id is a BIN/type tuple
        # shared by thousands of unrelated cards (see DEVLOG Phase 2b blob).
        if t.source == "razorpay" and t.card_id:
            groups.setdefault(("card", t.card_id), []).append(t)

    # distinct-member count: accounts for device/card rings, cards for address rings
    def member_key(kind):
        return (lambda x: x.card_id) if kind == "address" else (lambda x: x.uid)

    newly_flagged: list[Ring] = []
    for (kind, key), members in groups.items():
        member_field = member_key(kind)
        distinct_members = len({member_field(m) for m in members if member_field(m)})
        if not (2 <= distinct_members <= settings.ring_cap):
            continue
        scores = [m.score for m in members]
        mean_s = sum(scores) / len(scores)
        srcs = {m.source for m in members}
        ring = Ring(
            kind=kind, key=key, size=len(members),
            source=(srcs.pop() if len(srcs) == 1 else "mixed"),
            distinct_members=distinct_members,
            distinct_cards=len({m.card_id for m in members if m.card_id}),
            n_merchants=len({m.merchant for m in members if m.merchant}),
            score_mean=mean_s, score_max=max(scores),
            amount_total=sum(m.amount for m in members),
            first_ts=min(m.ts for m in members), last_ts=max(m.ts for m in members),
            n_fraud=sum(1 for m in members if m.is_fraud == 1),
            n_disputed=sum(1 for m in members if m.disputed),
            flagged=mean_s >= RING_THRESHOLD,
        )
        session.add(ring)
        session.flush()  # assign ring.id
        for m in members:
            m.ring_id = ring.id
        if ring.flagged and (kind + "|" + key) not in prev_flagged:
            newly_flagged.append(ring)
            session.add(Alert(
                ts=ring.last_ts, kind="ring", ring_id=ring.id,
                score=mean_s,
                summary=(f"{kind} ring — {ring.size} txns, {distinct_members} accounts, "
                         f"{ring.distinct_cards} cards share {key[:48]}; mean risk {mean_s:.2f}"),
            ))
    session.commit()
    return newly_flagged
