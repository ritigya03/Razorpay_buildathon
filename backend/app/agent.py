"""Phase 5 — the risk-analyst agent.

A conversational layer over the same event store the dashboard reads. The agent
is given read-only tools (situation summary, flagged rings, ring detail, recent
disputes) and a system brief; it grounds every number in a tool call and writes
plain-language forensic reports / escalation notes a human reviewer can act on.

Advisory only: it never blocks anything, and it has no write tools. Powered by
Google Gemini (free tier) — see backend/.env (SENTINEL_GEMINI_API_KEY).
"""
# NB: no `from __future__ import annotations` here on purpose — the Gemini SDK's
# automatic-function-calling coerces tool args against the real annotation
# objects (isinstance checks), which breaks if annotations are lazy strings.

import functools
import threading

from sqlalchemy import func
from sqlmodel import Session, select

from .config import settings
from .db import engine
from .models import Dispute, Ring, Transaction
from .replay import replay

try:  # keep the backend importable even if the package isn't installed
    from google import genai
    from google.genai import types as genai_types
    from google.genai import errors as genai_errors
    _IMPORT_OK = True
except Exception:  # noqa: BLE001
    genai = None  # type: ignore
    genai_types = None  # type: ignore
    genai_errors = None  # type: ignore
    _IMPORT_OK = False


# --------------------------------------------------------------------------- #
# tool-call recorder — AFC runs the tool fns synchronously on the calling
# thread, so a plain module list (reset under the agent lock) is enough to
# surface "which tools ran" back to the UI.
# --------------------------------------------------------------------------- #
_tool_calls: list[dict] = []


def _record(tool: str, args: dict, summary: str) -> None:
    _tool_calls.append({"tool": tool, "args": args, "summary": summary})


def _tool(fn):
    """Wrap a tool: record the call, and turn any exception into an {"error": ...}
    result (the model should see a clean message, never a raw traceback), while
    still surfacing it in tool_calls for the UI. functools.wraps keeps the real
    signature/annotations so the Gemini SDK can still build the tool schema."""
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        call_args = {**{f"arg{i}": a for i, a in enumerate(args)}, **kwargs}
        try:
            result = fn(*args, **kwargs)
            _record(fn.__name__, call_args, result.get("_summary", "ok"))
            result.pop("_summary", None)
            return result
        except Exception as e:  # noqa: BLE001
            _record(fn.__name__, call_args, f"ERROR: {e}")
            return {"error": f"{type(e).__name__}: {e}"}
    return wrapper


# --------------------------------------------------------------------------- #
# read-only tools (the agent's whole surface area)
#
# NOTE: the ring engine does a full rebuild every replay tick, so a ring's
# numeric id is NOT stable. Rings are addressed here by `rank` (1 = highest
# risk, from list_flagged_rings) or by their stable `ring_key` ("kind|key").
# --------------------------------------------------------------------------- #
@_tool
def get_situation_summary() -> dict:
    """Current state of the monitored feed: replay progress, transaction and
    fraud counts, how many transactions the model has flagged and how many of
    those are truly fraud (live precision so far), how many coordinated rings
    are flagged, and the dispute / lead-time picture. Call this first for any
    "what is happening" / "brief me" style question. Takes no arguments."""
    with Session(engine) as s:
        def count(model, *where):
            q = select(func.count()).select_from(model)
            for w in where:
                q = q.where(w)
            return int(s.exec(q).one())

        n_txn = count(Transaction)
        n_fraud = count(Transaction, Transaction.is_fraud == 1)
        n_flagged = count(Transaction, Transaction.flagged == True)  # noqa: E712
        tp = count(Transaction, Transaction.flagged == True, Transaction.is_fraud == 1)  # noqa: E712
        rings_total = count(Ring)
        rings_flagged = count(Ring, Ring.flagged == True)  # noqa: E712
        disputes = s.exec(select(Dispute)).all()
        flagged_in_adv = [d for d in disputes if d.was_flagged]
        st = replay.status()

    return {
        "replay_progress_pct": round(st["progress"] * 100, 1),
        "replay_virtual_day": round(st["virtual_day"], 1),
        "transactions_ingested": n_txn,
        "fraud_transactions": n_fraud,
        "flagged_transactions": n_flagged,
        "flagged_true_positive": tp,
        "live_precision_so_far": round(tp / n_flagged, 3) if n_flagged else None,
        "rings_total": rings_total,
        "rings_flagged": rings_flagged,
        "disputes": len(disputes),
        "disputes_we_flagged_before_the_chargeback": len(flagged_in_adv),
        "avg_lead_time_hours": (
            round(sum(d.lead_time_hours or 0 for d in flagged_in_adv) / len(flagged_in_adv), 1)
            if flagged_in_adv else None
        ),
        "_summary": f"{n_flagged} flagged txns, {rings_flagged} flagged rings",
    }


def _flagged_rings(s: Session, limit: int = 200) -> list[Ring]:
    return list(s.exec(
        select(Ring).where(Ring.flagged == True)  # noqa: E712
        .order_by(Ring.score_mean.desc()).limit(limit)
    ).all())


def _ring_row(rank: int, r: Ring) -> dict:
    return {
        "rank": rank, "ring_key": f"{r.kind}|{r.key}", "kind": r.kind,
        "fingerprint": r.key,
        "source": r.source,                      # "replay" (held-out) | "razorpay" (live) | "mixed"
        "merchants_spanned": r.n_merchants,      # >1 = cross-merchant (razorpay)
        "transactions": r.size, "distinct_accounts": r.distinct_members,
        "distinct_cards": r.distinct_cards,
        "risk_mean": round(r.score_mean, 3), "risk_max": round(r.score_max, 3),
        "amount_total": round(r.amount_total, 2),
        "ground_truth_fraud": r.n_fraud, "disputed": r.n_disputed,
    }


@_tool
def list_flagged_rings(limit: int = 12) -> dict:
    """List the coordinated fraud rings currently flagged for review, worst
    first. Each entry has: `rank` (1 = highest risk), `ring_key` (the stable
    identifier — pass this or the rank to get_ring_detail), kind (device or
    address), the shared fingerprint, transaction count, distinct accounts and
    cards, mean/max risk, total amount, and ground-truth fraud / disputed counts.

    Args:
        limit: maximum rings to return (default 12).
    """
    limit = max(1, min(int(limit), 50))
    with Session(engine) as s:
        rings = _flagged_rings(s, limit)
        rows = [_ring_row(i + 1, r) for i, r in enumerate(rings)]
    return {"count": len(rows), "rings": rows,
            "_summary": f"returned {len(rows)} flagged rings"}


@_tool
def get_ring_detail(ring: str) -> dict:
    """Full detail for one flagged ring: its summary plus every member
    transaction (id, risk, amount, card id, account, ground-truth fraud/disputed
    flags). Use this to write a forensic explanation or an escalation note.

    Args:
        ring: which ring — either its `rank` as a string ("1" for the
            highest-risk ring) or its `ring_key` ("device|SM-J105B ..."), both
            from list_flagged_rings. Numeric ids are not stable; do not use them.
    """
    ref = str(ring).strip()
    with Session(engine) as s:
        rings = _flagged_rings(s)
        if not rings:
            return {"error": "no flagged rings right now", "_summary": "no rings"}

        chosen = None
        if ref.isdigit() and 1 <= int(ref) <= len(rings):
            chosen = rings[int(ref) - 1]
        else:
            for r in rings:
                if ref in (f"{r.kind}|{r.key}", r.key):
                    chosen = r
                    break
        if chosen is None:
            return {
                "error": f"no ring matched {ref!r}",
                "valid_ranks": f"1..{len(rings)}",
                "valid_keys": [f"{r.kind}|{r.key}" for r in rings[:12]],
                "_summary": f"no match for {ref!r}",
            }

        rank = rings.index(chosen) + 1
        members = s.exec(
            select(Transaction).where(Transaction.ring_id == chosen.id)
            .order_by(Transaction.ts)
        ).all()
        detail = _ring_row(rank, chosen)
        detail["window_hours"] = round((chosen.last_ts - chosen.first_ts) / 3600.0, 1)
        detail["members"] = [{
            "txn_id": m.id, "risk": round(m.score, 3), "amount": round(m.amount, 2),
            "card_id": m.card_id, "account": m.uid,
            "is_fraud": m.is_fraud, "disputed": m.disputed,
        } for m in members[:40]]
    detail["_summary"] = (f"rank #{rank} {chosen.kind} ring: {detail['transactions']} txns, "
                          f"{detail['ground_truth_fraud']} fraud")
    return detail


@_tool
def get_recent_disputes(limit: int = 10) -> dict:
    """The most recent chargeback disputes, newest first: dispute id, the
    payment it disputes, amount, status, whether Sentinel had already flagged
    that transaction, and how many hours earlier it was flagged (lead time).

    Args:
        limit: maximum disputes to return (default 10).
    """
    limit = max(1, min(int(limit), 50))
    with Session(engine) as s:
        disputes = s.exec(
            select(Dispute).order_by(Dispute.created_at.desc()).limit(limit)
        ).all()
        rows = [{
            "dispute_id": d.id, "payment_id": d.payment_id,
            "amount": round(d.amount, 2), "status": d.status,
            "was_flagged_in_advance": d.was_flagged,
            "lead_time_hours": round(d.lead_time_hours, 1) if d.lead_time_hours is not None else None,
        } for d in disputes]
    return {"count": len(rows), "disputes": rows,
            "_summary": f"returned {len(rows)} disputes"}


@_tool
def get_federated_report() -> dict:
    """The federated-learning results (Phase 4 + Phase 8). The replay / live feed
    you see through the other tools is the CENTRALISED baseline; this tool
    reports the separate experiments that federate the pipeline so no merchant
    shares raw data. Call this whenever asked what federated learning is doing,
    or for the privacy story. Takes no arguments. 404-safe: returns an "error"
    field if the reports haven't been generated."""
    import json

    base = settings.metrics_file.parent
    out: dict = {}

    p8 = base / "fl_ring_metrics.json"
    if p8.exists():
        d = json.loads(p8.read_text())
        gt = d["ground_truth"]
        sweep = {r["epsilon"]: r["combined"]["f1"] for r in d["dp_sweep"] if r["epsilon"]}
        out["phase8_federated_ring_detection"] = {
            "what": "8 merchants each release only DP-noised salted-HMAC risk-bucket "
                    "histograms; the aggregator sums them to flag cross-merchant rings. "
                    "Commit/reveal + Merkle root + robust aggregation. Same LightGBM "
                    "per-transaction score as the centralised arm, held fixed.",
            "cross_merchant_fraud_rings_in_test": (
                gt["device"]["cross_merchant_fraud_rings"]
                + gt["address"]["cross_merchant_fraud_rings"]),
            "centralised_f1": d["centralized"]["combined"]["f1"],
            "federated_no_dp_f1": d["federated_no_dp"]["combined"]["f1"],
            "identical_no_dp": (d["centralized"]["combined"]["f1"]
                                == d["federated_no_dp"]["combined"]["f1"]),
            "dp_sweep_f1_by_epsilon": sweep,
            "poison_hot_flood_f1": {
                "no_defense": d["poison_demo"]["hot_flood_no_defense"]["combined"]["f1"],
                "defended": d["poison_demo"]["hot_flood_defended"]["combined"]["f1"],
            },
            "merkle_root": d["federated_no_dp"]["merkle_root"],
        }

    p4 = base / "fl_metrics.json"
    if p4.exists():
        d = json.loads(p4.read_text())
        try:
            eps2 = next(x for x in d["dp_sweep"] if x.get("target_epsilon") == 2)
        except (StopIteration, KeyError):
            eps2 = None
        out["phase4_federated_classifier"] = {
            "what": "8 merchants train a local MLP; FedAvg aggregates the weights "
                    "(not raw data) with Opacus DP-SGD, commit/reveal, poison filter.",
            "centralised_pr_auc": d["centralized"]["pr_auc"],
            "federated_no_dp_pr_auc": d["federated_no_dp"]["pr_auc"],
            "federated_dp_eps2_pr_auc": eps2["pr_auc"] if eps2 else None,
            "note": d["meta"].get("disclaimer", ""),
        }

    if not out:
        return {"error": "federated reports not generated — run `make fl-rings` (and `make fl`)",
                "_summary": "no federated report on disk"}
    out["_summary"] = "federated report (Phase 4 classifier + Phase 8 ring detection)"
    return out


TOOLS = [get_situation_summary, list_flagged_rings, get_ring_detail,
         get_recent_disputes, get_federated_report]

SYSTEM_BRIEF = """\
You are the risk-analyst assistant inside Project Sentinel, an advisory system \
that detects COORDINATED fraud rings — one actor spreading many small \
transactions across cards, devices and merchants so no single merchant sees \
enough to react.

What you are looking at (this is the CENTRALISED baseline):
- A time-compressed replay of a held-out temporal test set (the final 15% of \
the timeline, never seen in training). Every transaction is scored by the real \
LightGBM model and grouped by the real ring engine, all in one place. \
Ground-truth fraud labels are known for the replay, so "lead time" and "flagged \
before the chargeback" claims are backed by truth.
- Real Razorpay test-mode disputes flow in alongside and act as ground-truth \
labels for the money loop.

Federated learning is SEPARATE from the feed above and you only see it through \
get_federated_report. Two experiments federate the pipeline so no merchant \
shares raw data: Phase 4 federates the *classifier* (merchants share model \
weights, not rows; FedAvg + DP-SGD), Phase 8 federates the *ring detection* \
(merchants share only DP-noised hashed histograms; the aggregator sums them to \
find cross-merchant rings). With no DP noise the federated ring detector is \
IDENTICAL to the centralised one (additive histograms decompose losslessly over \
the merchant partition) — that is the point, not a bug. DP adds a real cost \
below epsilon ~= 16, and a poisoned merchant is caught and recovered. If asked \
"what is federated learning doing", call get_federated_report and explain that \
the feed you monitor is the centralised baseline that Phase 8 is measured \
against — do NOT say federated learning is inactive.
- Some rings form from LIVE Razorpay test-mode payments (source="razorpay") \
rather than the replay. A `card` ring = one card identity used across several \
customer identities; an `address` ring on live data = one identity running \
several cards (carding). When `merchants_spanned` > 1 the pattern crosses \
merchants — a coordinated card no single merchant could see alone. Say so \
explicitly: it is the core cross-merchant point. The live path is scored by a \
lightweight rules model, not the graded LightGBM — note that when reporting on a \
source="razorpay" ring.

Addressing a ring: the ring engine rebuilds every tick, so numeric ring ids are \
NOT stable. Identify a ring by its `rank` ("1" = highest risk) or its \
`ring_key` — both come from list_flagged_rings. Never refer to a ring by a \
number you remembered from an earlier turn; re-list if unsure.

How to answer:
- Ground EVERY number in a tool call. Never invent ids, counts or scores. If a \
tool returns an "error" field, tell the user plainly and, if useful, re-list.
- Be concise and concrete. Lead with the finding, then the evidence.
- "Brief me" / "what happened" -> call get_situation_summary (and \
list_flagged_rings / get_recent_disputes as needed), then give a 4-6 line \
briefing.
- "Explain the top ring" / "forensic report" -> call get_ring_detail("1") (or \
the rank / ring_key the user named), then write: (1) what was observed (shared \
fingerprint, # accounts/cards, window, amounts), (2) why it is consistent with \
coordination, (3) confidence and the honest caveat (the ring engine only \
covers the coordinated slice — ~16% of all fraud; device data is present on \
~24% of transactions), (4) recommended next step.
- "Mitigation" / "escalation note" -> call get_ring_detail(...), then draft a \
short note (5-8 lines) a risk manager could send to the merchant / issuer.
- "What is federated learning doing" / privacy story -> call \
get_federated_report and summarise Phase 8 (federated == centralised ring \
detection at F1 with no DP; DP cost below epsilon ~= 16; poison caught) and \
Phase 4 (federated classifier ~= centralised). Make clear the live feed is the \
centralised baseline for that comparison.

Hard rules:
- Sentinel is ADVISORY. Never recommend automatic blocking. Recommend human \
review, step-up auth, watchlisting the fingerprint, or issuer escalation — any \
block is a human decision made after that review.
- Do not claim real-time or production deployment. This is a demo over replayed \
held-out data plus Razorpay test mode.
"""


# --------------------------------------------------------------------------- #
class SentinelAgent:
    def __init__(self) -> None:
        self.ok = False
        self.error: str | None = None
        self._client = None
        self._chats: dict[str, object] = {}
        self._lock = threading.Lock()

        if not _IMPORT_OK:
            self.error = "google-genai not installed (pip install -r backend/requirements.txt)"
            return
        key = settings.gemini_key()
        if not key:
            self.error = "no Gemini API key — set SENTINEL_GEMINI_API_KEY in backend/.env"
            return
        try:
            self._client = genai.Client(api_key=key)
            self.ok = True
        except Exception as e:  # noqa: BLE001
            self.error = f"gemini client init failed: {e}"

    # ------------------------------------------------------------------ #
    def _new_chat(self):
        cfg = genai_types.GenerateContentConfig(
            system_instruction=SYSTEM_BRIEF,
            tools=TOOLS,
            temperature=0.3,
            automatic_function_calling=genai_types.AutomaticFunctionCallingConfig(
                maximum_remote_calls=8
            ),
        )
        return self._client.chats.create(model=settings.gemini_model, config=cfg)

    def reset(self, session_id: str = "default") -> None:
        with self._lock:
            self._chats.pop(session_id, None)

    def chat(self, message: str, session_id: str = "default") -> dict:
        if not self.ok:
            raise RuntimeError(self.error or "agent not configured")
        message = (message or "").strip()
        if not message:
            raise ValueError("empty message")

        with self._lock:
            chat = self._chats.get(session_id)
            if chat is None:
                chat = self._new_chat()
                self._chats[session_id] = chat
            _tool_calls.clear()
            try:
                resp = chat.send_message(message)
            except genai_errors.APIError as e:  # noqa: BLE001
                code = getattr(e, "code", None)
                if code == 429:
                    raise RuntimeError(
                        "RATE_LIMIT: Gemini free-tier quota hit — wait ~30s and try again."
                    )
                raise RuntimeError(f"gemini error {code or ''}: {getattr(e, 'message', e)}")
            calls = list(_tool_calls)

        try:
            text = (resp.text or "").strip()
        except Exception:  # noqa: BLE001 — blocked / no text part
            text = ""
        if not text:
            text = "_(the model returned no text — try rephrasing)_"

        return {
            "reply": text,
            "tool_calls": calls,
            "model": settings.gemini_model,
        }

    def status(self) -> dict:
        return {"ok": self.ok, "model": settings.gemini_model if self.ok else None,
                "error": self.error}


agent = SentinelAgent()
