"""Replay engine — streams the held-out test split through the system as a live feed.

The trained model needs ~430 IEEE-CIS features that a real Razorpay webhook does
not carry, so the "live" data the dashboard shows is a time-compressed replay of
the labelled held-out split: every row is scored by the real model and grouped by
the real ring engine, and its ground-truth label is known (so dispute simulation
and lead-time claims are backed by truth). Real Razorpay payments flow in
alongside it (see razorpay_client.py) and are scored by rules.
"""
from __future__ import annotations

import asyncio
import json
import time

import pandas as pd
from sqlmodel import Session, select

from .config import settings
from .db import engine
from .models import Alert, Ring, Transaction
from .rings import recompute_rings
from .scoring import scorer

RING_WINDOW_DAYS = 7
MAX_PER_TICK = 2500  # ingestion ceiling per tick; virtual_now never outruns it


class ReplayEngine:
    def __init__(self) -> None:
        self.loaded = False
        self.running = False
        self.cursor = 0
        self.virtual_now = 0.0
        self._wall_start = 0.0
        self._virtual_start = 0.0
        self._task: asyncio.Task | None = None
        self.rows: pd.DataFrame | None = None
        self.per_txn_threshold = 0.02

    # ------------------------------------------------------------------ #
    def load(self) -> None:
        if self.loaded:
            return
        # `pd.read_parquet` on the whole file peaks at ~850MB above baseline
        # (434 columns x 88,581 rows going through pyarrow's Arrow->pandas
        # conversion) even though the resulting frame is only ~300MB — too
        # much headroom for a memory-capped deploy host. Stream it in
        # row-group-sized batches instead: score each batch, keep only the
        # handful of columns anything downstream actually reads, and drop
        # the rest before the next batch is even read.
        import pyarrow.parquet as pq

        KEEP = ["TransactionID", "TransactionDT", "TransactionAmt", "isFraud",
                "_score", "_email", "_card_id", "_device_id", "_uid", "_addr_key"]
        chunks: list[pd.DataFrame] = []
        pf = pq.ParquetFile(settings.test_split)
        for batch in pf.iter_batches(batch_size=1_000):
            chunk = batch.to_pandas()
            chunk = scorer.entity_keys(chunk)
            chunk["_score"] = scorer.score(chunk)
            chunk["_email"] = chunk["P_emaildomain"].astype("string")
            addr = (chunk["addr1"].astype("string").fillna("n") + "|"
                    + chunk["addr2"].astype("string").fillna("n") + "|"
                    + chunk["P_emaildomain"].astype("string").fillna("n"))
            chunk["_addr_key"] = addr.where(chunk["addr1"].notna() & chunk["P_emaildomain"].notna())
            chunks.append(chunk[KEEP].copy())
        self.rows = pd.concat(chunks, ignore_index=True).sort_values("TransactionDT").reset_index(drop=True)
        self._dt = self.rows["TransactionDT"].to_numpy()
        self._virtual_start = float(self._dt[0])
        self.virtual_now = self._virtual_start
        try:
            self.per_txn_threshold = float(
                json.loads(settings.metrics_file.read_text())["operating_point"]["threshold"]
            )
        except Exception:
            pass
        self.loaded = True

    # ------------------------------------------------------------------ #
    def status(self) -> dict:
        total = 0 if self.rows is None else len(self.rows)
        return {
            "loaded": self.loaded, "running": self.running,
            "ingested": self.cursor, "total": total,
            "progress": (self.cursor / total) if total else 0.0,
            "virtual_day": (self.virtual_now - self._virtual_start) / 86400 if self.loaded else 0.0,
            "days_per_sec": settings.replay_days_per_sec,
            "per_txn_threshold": self.per_txn_threshold,
        }

    def start(self) -> None:
        self.load()
        if self.running:
            return
        self.running = True
        self._wall_start = time.monotonic()
        self._virtual_start_at_resume = self.virtual_now
        self._task = asyncio.create_task(self._loop())

    def pause(self) -> None:
        self.running = False

    def reset(self) -> None:
        self.running = False
        self.cursor = 0
        if self.loaded:
            self.virtual_now = self._virtual_start
        with Session(engine) as s:
            for tbl in (Alert, Transaction, Ring):
                for row in s.exec(select(tbl)).all():
                    s.delete(row)
            s.commit()

    # ------------------------------------------------------------------ #
    async def _loop(self) -> None:
        while self.running and self.cursor < len(self.rows):
            elapsed = time.monotonic() - self._wall_start
            target = (self._virtual_start_at_resume
                      + elapsed * settings.replay_days_per_sec * 86400)
            self._ingest_due(target)
            await asyncio.sleep(1.0)
        self.running = False

    def _ingest_due(self, target_virtual: float) -> None:
        rows, cur, dt = self.rows, self.cursor, self._dt
        end = cur
        while (end < len(rows) and dt[end] <= target_virtual
               and end - cur < MAX_PER_TICK):
            end += 1
        if end == cur:
            # nothing due; move the clock up to (not past) the next unseen row
            self.virtual_now = min(target_virtual, float(dt[cur])) if cur < len(rows) else target_virtual
            return
        batch = rows.iloc[cur:end]
        with Session(engine) as s:
            ids = [str(int(x)) for x in batch["TransactionID"]]
            existing = set(
                s.exec(select(Transaction.id).where(Transaction.id.in_(ids))).all()
            )
            for _, r in batch.iterrows():
                if str(int(r["TransactionID"])) in existing:
                    continue
                score = float(r["_score"])
                flagged = score >= self.per_txn_threshold
                s.add(Transaction(
                    id=str(int(r["TransactionID"])), source="replay",
                    ts=float(r["TransactionDT"]), amount=float(r["TransactionAmt"]),
                    email_domain=None if pd.isna(r["_email"]) else str(r["_email"]),
                    card_id=str(r["_card_id"]), device_id=None if pd.isna(r["_device_id"]) else str(r["_device_id"]),
                    uid=str(r["_uid"]),
                    addr_key=None if pd.isna(r["_addr_key"]) else str(r["_addr_key"]),
                    score=score, scorer="model", flagged=flagged,
                    is_fraud=int(r["isFraud"]),
                ))
                if flagged:
                    s.add(Alert(ts=float(r["TransactionDT"]), kind="txn",
                                txn_id=str(int(r["TransactionID"])), score=score,
                                summary=f"transaction risk {score:.2f} (>= {self.per_txn_threshold:.2f})"))
            s.commit()
        self.cursor = end
        self.virtual_now = float(dt[end - 1])  # clock tracks the data frontier
        with Session(engine) as s:
            recompute_rings(s, self.virtual_now - RING_WINDOW_DAYS * 86400)


replay = ReplayEngine()
