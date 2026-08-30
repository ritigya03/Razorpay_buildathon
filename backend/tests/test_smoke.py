"""End-to-end: boot the app, let the replay run, exercise the API + dispute loop."""
from __future__ import annotations

import time

from fastapi.testclient import TestClient

from backend.app.main import app


def test_replay_pipeline_and_dispute_loop():
    with TestClient(app) as client:
        assert client.get("/api/health").json()["model_loaded"] is True

        for _ in range(40):
            time.sleep(1.0)
            st = client.get("/api/stats").json()
            if st["rings_flagged"] >= 1 and st["replay"]["ingested"] > 8000:
                break

        st = client.get("/api/stats").json()
        assert st["transactions"] > 3000
        assert st["fraud_transactions"] > 0
        assert st["flagged"] > 0
        assert st["flagged_true_positive"] > 0
        assert st["rings_total"] > 0
        assert st["rings_flagged"] >= 1
        assert st["model_metrics"]["pr_auc_test"] > 0.5

        flagged_txns = client.get("/api/transactions?flagged=true&limit=20").json()
        assert flagged_txns and all(t["flagged"] for t in flagged_txns)

        rings = client.get("/api/rings?flagged=true").json()
        assert rings
        assert all(r["score_mean"] >= 0 for r in rings)
        detail = client.get(f"/api/rings/{rings[0]['id']}").json()
        assert detail["members"]
        assert detail["ring"]["id"] == rings[0]["id"]

        # money loop: dispute a flagged fraudulent txn sitting in a flagged ring
        d = client.post("/api/simulate/dispute", json={}).json()
        assert d["was_flagged"] is True
        assert d["lead_time_hours"] is not None and d["lead_time_hours"] >= 0
        assert d["ring_id"] is not None

        disputes = client.get("/api/disputes").json()
        assert len(disputes) == 1 and disputes[0]["was_flagged"] is True


def test_orders_endpoint_503_without_keys():
    with TestClient(app) as client:
        r = client.post("/api/orders", json={"amount_paise": 50000})
        assert r.status_code == 503
