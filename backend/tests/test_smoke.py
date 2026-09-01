"""End-to-end: boot the app, let the replay run, exercise the API + dispute loop."""
from __future__ import annotations

import json
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

        # Phase 5 — the agent's read-only tools work off the same store (no LLM
        # call here; that would spend API quota on every `make test`).
        from backend.app import agent as agent_mod

        summ = agent_mod.get_situation_summary()
        assert summ["rings_flagged"] >= 1 and summ["flagged_transactions"] > 0
        listed = agent_mod.list_flagged_rings(5)
        assert listed["count"] >= 1
        assert listed["rings"][0]["rank"] == 1 and "|" in listed["rings"][0]["ring_key"]
        det = agent_mod.get_ring_detail("1")
        assert det.get("members") and det["rank"] == 1
        assert agent_mod.get_ring_detail("999999").get("error")  # bad ref -> clean error


def test_agent_health_and_input_validation():
    with TestClient(app) as client:
        h = client.get("/api/agent/health").json()
        assert set(h) == {"ok", "model", "error"}
        # empty message is rejected before any model call
        r = client.post("/api/agent/chat", json={"message": "  "})
        assert r.status_code == 400


def test_live_razorpay_ring_and_dispute():
    """Live-payment path: coordinated Razorpay payments form a flagged ring
    (same code path as a real webhook), and the dispute loop can close on it."""
    with TestClient(app) as client:
        r = client.post("/api/demo/scenario", json={"kind": "shared_card", "size": 4})
        assert r.status_code == 200
        d = r.json()
        assert len(d["payments"]) == 4
        card_rings = [x for x in d["flagged_rings"] if x["kind"] == "card"]
        assert card_rings, "expected a flagged card ring from the shared-card scenario"
        cr = card_rings[0]
        assert cr["accounts"] >= 3 and cr["cards"] == 1 and cr["merchants"] >= 2

        rings = client.get("/api/rings?source=razorpay&flagged=true").json()
        assert any(rg["kind"] == "card" and rg["n_merchants"] >= 2 for rg in rings)

        # carding: one identity, several cards -> address ring on live data
        client.post("/api/demo/scenario", json={"kind": "carding", "size": 4})
        rings = client.get("/api/rings?source=razorpay&flagged=true").json()
        assert any(rg["kind"] == "address" and rg["distinct_cards"] >= 3 for rg in rings)

        # money loop closes on a flagged live payment in a flagged ring
        disp = client.post("/api/simulate/dispute", json={}).json()
        assert disp["was_flagged"] is True and disp["ring_id"] is not None


def test_federated_live_detection():
    """The live federated protocol over Razorpay payments: per-merchant salted-
    HMAC sketches -> Merkle root -> aggregate -> flag a cross-merchant ring,
    with nothing raw leaving a node."""
    with TestClient(app) as client:
        client.post("/api/demo/scenario", json={"kind": "shared_card", "size": 4})
        d = client.post("/api/fl/detect-live", json={"epsilon": None}).json()

        assert len(d["merkle_root"]) == 64
        assert d["merchant_sketches"], "expected per-merchant sketches"
        blob = json.dumps(d["merchant_sketches"])
        assert "last4" not in blob and "@" not in blob  # no raw card / email leaked

        fed = [r for r in d["federated_rings"] if r["flagged"]]
        assert fed and fed[0]["merchants"] >= 2
        assert d["flagged_federated"] >= 1 and d["flagged_centralized"] >= 1

        # DP variant still returns a well-formed result
        dp = client.post("/api/fl/detect-live", json={"epsilon": 8}).json()
        assert dp["epsilon"] == 8.0 and len(dp["merkle_root"]) == 64


def test_orders_endpoint_requires_keys():
    """/api/orders 503s without Razorpay keys; 200 with them. backend/.env may
    carry real test keys locally, so assert the behaviour that actually holds."""
    from backend.app.config import settings

    with TestClient(app) as client:
        r = client.post("/api/orders", json={"amount_paise": 50000})
        if settings.razorpay_ready:
            assert r.status_code in (200, 502)  # 502 only if Razorpay rejects the call
        else:
            assert r.status_code == 503
