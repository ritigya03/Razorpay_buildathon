# Project Sentinel — reproducibility entrypoints
# `make reproduce` regenerates every number in report/*.json (seed = 42).

DATA_DIR ?= /Users/ritigya/Downloads/ieee-fraud-detection
PY := .venv/bin/python
PIP := .venv/bin/pip

.PHONY: venv splits baseline ring-features ring-engine reproduce backend test clean

venv:
	python3 -m venv .venv
	$(PIP) install -U pip
	$(PIP) install -r requirements.txt

# Temporal 70/15/15 split of train_transaction.csv (+ identity join). Never shuffled.
splits:
	SENTINEL_DATA_DIR=$(DATA_DIR) $(PY) data/prepare_splits.py

# Phase 1 — LightGBM on raw features. Held-out test touched once. -> report/metrics.json
baseline:
	$(PY) train/baseline.py

# Phase 2a — same model + explicit ring/entity features (the documented NEGATIVE
# result: +0.003 PR-AUC, within noise). -> report/metrics_ring.json
ring-features:
	$(PY) train/model_ring.py

# Phase 2b — the ring engine: group transactions into coordinated device/address
# rings, score at ring level. -> report/ring_metrics.json
ring-engine:
	$(PY) train/ring_engine.py

reproduce: splits baseline ring-engine
	@echo ""
	@echo "Done -> report/metrics.json, report/ring_metrics.json"

# Phase 3 — run the backend (needs `make baseline` first for the model + spec)
backend:
	.venv/bin/uvicorn backend.app.main:app --reload --port 8000

test:
	.venv/bin/python -m pytest backend/tests/ -q

clean:
	rm -rf data/splits report/metrics.json report/metrics_ring.json report/ring_metrics.json report/figures
