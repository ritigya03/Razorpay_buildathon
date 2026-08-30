"""
Phase 2 model: raw features + ring / entity-graph features.

Runs the *same* evaluation as the Phase 1 baseline (shared `evaluate.py`) so the
two are directly comparable. Writes report/metrics_ring.json.

    python train/model_ring.py   ->   report/metrics_ring.json
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import load_splits, prepare_features  # noqa: E402
from evaluate import ROOT, train_and_report  # noqa: E402
from ring_features import build_entity_features  # noqa: E402


def main() -> None:
    print("Loading splits ...")
    train, val, test = load_splits()
    print("Building ring features ...")
    train, val, test = build_entity_features(train, val, test)
    data = prepare_features(train, val, test)
    train_and_report(
        data,
        model_name="lightgbm — raw transaction features + ring/entity-graph features",
        phase=2,
        report_path=ROOT / "report" / "metrics_ring.json",
        model_stub="ring_lgb",
    )


if __name__ == "__main__":
    main()
