"""
Phase 1 baseline: LightGBM on raw IEEE-CIS transaction features (no ring features).

The project's insurance policy — a reproducible, honestly-measured
precision/recall + false-positive-cost result on a held-out temporal test set.
Phase 2 (`model_ring.py`) adds ring/entity-graph features and re-runs the same
evaluation (shared `evaluate.py`) to measure the lift.

    python train/baseline.py   ->   report/metrics.json
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import load_splits, prepare_features  # noqa: E402
from evaluate import ROOT, train_and_report  # noqa: E402


def main() -> None:
    print("Loading splits ...")
    train, val, test = load_splits()
    data = prepare_features(train, val, test)
    train_and_report(
        data,
        model_name="lightgbm baseline — raw transaction features, no ring features",
        phase=1,
        report_path=ROOT / "report" / "metrics.json",
        model_stub="baseline_lgb",
    )


if __name__ == "__main__":
    main()
