"""
Shared training + honest evaluation for Project Sentinel models.

Both `baseline.py` (Phase 1, raw features) and `model_ring.py` (Phase 2, +ring
features) call `train_and_report()` so their numbers are produced by identical
code and stay directly comparable.

Cost model (stated openly, tunable here):
  * The detector is ADVISORY. A flagged transaction goes to a human reviewer;
    nothing is auto-blocked.
  * False positive -> C_REVIEW currency units (analyst time to clear a legit txn).
  * False negative -> the full TransactionAmt of the missed fraud (loss/chargeback).
  * True positive  -> loss prevented, cost 0 (optimistic; noted).
  * "Do nothing"   -> every fraud becomes a loss = sum of all fraud amounts.

Operating threshold is chosen on VALIDATION (min expected cost), then applied
exactly once to the held-out TEST split.
"""
from __future__ import annotations

import json
import platform
from datetime import datetime, timezone
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import sklearn
from sklearn.metrics import average_precision_score, roc_auc_score

SEED = 42
C_REVIEW = 3.0  # currency units to manually review one flagged legit transaction

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "models"

LGB_PARAMS = dict(
    objective="binary",
    n_estimators=2000,
    learning_rate=0.05,
    num_leaves=63,
    feature_fraction=0.7,
    bagging_fraction=0.8,
    bagging_freq=1,
    min_child_samples=100,
    reg_lambda=1.0,
    # high-cardinality categoricals (DeviceInfo ~1800 values, browsers, etc.)
    # overfit easily via categorical splits under a temporal shift — smooth them.
    min_data_per_group=200,
    cat_smooth=20.0,
    # NOTE: no is_unbalance / scale_pos_weight. We only need good *ranking*
    # (PR-AUC); the operating threshold is chosen later from the cost curve.
    random_state=SEED,
    n_jobs=-1,
    verbose=-1,
)


def expected_cost(y_true, score, amount, threshold):
    flagged = score >= threshold
    fp = int(np.sum(flagged & (y_true == 0)))
    fn_amount = float(np.sum(amount[(~flagged) & (y_true == 1)]))
    return C_REVIEW * fp + fn_amount


def point_metrics(y_true, score, amount, threshold):
    flagged = score >= threshold
    tp = int(np.sum(flagged & (y_true == 1)))
    fp = int(np.sum(flagged & (y_true == 0)))
    fn = int(np.sum(~flagged & (y_true == 1)))
    tn = int(np.sum(~flagged & (y_true == 0)))
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return dict(
        threshold=float(threshold),
        precision=precision, recall=recall, f1=f1,
        tp=tp, fp=fp, fn=fn, tn=tn,
        flagged=int(np.sum(flagged)),
        amount_lost_fn=float(np.sum(amount[~flagged & (y_true == 1)])),
        amount_caught_tp=float(np.sum(amount[flagged & (y_true == 1)])),
        review_cost_fp=C_REVIEW * fp,
        total_expected_cost=C_REVIEW * fp + float(np.sum(amount[~flagged & (y_true == 1)])),
    )


def threshold_for_recall(y_true, score, target_recall):
    order = np.argsort(-score)
    y_sorted = y_true[order]
    total_pos = y_sorted.sum()
    if total_pos == 0:
        return 1.0
    recall = np.cumsum(y_sorted) / total_pos
    idx = min(np.searchsorted(recall, target_recall), len(score) - 1)
    return float(score[order][idx])


def train_and_report(data: dict, *, model_name: str, phase: int,
                     report_path: Path, model_stub: str) -> dict:
    print(f"features: {len(data['feature_names'])}  (categorical: {len(data['cat_features'])})")
    print(f"train {len(data['y_train']):,} | val {len(data['y_val']):,} | test {len(data['y_test']):,}")

    clf = lgb.LGBMClassifier(**LGB_PARAMS)
    clf.fit(
        data["X_train"], data["y_train"],
        eval_set=[(data["X_val"], data["y_val"])],
        eval_metric="auc",
        categorical_feature=data["cat_features"],
        callbacks=[lgb.early_stopping(200, first_metric_only=True), lgb.log_evaluation(100)],
    )
    best_iter = clf.best_iteration_ or LGB_PARAMS["n_estimators"]
    aucs = clf.evals_result_["valid_0"]["auc"]
    print(f"best_iteration: {best_iter}  (val auc best={max(aucs):.4f} final={aucs[-1]:.4f})")

    p_val = clf.predict_proba(data["X_val"])[:, 1]
    p_test = clf.predict_proba(data["X_test"])[:, 1]
    y_val, y_test = data["y_val"], data["y_test"]
    amt_val, amt_test = data["amount_val"], data["amount_test"]

    prauc = dict(val=float(average_precision_score(y_val, p_val)),
                 test=float(average_precision_score(y_test, p_test)))
    rocauc = dict(val=float(roc_auc_score(y_val, p_val)),
                  test=float(roc_auc_score(y_test, p_test)))
    print(f"PR-AUC  val={prauc['val']:.4f}  test={prauc['test']:.4f}")
    print(f"ROC-AUC val={rocauc['val']:.4f}  test={rocauc['test']:.4f}")

    grid = np.unique(np.quantile(p_val, np.linspace(0.80, 0.99995, 400)))
    t_star = min(grid, key=lambda t: expected_cost(y_val, p_val, amt_val, t))

    test_op = point_metrics(y_test, p_test, amt_test, t_star)
    do_nothing = float(np.sum(amt_test[y_test == 1]))
    test_op["do_nothing_cost"] = do_nothing
    test_op["cost_reduction_vs_do_nothing"] = 1 - test_op["total_expected_cost"] / do_nothing
    print(f"\nTEST @ {t_star:.5f} (advisory): precision {test_op['precision']:.3f}  "
          f"recall {test_op['recall']:.3f}  f1 {test_op['f1']:.3f}  "
          f"cost {test_op['total_expected_cost']:,.0f} "
          f"({test_op['cost_reduction_vs_do_nothing']:.1%} < do-nothing)")

    reference_points = []
    for target in (0.50, 0.80, 0.90):
        t = threshold_for_recall(y_val, p_val, target)
        m = point_metrics(y_test, p_test, amt_test, t)
        m["name"] = f"recall~{target:.2f}_on_val"
        reference_points.append(m)

    curve_grid = np.unique(np.quantile(p_test, np.linspace(0.80, 0.99995, 200)))
    cost_curve = [
        dict(threshold=(m := point_metrics(y_test, p_test, amt_test, t))["threshold"],
             cost=m["total_expected_cost"], precision=m["precision"], recall=m["recall"],
             fp=m["fp"], fn=m["fn"], flagged=m["flagged"])
        for t in curve_grid
    ]

    importances = sorted(zip(data["feature_names"], clf.feature_importances_.tolist()),
                         key=lambda kv: -kv[1])[:30]

    report = {
        "meta": {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "model": model_name,
            "phase": phase,
            "seed": SEED,
            "python": platform.python_version(),
            "lib_versions": {"lightgbm": lgb.__version__, "scikit_learn": sklearn.__version__,
                             "numpy": np.__version__, "pandas": pd.__version__},
            "split_fracs": {"train": 0.70, "val": 0.15, "test": 0.15},
            "rows": {"train": int(len(data["y_train"])), "val": int(len(data["y_val"])),
                     "test": int(len(data["y_test"]))},
            "fraud_rate": {"train": float(np.mean(data["y_train"])),
                           "val": float(np.mean(data["y_val"])),
                           "test": float(np.mean(data["y_test"]))},
            "cost_model": {
                "C_REVIEW_per_false_positive": C_REVIEW,
                "false_negative_cost": "TransactionAmt of the missed fraud",
                "true_positive_cost": 0.0,
                "detector_action": "advisory only — flagged txns go to human review, never auto-blocked",
            },
            "n_features": len(data["feature_names"]),
            "n_categorical": len(data["cat_features"]),
            "best_iteration": int(best_iter),
        },
        "pr_auc": prauc,
        "roc_auc": rocauc,
        "operating_point": {"selected_on": "validation — minimum expected cost",
                            "threshold": float(t_star), "test": test_op},
        "reference_points": reference_points,
        "cost_curve_test": cost_curve,
        "top_feature_importance": [{"feature": f, "gain": g} for f, g in importances],
    }

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2))
    MODEL_DIR.mkdir(exist_ok=True)
    clf.booster_.save_model(str(MODEL_DIR / f"{model_stub}.txt"), num_iteration=best_iter)
    print(f"Wrote {report_path}  and  models/{model_stub}.txt")
    return report
