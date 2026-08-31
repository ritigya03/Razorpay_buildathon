"""
Phase 4 — federated learning + differential privacy side experiment.

Secondary to the graded Phase 1-3 numbers. Produces, all on the untouched
held-out split `data/splits/test.parquet`:

  1. centralized       — one MLP on the pooled data (the ceiling)
  2. federated, no DP   — 8 merchants, SentinelFedAvg (commit/reveal + cosine
                          filter + balance-weighted aggregation), no noise
  3. DP sweep           — federated + Opacus DP-SGD per client, target epsilon in
                          {8, 3, 1.5, 0.5}; records target vs spent epsilon
  4. poison, no defence — 1/8 merchants sends sign-flipped x10 weights, plain
                          FedAvg -> global model collapses
  5. poison, defended   — same attack, SentinelFedAvg -> attack rejected, model
                          recovers to ~the honest federated number

    .venv-fl/bin/python train/fl_experiment.py [--quick]   ->   report/fl_metrics.json
"""
from __future__ import annotations

import argparse
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from flwr.common import ndarrays_to_parameters, parameters_to_ndarrays
from flwr.server.strategy.aggregate import aggregate

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fl_client import MerchantClient  # noqa: E402
from fl_data import build_arrays, merchant_summary  # noqa: E402
from fl_model import (  # noqa: E402
    MLP, centralized_train, evaluate_weights, get_weights, seed_everything,
)
from fl_strategy import COS_THRESHOLD, SentinelFedAvg  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "report" / "fl_metrics.json"
FIG = ROOT / "report" / "figures" / "fl_epsilon_curve.png"
SEED = 42
DELTA = 1e-5
MAX_GRAD_NORM = 1.0
BATCH_SIZE = 4096       # centralized + non-DP federated
DP_BATCH_SIZE = 4096    # large batch is the biggest DP-SGD utility lever (TAN scaling
#                         laws): less effective noise + less clipping bias at fixed epsilon.
#                         Capped at 4096 — per-sample gradients cost batch x params, and
#                         8192 quadrupled wall time for a marginal utility gain here.
LR = 1e-3
SWA_K = 3               # average the last SWA_K rounds' global weights (FedSWA: flatter minima)


# --------------------------------------------------------------------------- #
# in-process federated driver (real Flower strategy, no Ray)
# --------------------------------------------------------------------------- #
LOCAL_STEPS = 20


def run_federated(clients: list[MerchantClient], n_feat: int, rounds: int, *,
                  robust: bool, Xva, yva, Xte, yte, label: str = "") -> dict:
    seed_everything(SEED)
    params = ndarrays_to_parameters(get_weights(MLP(n_feat)))
    strat = SentinelFedAvg(n_merchants=len(clients))
    fraud_counts = {c.cid: c.fraud_count for c in clients}
    num_examples = {c.cid: c.n_local for c in clients}
    round_log = []
    best = {"val_pr_auc": -1.0, "weights": parameters_to_ndarrays(params), "round": 0}
    recent: list[list[np.ndarray]] = []  # last SWA_K rounds' weights, for weight averaging

    for rnd in range(1, rounds + 1):
        gnd = parameters_to_ndarrays(params)
        for c in clients:
            c.compute_update(gnd, rnd)
        commits = {c.cid: c.commit(rnd) for c in clients}
        root = strat.receive_commitments(rnd, commits)
        reveals = {c.cid: c.reveal(rnd) for c in clients}

        if robust:
            params, info = strat.aggregate_reveal(rnd, params, reveals, fraud_counts)
        else:  # plain FedAvg: no verification, weight by row count, trust everyone
            weighted = [([u - w for u, w in zip(reveals[c.cid], gnd)], num_examples[c.cid])
                        for c in clients]
            params = ndarrays_to_parameters([w + d for w, d in zip(gnd, aggregate(weighted))])
            info = {"round": rnd, "merkle_root": root, "rejected": [],
                    "aggregated_merchants": [c.cid for c in clients]}

        w_now = parameters_to_ndarrays(params)
        va = evaluate_weights(w_now, n_feat, Xva, yva)
        te = evaluate_weights(w_now, n_feat, Xte, yte)
        if va["pr_auc"] > best["val_pr_auc"]:
            best = {"val_pr_auc": va["pr_auc"], "weights": w_now, "round": rnd}
        recent.append(w_now)
        recent[:] = recent[-SWA_K:]
        round_log.append({**{k: info[k] for k in ("round", "merkle_root", "rejected",
                                                  "aggregated_merchants")},
                          "val": va, "test": te})
        print(f"  [{label}] round {rnd:2d}/{rounds}  "
              f"agg={len(info['aggregated_merchants'])}/{len(clients)}  rejected={len(info['rejected'])}  "
              f"val {va['pr_auc']:.4f}  test {te['pr_auc']:.4f}  merkle={root[:12]}…")

    # FedSWA: average the last SWA_K rounds' weights. Return whichever of
    # {SWA, best-val-round} scores higher on VALIDATION (never selected on test).
    swa_w = [np.mean(layers, axis=0) for layers in zip(*recent)]
    swa_va = evaluate_weights(swa_w, n_feat, Xva, yva)["pr_auc"]
    if swa_va >= best["val_pr_auc"]:
        final_w, sel, sel_va = swa_w, f"SWA(last {len(recent)})", swa_va
    else:
        final_w, sel, sel_va = best["weights"], f"best-val round {best['round']}", best["val_pr_auc"]

    spent = [c.spent_epsilon for c in clients if c.dp_cfg is not None]
    print(f"  [{label}] selected: {sel}  (val PR-AUC {sel_va:.4f})")
    return {
        "final_weights": final_w,
        "test": evaluate_weights(final_w, n_feat, Xte, yte),
        "val_pr_auc": round(sel_va, 4),
        "selection": sel,
        "best_round": best["round"],
        "best_round_test": evaluate_weights(best["weights"], n_feat, Xte, yte)["pr_auc"],
        "swa_test": evaluate_weights(swa_w, n_feat, Xte, yte)["pr_auc"],
        "rounds": rounds,
        "spent_epsilon_max": max(spent) if spent else None,
        "spent_epsilon_mean": float(np.mean(spent)) if spent else None,
        "audit": strat.audit,
        "round_log": round_log,
    }


def make_clients(data: dict, n_feat: int, *, dp_cfg_fn=None, attacks: dict | None = None,
                 local_steps: int = LOCAL_STEPS) -> list[MerchantClient]:
    X, y, mids = data["X_train"], data["y_train"], data["merchant_ids"]
    attacks = attacks or {}
    clients = []
    for m in range(int(data["n_merchants"])):
        mask = mids == m
        dp_cfg = dp_cfg_fn(int(mask.sum())) if dp_cfg_fn else None
        clients.append(MerchantClient(
            m, X[mask], y[mask], n_feat,
            batch_size=DP_BATCH_SIZE if dp_cfg else BATCH_SIZE, lr=LR,
            local_steps=local_steps, dp_cfg=dp_cfg, attack=attacks.get(m), seed=SEED))
    return clients


def noise_multiplier_for(target_eps: float, n_local: int, total_steps: int) -> float:
    from opacus.accountants.utils import get_noise_multiplier
    return float(get_noise_multiplier(
        target_epsilon=target_eps, target_delta=DELTA,
        sample_rate=min(DP_BATCH_SIZE / max(n_local, 1), 1.0),
        steps=total_steps, accountant="rdp"))


# --------------------------------------------------------------------------- #
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true",
                    help="6 rounds, epsilon in {inf, 1.5} — for iteration")
    args = ap.parse_args()

    # FL is capped at a modest round count on purpose: past ~5-6 rounds this MLP
    # overfits the validation era and val stops tracking test (temporal shift).
    # SWA over the last few rounds is what stabilises the final model.
    rounds = 3 if args.quick else 5
    local_steps = 10 if args.quick else 15
    eps_grid = [4.0] if args.quick else [8.0, 4.0, 2.0, 1.0]
    central_epochs = 4 if args.quick else 6
    total_dp_steps = rounds * local_steps

    data = build_arrays()
    n_feat = int(data["X_train"].shape[1])
    Xte, yte = data["X_test"], data["y_test"]
    Xva, yva = data["X_val"], data["y_val"]
    fed_kw = dict(Xva=Xva, yva=yva, Xte=Xte, yte=yte)
    parts = merchant_summary(data["y_train"], data["merchant_ids"])
    print(f"\nfeatures {n_feat} | merchants {int(data['n_merchants'])} | rounds {rounds}\n")

    # 1. centralized -------------------------------------------------------- #
    print("== centralized ==")
    cen_w, cen_meta = centralized_train(
        data["X_train"], data["y_train"], data["X_val"], data["y_val"],
        epochs=central_epochs, lr=LR, batch_size=BATCH_SIZE, seed=SEED)
    centralized = {**evaluate_weights(cen_w, n_feat, Xte, yte), **cen_meta}
    print(f"   centralized test PR-AUC {centralized['pr_auc']:.4f}  ROC-AUC {centralized['roc_auc']:.4f}")

    # 2. federated, no DP ------------------------------------------------- #
    print("\n== federated, no DP ==")
    fed = run_federated(make_clients(data, n_feat, local_steps=local_steps), n_feat, rounds,
                        robust=True, label="fed", **fed_kw)
    federated_no_dp = {
        **fed["test"], "rounds": rounds,
        "val_pr_auc": fed["val_pr_auc"], "selection": fed["selection"],
        "best_round": fed["best_round"],
        "best_round_test_pr_auc": round(fed["best_round_test"], 4),
        "swa_test_pr_auc": round(fed["swa_test"], 4),
        "per_merchant": parts,
        "honest_rejections": fed["audit"],
        "merkle_root_round1": fed["round_log"][0]["merkle_root"] if fed["round_log"] else "",
    }

    # 3. DP sweep -------------------------------------------------------- #
    print("\n== DP sweep ==")
    dp_sweep = []
    for eps in eps_grid:
        def dp_cfg_fn(n_local, _eps=eps):
            return {"noise_multiplier": noise_multiplier_for(_eps, n_local, total_dp_steps),
                    "max_grad_norm": MAX_GRAD_NORM, "delta": DELTA}
        r = run_federated(make_clients(data, n_feat, dp_cfg_fn=dp_cfg_fn, local_steps=local_steps),
                          n_feat, rounds, robust=True, label=f"dp e={eps}", **fed_kw)
        row = {
            "target_epsilon": eps,
            "spent_epsilon_max": round(r["spent_epsilon_max"], 3),
            "spent_epsilon_mean": round(r["spent_epsilon_mean"], 3),
            "noise_multiplier_range": [
                round(noise_multiplier_for(eps, p["rows"], total_dp_steps), 3)
                for p in (min(parts, key=lambda x: x["rows"]), max(parts, key=lambda x: x["rows"]))
            ],
            **r["test"],
            "val_pr_auc": r["val_pr_auc"], "selection": r["selection"],
            "delta_pr_auc_vs_centralized": round(r["test"]["pr_auc"] - centralized["pr_auc"], 4),
        }
        dp_sweep.append(row)
        print(f"   eps target {eps}  spent≤{row['spent_epsilon_max']}  "
              f"test PR-AUC {row['pr_auc']:.4f}  (Δ vs central {row['delta_pr_auc_vs_centralized']:+.4f})")

    # 4 + 5. poison: no defence vs defended --------------------------- #
    print("\n== poison: 1/8 merchants malicious ==")
    mk = lambda atk: make_clients(data, n_feat, attacks={0: atk}, local_steps=local_steps)  # noqa: E731

    def _reasons(audit):
        return sorted({a["reason"] for a in audit})

    def _scenario(attack: str, robust: bool):
        r = run_federated(mk(attack), n_feat, rounds, robust=robust,
                          label=f"{attack}/{'def' if robust else 'nodef'}", **fed_kw)
        return {
            **r["test"],
            "rejected_updates": len(r["audit"]),
            "rejection_reasons": _reasons(r["audit"]),
            "rejections": r["audit"][:16],
            "delta_vs_honest_federated": round(r["test"]["pr_auc"] - federated_no_dp["pr_auc"], 4),
        }

    baseline_nodef = _scenario("sign_flip", robust=False)   # plain FedAvg, no checks
    robustness_demo = {
        "malicious_merchants": [0],
        "no_defense_plain_fedavg": {
            "attack": "sign_flip (global - 10x honest delta), 1/8 merchants",
            **baseline_nodef,
        },
        "defended": {
            "sign_flip": {"attack": "reversed direction, 10x magnitude",
                          "expected_catch": "norm_outlier", **_scenario("sign_flip", robust=True)},
            "flip": {"attack": "reversed direction, honest magnitude",
                     "expected_catch": "cosine_outlier", **_scenario("flip", robust=True)},
            "last_mover": {"attack": "honest commit, poisoned reveal",
                           "expected_catch": "commitment_mismatch",
                           **_scenario("last_mover", robust=True)},
        },
    }
    for name, s in [("no-defense (sign_flip)", robustness_demo["no_defense_plain_fedavg"]),
                    *[(f"defended/{k}", v) for k, v in robustness_demo["defended"].items()]]:
        print(f"   {name:<24} PR-AUC {s['pr_auc']:.4f}  "
              f"rejected {s['rejected_updates']:>2} {s['rejection_reasons']}")

    # ------------------------------------------------------------------ #
    report = {
        "meta": {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "phase": 4,
            "model": f"MLP {n_feat}->128->32->1, tanh, dropout 0.25, focal loss "
                     "(gamma=2, alpha=0.75). tanh + small net + large DP batch follow "
                     "DP-SGD utility guidance (DPMLBench, TAN). Features: standardized "
                     "IEEE-CIS numerics + frequency-encoded categoricals + 27 leakage-safe "
                     "engineered features (OOF reputation / causal velocity / 7d ring "
                     "structure, from train/ring_features.py).",
            "model_selection": f"FedSWA — mean of the last {SWA_K} rounds' global weights, "
                               "or the best single validation round, whichever scores higher "
                               "on validation (never on test).",
            "framework": "Flower 1.35 — FedAvg strategy + flwr.common types; in-process "
                         "round loop (no Ray/run_simulation: no Python 3.14 wheel, and the "
                         "two-phase commit/reveal does not map to one round-trip per round)",
            "n_merchants": int(data["n_merchants"]),
            "fl_rounds": rounds,
            "local_steps_per_round": local_steps,
            "batch_size": BATCH_SIZE,
            "dp_batch_size": DP_BATCH_SIZE,
            "lr": LR,
            "partition": "non-IID — hash(card1) % 8; per-merchant rows + fraud rate below",
            "seed": SEED,
            "python": platform.python_version(),
            "test_split": "data/splits/test.parquet (held out, untouched since Phase 1)",
            "dp": {
                "mechanism": "Opacus DP-SGD per client, Poisson sampling, RDP accountant",
                "max_grad_norm": MAX_GRAD_NORM,
                "delta": DELTA,
                "accounting_note": "noise multiplier fixed per merchant to hit target "
                                   "epsilon after all rounds; one persistent PrivacyEngine "
                                   "per client so the accountant accumulates. spent_epsilon "
                                   "is the cumulative per-client budget after the final round.",
            },
            "robustness": {
                "commit_reveal": "SHA-256 digest committed before weights revealed; "
                                 "digest mismatch on reveal -> update rejected (last-mover defense)",
                "norm_filter": "delta L2 norm > 3x median norm -> rejected (scaled-negation attacks)",
                "cosine_filter": "delta cosine distance from the median that is both a "
                                 ">3-MAD outlier and past an 0.85 floor -> rejected (direction-"
                                 f"reversal attacks). project_sentinel.md's fixed {COS_THRESHOLD} "
                                 "threshold rejected every honest client on this non-IID split; "
                                 "see PHASE4.md.",
                "aggregation": "balance-weighted by fraud (minority-class) row count",
                "merkle_root": "binary SHA-256 tree over each round's commitments, logged",
            },
            "disclaimer": "SECONDARY experiment. Does NOT affect the graded LightGBM "
                          "baseline (report/metrics.json) or ring engine "
                          "(report/ring_metrics.json). The MLP is weaker than LightGBM on "
                          "this data by construction — the comparisons here are "
                          "federated-vs-centralized and DP-vs-no-DP within the same MLP.",
        },
        "centralized": centralized,
        "federated_no_dp": federated_no_dp,
        "dp_sweep": dp_sweep,
        "robustness_demo": robustness_demo,
    }
    OUT.write_text(json.dumps(report, indent=2))
    print(f"\nWrote {OUT}")

    _plot(report)
    _sanity(report)


# --------------------------------------------------------------------------- #
def _plot(report: dict) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cen = report["centralized"]["pr_auc"]
    fed = report["federated_no_dp"]["pr_auc"]
    sweep = sorted(report["dp_sweep"], key=lambda r: r["target_epsilon"])
    eps = [r["target_epsilon"] for r in sweep]
    pr = [r["pr_auc"] for r in sweep]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2), gridspec_kw={"width_ratios": [3, 2]})

    ax1.axhline(cen, ls="--", color="#0d94fb", label=f"centralized ({cen:.3f})")
    ax1.axhline(fed, ls="--", color="#4de1f2", label=f"federated, no DP ({fed:.3f})")
    ax1.plot(eps, pr, "o-", color="#02042b", lw=2, label="federated + DP")
    ax1.set_xscale("log")
    ax1.set_xticks(eps)
    ax1.set_xticklabels([f"{e:g}" for e in eps])
    ax1.minorticks_off()
    ax1.set_xlabel("privacy budget ε (log scale, lower = more private)")
    ax1.set_ylabel("test PR-AUC")
    ax1.set_title("Utility vs. privacy budget")
    ax1.invert_xaxis()
    lo = min([cen, fed, *pr]) - 0.01
    ax1.set_ylim(lo, max(pr) + 0.01)
    ax1.grid(alpha=0.25)
    ax1.legend(fontsize=8, loc="center left")

    d = report["robustness_demo"]
    dd = d["defended"]
    bars = ["no defense\n(sign-flip)", "defended\nsign-flip", "defended\nflip", "defended\nlast-mover"]
    vals = [d["no_defense_plain_fedavg"]["pr_auc"], dd["sign_flip"]["pr_auc"],
            dd["flip"]["pr_auc"], dd["last_mover"]["pr_auc"]]
    ax2.bar(bars, vals, color=["#d23c58", "#0d94fb", "#0d94fb", "#0d94fb"])
    ax2.axhline(fed, ls="--", color="#4de1f2", label=f"honest federated ({fed:.3f})")
    ax2.set_ylabel("test PR-AUC")
    ax2.set_title("1/8 merchants malicious")
    for i, v in enumerate(vals):
        ax2.text(i, v + 0.005, f"{v:.3f}", ha="center", fontsize=8)
    ax2.legend(fontsize=8)

    fig.tight_layout()
    FIG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG, dpi=130)
    print(f"Wrote {FIG}")


def _sanity(report: dict) -> None:
    cen = report["centralized"]["pr_auc"]
    fed = report["federated_no_dp"]["pr_auc"]
    checks = []
    checks.append(("centralized and federated (no DP) within 0.1 PR-AUC",
                   abs(cen - fed) < 0.1))
    prev = None
    for r in report["dp_sweep"]:
        checks.append((f"spent ε ≤ target ε ({r['target_epsilon']})",
                       r["spent_epsilon_max"] <= r["target_epsilon"] + 1e-6))
        if prev is not None:
            checks.append((f"PR-AUC not increasing as ε tightens ({prev[0]}→{r['target_epsilon']})",
                           r["pr_auc"] <= prev[1] + 0.03))
        prev = (r["target_epsilon"], r["pr_auc"])
    d = report["robustness_demo"]
    nd, dd = d["no_defense_plain_fedavg"], d["defended"]
    checks.append(("defended sign-flip beats plain FedAvg",
                   dd["sign_flip"]["pr_auc"] > nd["pr_auc"] + 0.02))
    checks.append(("norm filter fires on sign-flip",
                   "norm_outlier" in dd["sign_flip"]["rejection_reasons"]))
    checks.append(("cosine filter fires on flip",
                   "cosine_outlier" in dd["flip"]["rejection_reasons"]))
    checks.append(("commit/reveal fires on last-mover",
                   "commitment_mismatch" in dd["last_mover"]["rejection_reasons"]))
    for k in ("sign_flip", "flip", "last_mover"):
        checks.append((f"defended/{k} within 0.05 PR-AUC of honest federated",
                       abs(dd[k]["pr_auc"] - fed) < 0.05))
    print("\nsanity:")
    for name, ok in checks:
        print(f"  [{'ok ' if ok else 'MISS'}] {name}")


if __name__ == "__main__":
    main()
