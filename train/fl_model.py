"""
Phase 4 — the model under test: a small MLP trained with focal loss.

Deliberately NOT LightGBM. DP-SGD and FedAvg both need a differentiable model
with averageable parameters; gradient-boosted trees have neither. This MLP is
weaker than the graded Phase-1 baseline (PR-AUC 0.546) by construction — the
Phase-4 comparison is *federated vs centralized* and *DP vs no-DP* within this
same MLP, never MLP-vs-LightGBM.

Opacus forbids BatchNorm, so the net is Linear -> ReLU -> Dropout only.
"""
from __future__ import annotations

import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

SEED = 42
DEVICE = torch.device("cpu")  # no CUDA on the build box; Opacus + MPS is unreliable


def seed_everything(seed: int = SEED) -> None:
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


class MLP(nn.Module):
    # 128->32: wide enough for this task, small enough that DP-SGD's per-sample
    # gradients stay cheap and the noise isn't spread over too many parameters
    # (DPMLBench: parameter-dimensionality reduction helps DP utility).
    # tanh, not ReLU: measurably better under low-epsilon DP-SGD (DPMLBench).
    # No norm layer: BatchNorm breaks Opacus (mixes samples in a batch), and
    # tanh + GroupNorm together hurts at epsilon <= 10.
    def __init__(self, n_feat: int, p_drop: float = 0.25):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_feat, 128), nn.Tanh(), nn.Dropout(p_drop),
            nn.Linear(128, 32), nn.Tanh(), nn.Dropout(p_drop),
            nn.Linear(32, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


class FocalLoss(nn.Module):
    """Binary focal loss (Lin et al. 2017). alpha weights the positive (fraud) class."""

    def __init__(self, alpha: float = 0.75, gamma: float = 2.0):
        super().__init__()
        self.alpha, self.gamma = alpha, gamma

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        p = torch.sigmoid(logits)
        ce = nn.functional.binary_cross_entropy_with_logits(logits, target, reduction="none")
        p_t = p * target + (1 - p) * (1 - target)
        alpha_t = self.alpha * target + (1 - self.alpha) * (1 - target)
        return (alpha_t * (1 - p_t).pow(self.gamma) * ce).mean()


# --------------------------------------------------------------------------- #
# weight <-> ndarray plumbing (Flower's parameter format)
# --------------------------------------------------------------------------- #
def get_weights(model: nn.Module) -> list[np.ndarray]:
    return [v.detach().cpu().numpy().copy() for v in model.state_dict().values()]


def set_weights(model: nn.Module, weights: list[np.ndarray]) -> None:
    sd = model.state_dict()
    for k, w in zip(sd.keys(), weights):
        sd[k] = torch.tensor(np.asarray(w), dtype=sd[k].dtype)
    model.load_state_dict(sd, strict=True)


def new_model(n_feat: int, seed: int = SEED) -> MLP:
    seed_everything(seed)
    return MLP(n_feat).to(DEVICE)


# --------------------------------------------------------------------------- #
# data / train / eval
# --------------------------------------------------------------------------- #
def make_loader(X: np.ndarray, y: np.ndarray, batch_size: int,
                *, shuffle: bool = True, seed: int = SEED) -> DataLoader:
    ds = TensorDataset(torch.from_numpy(np.ascontiguousarray(X, dtype=np.float32)),
                       torch.from_numpy(np.ascontiguousarray(y, dtype=np.float32)))
    g = torch.Generator().manual_seed(seed)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, generator=g, drop_last=False)


def train_local(model: nn.Module, loader: DataLoader, optimizer, loss_fn: nn.Module,
                max_steps: int | None = None) -> float:
    """One local pass (or `max_steps` gradient steps, whichever is shorter)."""
    model.train()
    total, n, step = 0.0, 0, 0
    for xb, yb in loader:
        xb, yb = xb.to(DEVICE), yb.to(DEVICE)
        optimizer.zero_grad(set_to_none=True)
        loss = loss_fn(model(xb), yb)
        loss.backward()
        optimizer.step()
        total += loss.detach().item() * len(xb)
        n += len(xb)
        step += 1
        if max_steps is not None and step >= max_steps:
            break
    return total / max(n, 1)


@torch.no_grad()
def predict_proba(model: nn.Module, X: np.ndarray, batch_size: int = 16384) -> np.ndarray:
    model.eval()
    out = []
    for i in range(0, len(X), batch_size):
        xb = torch.from_numpy(np.ascontiguousarray(X[i:i + batch_size], dtype=np.float32)).to(DEVICE)
        out.append(torch.sigmoid(model(xb)).cpu().numpy())
    return np.concatenate(out) if out else np.zeros(0, dtype=np.float32)


def evaluate(model: nn.Module, X: np.ndarray, y: np.ndarray) -> dict:
    p = predict_proba(model, X)
    p = np.nan_to_num(p, nan=0.0, posinf=1.0, neginf=0.0)  # a diverged poisoned model can emit nan
    return {
        "pr_auc": float(average_precision_score(y, p)),
        "roc_auc": float(roc_auc_score(y, p)),
    }


def evaluate_weights(weights: list[np.ndarray], n_feat: int,
                     X: np.ndarray, y: np.ndarray) -> dict:
    m = MLP(n_feat).to(DEVICE)
    set_weights(m, weights)
    return evaluate(m, X, y)


def centralized_train(X: np.ndarray, y: np.ndarray, Xval: np.ndarray, yval: np.ndarray,
                      *, epochs: int = 20, lr: float = 1e-3, batch_size: int = 4096,
                      seed: int = SEED, patience: int = 4, verbose: bool = True) -> tuple:
    """Plain centralized MLP — the ceiling the federated runs are measured against."""
    seed_everything(seed)
    n_feat = X.shape[1]
    model = MLP(n_feat).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = FocalLoss()
    loader = make_loader(X, y, batch_size, shuffle=True, seed=seed)

    best_prauc, best_w, bad = -1.0, get_weights(model), 0
    for ep in range(1, epochs + 1):
        tr_loss = train_local(model, loader, opt, loss_fn)
        val = evaluate(model, Xval, yval)
        if verbose:
            print(f"  [central] epoch {ep:2d}  loss {tr_loss:.4f}  "
                  f"val PR-AUC {val['pr_auc']:.4f}  ROC-AUC {val['roc_auc']:.4f}")
        if val["pr_auc"] > best_prauc + 1e-4:
            best_prauc, best_w, bad = val["pr_auc"], get_weights(model), 0
        else:
            bad += 1
            if bad >= patience:
                if verbose:
                    print(f"  [central] early stop at epoch {ep}")
                break
    return best_w, {"val_pr_auc": best_prauc, "epochs_run": ep}
