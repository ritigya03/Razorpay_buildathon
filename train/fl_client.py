"""
Phase 4 — one federated client = one synthetic merchant.

Holds the merchant's local slice (never leaves this object), runs a fixed number
of local gradient steps per round, and speaks the commit/reveal protocol that
`SentinelFedAvg` expects:

  compute_update(global_weights, round)  -> trains locally, caches new weights
  commit(round)                          -> SHA-256 digest of the *committed* weights
  reveal(round)                          -> the weights actually sent for aggregation

Differential privacy (when enabled) uses a single Opacus `PrivacyEngine` per
client, created once and reused so its accountant accumulates across rounds —
`spent_epsilon` is the client's cumulative budget after every round so far.

Attacks (all model-degradation, defence-only research):
  * "sign_flip"  — commit and reveal the SAME weights, global - 10x the honest
                   delta. Large norm -> caught by the strategy's norm test.
  * "flip"       — commit and reveal global - 1x the honest delta (direction
                   reversed, honest-looking magnitude) -> caught by the cosine
                   test, not the norm test.
  * "last_mover" — commit an honest-looking digest, then reveal DIFFERENT
                   (poisoned) weights -> caught by the commit/reveal hash check.
"""
from __future__ import annotations

import numpy as np
import torch

from fl_model import (
    FocalLoss, MLP, get_weights, make_loader, seed_everything, set_weights, train_local,
)
from fl_strategy import digest_ndarrays

SIGN_FLIP_SCALE = -10.0   # "sign_flip": reversed direction, blown-up magnitude
FLIP_SCALE = -1.0         # "flip": reversed direction, honest magnitude


class MerchantClient:
    # DP-SGD is run with plain SGD+momentum (Adam's second-moment estimates are
    # corrupted by the per-step Gaussian noise); non-DP local training uses Adam.
    # Higher LR pairs with the large DP batch (TAN scaling laws).
    DP_LR = 2.0
    DP_MOMENTUM = 0.9

    def __init__(self, cid: int, X: np.ndarray, y: np.ndarray, n_feat: int, *,
                 batch_size: int = 1024, lr: float = 1e-3, local_steps: int = 20,
                 dp_cfg: dict | None = None, attack: str | None = None, seed: int = 42):
        self.cid = cid
        self.X, self.y = X, y
        self.n_feat = n_feat
        self.n_local = len(y)
        self.fraud_count = int(y.sum())
        self.attack = attack
        self.malicious = attack is not None
        self.seed = seed + cid
        self.batch_size = min(batch_size, max(self.n_local, 1))
        self.lr = lr
        self.local_steps = local_steps
        self.dp_cfg = dp_cfg

        self.model = MLP(n_feat)
        self.loss_fn = FocalLoss()
        self._pe = None
        self._dp_opt = None
        self._dp_loader = None
        self._dp_model = None
        self._committed: list[np.ndarray] | None = None
        self._revealed: list[np.ndarray] | None = None
        self.spent_epsilon = 0.0

    # ------------------------------------------------------------------ #
    def _ensure_dp_engine(self) -> None:
        from opacus import PrivacyEngine

        seed_everything(self.seed)
        opt = torch.optim.SGD(self.model.parameters(), lr=self.DP_LR, momentum=self.DP_MOMENTUM)
        loader = make_loader(self.X, self.y, self.batch_size, shuffle=True, seed=self.seed)
        self._pe = PrivacyEngine(accountant="rdp")
        self._dp_model, self._dp_opt, self._dp_loader = self._pe.make_private(
            module=self.model, optimizer=opt, data_loader=loader,
            noise_multiplier=self.dp_cfg["noise_multiplier"],
            max_grad_norm=self.dp_cfg["max_grad_norm"],
            poisson_sampling=True,
        )

    def _local_fit(self, server_round: int) -> None:
        if self.dp_cfg is not None:
            if self._pe is None:
                self._ensure_dp_engine()
            train_local(self._dp_model, self._dp_loader, self._dp_opt, self.loss_fn,
                        max_steps=self.local_steps)
            self.spent_epsilon = float(self._pe.get_epsilon(self.dp_cfg["delta"]))
        else:
            seed_everything(self.seed + server_round)
            opt = torch.optim.Adam(self.model.parameters(), lr=self.lr)
            loader = make_loader(self.X, self.y, self.batch_size, shuffle=True,
                                 seed=self.seed + server_round)
            train_local(self.model, loader, opt, self.loss_fn, max_steps=self.local_steps)

    def compute_update(self, global_weights: list[np.ndarray], server_round: int) -> list[np.ndarray]:
        set_weights(self.model, global_weights)
        self._local_fit(server_round)
        honest = get_weights(self.model)

        def poison(scale: float) -> list[np.ndarray]:
            return [(g + scale * (h - g)).astype(np.float32)
                    for h, g in zip(honest, global_weights)]

        if self.attack == "sign_flip":
            self._committed = self._revealed = poison(SIGN_FLIP_SCALE)
        elif self.attack == "flip":
            self._committed = self._revealed = poison(FLIP_SCALE)
        elif self.attack == "last_mover":
            self._committed = honest                 # commit looks honest ...
            self._revealed = poison(SIGN_FLIP_SCALE)  # ... reveal is poisoned
        else:
            self._committed = self._revealed = honest
        return self._revealed

    def commit(self, server_round: int) -> str:
        assert self._committed is not None, "compute_update must run before commit"
        return digest_ndarrays(self._committed)

    def reveal(self, server_round: int) -> list[np.ndarray]:
        assert self._revealed is not None, "compute_update must run before reveal"
        return self._revealed
