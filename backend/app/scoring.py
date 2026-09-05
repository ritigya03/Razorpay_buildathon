"""Transaction scoring — wraps the Phase-1 LightGBM model + feature spec.

Reuses the exact training-time feature transform (train/common.py) so the model
sees inputs shaped identically to how it was trained.
"""
from __future__ import annotations

import sys

import lightgbm as lgb
import numpy as np
import pandas as pd

from .config import REPO_ROOT, settings

sys.path.insert(0, str(REPO_ROOT / "train"))
import common  # noqa: E402
from ring_features import _add_keys  # noqa: E402


class Scorer:
    def __init__(self) -> None:
        self.ok = settings.model_file.exists() and settings.feature_spec.exists()
        if self.ok:
            self.booster = lgb.Booster(model_file=str(settings.model_file))
            self.spec = common.load_spec(settings.feature_spec)
        else:
            self.booster = self.spec = None

    def score(self, df: pd.DataFrame, chunk_size: int = 10_000) -> np.ndarray:
        """df: raw IEEE-CIS-shaped rows (replay). Returns P(fraud) in [0, 1].

        Scored in chunks so a full-replay-sized call (~88k rows x ~430 raw
        columns) never materializes one huge transformed matrix at once —
        this is the difference between a transient ~150MB peak and a ~1.5GB
        one on the same data, which matters on a memory-capped deploy host.
        Single-row / small-batch callers (a live Razorpay payment) are
        unaffected: below chunk_size this is exactly the old single-shot path.
        """
        if not self.ok:
            raise RuntimeError("model artifacts missing — run `make baseline`")
        if len(df) <= chunk_size:
            X = common.transform(df, self.spec)
            return self.booster.predict(X)
        out = np.empty(len(df), dtype=np.float64)
        for start in range(0, len(df), chunk_size):
            chunk = df.iloc[start:start + chunk_size]
            out[start:start + len(chunk)] = self.booster.predict(common.transform(chunk, self.spec))
        return out

    @staticmethod
    def entity_keys(df: pd.DataFrame) -> pd.DataFrame:
        """Attach _uid / _card_id / _device_id / _emailaddr_id (replay rows)."""
        return _add_keys(df)


scorer = Scorer()
