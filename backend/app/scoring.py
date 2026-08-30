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

    def score(self, df: pd.DataFrame) -> np.ndarray:
        """df: raw IEEE-CIS-shaped rows (replay). Returns P(fraud) in [0, 1]."""
        if not self.ok:
            raise RuntimeError("model artifacts missing — run `make baseline`")
        X = common.transform(df, self.spec)
        return self.booster.predict(X)

    @staticmethod
    def entity_keys(df: pd.DataFrame) -> pd.DataFrame:
        """Attach _uid / _card_id / _device_id / _emailaddr_id (replay rows)."""
        return _add_keys(df)


scorer = Scorer()
