"""Small, auditable LightGBM signal adapter used by the research backtest.

The production dependency is optional so the deterministic offline Demo still
works in a clean checkout.  When LightGBM is unavailable, the adapter uses a
documented rank based probability approximation instead of silently disabling
the model.
"""
from __future__ import annotations

from typing import Tuple

import numpy as np
import pandas as pd


class LightGBMSignal:
    def __init__(self, seed: int = 42):
        self.seed = seed
        self.backend = "rank-compatible-fallback"
        self.model = None

    def probabilities(self, frame: pd.DataFrame) -> Tuple[pd.DataFrame, str]:
        if frame.empty:
            return pd.DataFrame(index=frame.index), self.backend
        numeric = frame.select_dtypes(include=["number"]).copy()
        numeric = numeric.replace([np.inf, -np.inf], np.nan).fillna(numeric.median()).fillna(0)
        # Prefer LightGBM when installed.  The labels are a transparent
        # cross-sectional proxy for the frozen 20-day event in Demo mode.
        try:
            from lightgbm import LGBMClassifier

            x = numeric.to_numpy(dtype=float)
            rank = pd.Series(x.mean(axis=1)).rank(pct=True).to_numpy()
            y = np.where(rank >= .67, 2, np.where(rank <= .33, 0, 1))
            clf = LGBMClassifier(objective="multiclass", num_class=3, n_estimators=32,
                                 learning_rate=.05, max_depth=3, random_state=self.seed,
                                 verbosity=-1)
            clf.fit(x, y)
            proba = clf.predict_proba(x)
            classes = {int(c): i for i, c in enumerate(clf.classes_)}
            out = np.zeros((len(frame), 3), dtype=float)
            for label, col in classes.items():
                out[:, label] = proba[:, col]
            self.model = clf
            self.backend = "lightgbm"
        except Exception:
            score = pd.Series(numeric.mean(axis=1), index=frame.index).rank(pct=True).to_numpy()
            p_up = np.clip((score - .35) / .65, 0, 1)
            p_down = np.clip((.65 - score) / .65, 0, 1)
            p_neutral = np.clip(1 - np.abs(score - .5) * 2, 0, 1)
            out = np.column_stack([p_down, p_neutral, p_up])
            out = out / np.maximum(out.sum(axis=1, keepdims=True), 1e-12)
        return pd.DataFrame({"p_down": out[:, 0], "p_neutral": out[:, 1], "p_up": out[:, 2]}, index=frame.index), self.backend
