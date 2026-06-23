"""DemoGuard — Phase 2: Isolation Forest quality scorer over trajectory features.

Standardizes the per-episode features and fits an unsupervised Isolation Forest.
Each episode gets an anomaly score mapped to a quality score in [0, 1], where
1 = looks like a normal demo, 0 = strong statistical outlier.

CPU-only. Run from repo root (after trajectory_features.py has produced the CSV,
or it will rebuild the table itself):
    python -m src.scoring.isoforest
"""
from __future__ import annotations

import os
from typing import List, Optional

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

from src.features.trajectory_features import build_feature_table, FEATURE_COLS

FEATURES_CSV = "results/episode_features.csv"
SCORES_CSV = "results/episode_scores.csv"


def load_or_build_features() -> pd.DataFrame:
    if os.path.exists(FEATURES_CSV):
        return pd.read_csv(FEATURES_CSV)
    return build_feature_table()


def score_episodes(df: pd.DataFrame, feature_cols: Optional[List[str]] = None,
                   contamination: float = 0.1, random_state: int = 0) -> pd.DataFrame:
    """Fit Isolation Forest on standardized features; return df with scores.

    Adds two columns:
      iso_raw     - raw IsolationForest.score_samples (higher = more normal)
      quality     - iso_raw min-max normalized to [0, 1] (1 = good, 0 = anomalous)
    """
    feature_cols = feature_cols or FEATURE_COLS
    X = df[feature_cols].to_numpy(dtype=float)

    Xs = StandardScaler().fit_transform(X)
    iso = IsolationForest(
        n_estimators=300,
        contamination=contamination,
        random_state=random_state,
    )
    iso.fit(Xs)

    raw = iso.score_samples(Xs)  # higher = more normal
    lo, hi = raw.min(), raw.max()
    quality = (raw - lo) / (hi - lo) if hi > lo else np.ones_like(raw)

    out = df.copy()
    out["iso_raw"] = raw
    out["quality"] = quality
    out["flagged"] = iso.predict(Xs) == -1  # True = outlier
    return out.sort_values("quality").reset_index(drop=True)


if __name__ == "__main__":
    df = load_or_build_features()
    scored = score_episodes(df)

    n_flagged = int(scored["flagged"].sum())
    print(f"Scored {len(scored)} episodes; {n_flagged} flagged as outliers.\n")

    print("Bottom 10 (lowest quality = most anomalous):")
    cols = ["episode_index", "quality"] + FEATURE_COLS
    print(scored.head(10)[cols].round(3).to_string(index=False))

    scored_by_ep = scored.sort_values("episode_index").reset_index(drop=True)
    scored_by_ep.to_csv(SCORES_CSV, index=False)
    print(f"\nSaved -> {SCORES_CSV}")