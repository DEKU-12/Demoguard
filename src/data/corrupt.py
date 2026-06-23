"""DemoGuard — Phase 4: inject known corruption and prove the detector.

One pure corruption function operates on (state, action) arrays. It is used:
  (a) in-memory now, to build a labeled feature table and measure the
      Isolation Forest's precision/recall against KNOWN corrupted episodes;
  (b) later, to write a corrupted dataset to disk for the training comparison.

CPU-only. Run from repo root:
    python -m src.data.corrupt
"""
from __future__ import annotations

from typing import Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import (precision_score, recall_score, f1_score,
                             roc_auc_score, average_precision_score)

from src.data.load import load_dataset, REPO_ID
from src.features.trajectory_features import compute_features, FEATURE_COLS
from src.scoring.isoforest import score_episodes

CORRUPT_KINDS = ("jitter", "spike", "reverse")


def corrupt_episode(state: np.ndarray, action: np.ndarray, rng: np.random.Generator,
                    kinds: Sequence[str] = CORRUPT_KINDS,
                    strength: float = 1.0) -> Tuple[np.ndarray, np.ndarray]:
    """Return a corrupted copy of (state, action).

    Corruptions are applied consistently to both the agent path (state) and the
    commanded action, so the damage shows up in the position-derived features
    (jerk, accel) and the action-derived features (reversal_rate) alike.
    """
    s = np.asarray(state, dtype=float).copy()
    a = np.asarray(action, dtype=float).copy()
    T = s.shape[0]
    s_std = s.std(axis=0) + 1e-6
    a_std = a.std(axis=0) + 1e-6

    if "jitter" in kinds:
        s += rng.normal(0, 1, s.shape) * s_std * 0.8 * strength
        a += rng.normal(0, 1, a.shape) * a_std * 0.8 * strength

    if "spike" in kinds and T > 4:
        n = max(1, T // 15)
        idx = rng.choice(T, size=n, replace=False)
        s[idx] += rng.normal(0, 1, (n, s.shape[1])) * s_std * 4.0 * strength
        a[idx] += rng.normal(0, 1, (n, a.shape[1])) * a_std * 4.0 * strength

    if "reverse" in kinds and T > 8:
        L = int(rng.integers(T // 5, T // 2))
        start = int(rng.integers(0, T - L))
        s[start:start + L] = s[start:start + L][::-1]
        a[start:start + L] = a[start:start + L][::-1]

    return s, a


def build_corrupted_feature_table(repo_id: str = REPO_ID, corrupt_fraction: float = 0.25,
                                  seed: int = 0, strength: float = 1.0) -> pd.DataFrame:
    """Load clean episodes, corrupt a random subset, return labeled features.

    Adds column `is_corrupted` (the ground-truth label) to the feature table.
    """
    ds = load_dataset(repo_id)
    hf = ds.hf_dataset.with_format("numpy")
    ep_idx = np.asarray(hf["episode_index"])
    fr_idx = np.asarray(hf["frame_index"])
    states = np.asarray(hf["observation.state"])
    actions = np.asarray(hf["action"])

    episodes = np.unique(ep_idx)
    rng = np.random.default_rng(seed)
    n_corrupt = int(round(len(episodes) * corrupt_fraction))
    corrupt_set = set(rng.choice(episodes, size=n_corrupt, replace=False).tolist())

    rows = []
    for ep in episodes:
        m = ep_idx == ep
        order = np.argsort(fr_idx[m])
        s = np.atleast_2d(states[m][order])
        a = np.atleast_2d(actions[m][order])
        is_corrupt = int(ep) in corrupt_set
        if is_corrupt:
            s, a = corrupt_episode(s, a, rng, strength=strength)
        feats = compute_features(s, a)
        feats["episode_index"] = int(ep)
        feats["is_corrupted"] = int(is_corrupt)
        rows.append(feats)

    df = pd.DataFrame(rows).sort_values("episode_index").reset_index(drop=True)
    return df


def evaluate_detector(df: pd.DataFrame, contamination: float = 0.25,
                      seed: int = 0) -> dict:
    """Score with Isolation Forest and measure precision/recall vs ground truth."""
    scored = score_episodes(df, contamination=contamination, random_state=seed)
    scored = scored.sort_values("episode_index").reset_index(drop=True)
    y_true = scored["is_corrupted"].to_numpy()
    y_pred = scored["flagged"].astype(int).to_numpy()
    anomaly_score = -scored["quality"].to_numpy()  # higher = more anomalous

    metrics = {
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, anomaly_score) if y_true.sum() else float("nan"),
        "avg_precision": average_precision_score(y_true, anomaly_score) if y_true.sum() else float("nan"),
        "n_corrupted": int(y_true.sum()),
        "n_flagged": int(y_pred.sum()),
        "n_total": len(y_true),
    }
    return metrics, scored


if __name__ == "__main__":
    FRACTION = 0.25
    df = build_corrupted_feature_table(corrupt_fraction=FRACTION, seed=0)
    n_corrupt = int(df["is_corrupted"].sum())
    print(f"Corrupted {n_corrupt}/{len(df)} episodes (fraction={FRACTION}).\n")

    metrics, scored = evaluate_detector(df, contamination=FRACTION, seed=0)
    print("Detector performance (Isolation Forest vs known labels):")
    for k in ("precision", "recall", "f1", "roc_auc", "avg_precision"):
        print(f"  {k:14s}: {metrics[k]:.3f}")
    print(f"  flagged {metrics['n_flagged']}, corrupted {metrics['n_corrupted']}, "
          f"total {metrics['n_total']}")

    out = "results/corrupted_scores.csv"
    scored.to_csv(out, index=False)
    print(f"\nSaved -> {out}")