
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import List

import numpy as np
import pandas as pd

from src.data.load import load_dataset, REPO_ID


# Feature columns produced per episode (order is stable).
FEATURE_COLS = [
    "length",
    "mean_jerk",
    "max_accel",
    "reversal_rate",
    "path_efficiency",
    "mean_speed",
]


@dataclass
class EpisodeFeatures:
    episode_index: int
    length: int
    mean_jerk: float
    max_accel: float
    reversal_rate: float
    path_efficiency: float
    mean_speed: float
    terminal_success: float  # from next.success if available, else NaN


def _safe_diff(x: np.ndarray) -> np.ndarray:
    """First difference along time; returns shape (T-1, D)."""
    return np.diff(x, axis=0) if x.shape[0] > 1 else np.zeros((1, x.shape[1]))


def compute_features(state: np.ndarray, action: np.ndarray,
                     terminal_success: float = float("nan")) -> dict:
    """Compute trajectory features from one episode's arrays.

    state:  (T, >=2) agent positions (we use first two dims as x, y)
    action: (T, >=2) commanded targets
    """
    pos = np.asarray(state, dtype=float)[:, :2]
    act = np.asarray(action, dtype=float)[:, :2]
    T = pos.shape[0]

    vel = _safe_diff(pos)                 # (T-1, 2) velocity
    accel = _safe_diff(vel)              # (T-2, 2) acceleration
    jerk = _safe_diff(accel)            # (T-3, 2) jerk

    speed = np.linalg.norm(vel, axis=1) if vel.size else np.array([0.0])
    accel_mag = np.linalg.norm(accel, axis=1) if accel.size else np.array([0.0])
    jerk_mag = np.linalg.norm(jerk, axis=1) if jerk.size else np.array([0.0])

    # Path efficiency: net displacement / total path length (1.0 = perfectly direct).
    path_len = float(speed.sum())
    net_disp = float(np.linalg.norm(pos[-1] - pos[0])) if T > 1 else 0.0
    path_efficiency = (net_disp / path_len) if path_len > 1e-8 else 0.0

    # Action reversals: how often the action delta flips direction (per axis), averaged.
    act_delta = _safe_diff(act)
    if act_delta.shape[0] > 1:
        sign_changes = np.sum(np.diff(np.sign(act_delta), axis=0) != 0, axis=0)
        reversal_rate = float(np.mean(sign_changes) / max(act_delta.shape[0], 1))
    else:
        reversal_rate = 0.0

    return {
        "length": int(T),
        "mean_jerk": float(jerk_mag.mean()),
        "max_accel": float(accel_mag.max()),
        "reversal_rate": reversal_rate,
        "path_efficiency": path_efficiency,
        "mean_speed": float(speed.mean()),
        "terminal_success": float(terminal_success),
    }


def build_feature_table(repo_id: str = REPO_ID) -> pd.DataFrame:
    """Extract features for every episode -> DataFrame (one row per episode).

    Reads observation.state, action, and next.success straight from the dataset's
    tabular data (no video decode), grouping frames by episode_index.
    """
    ds = load_dataset(repo_id)
    hf = ds.hf_dataset.with_format("numpy")

    ep_idx = np.asarray(hf["episode_index"])
    fr_idx = np.asarray(hf["frame_index"])
    states = np.asarray(hf["observation.state"])
    actions = np.asarray(hf["action"])
    has_success = "next.success" in hf.column_names
    succ = np.asarray(hf["next.success"]) if has_success else None

    rows: List[dict] = []
    for ep in np.unique(ep_idx):
        m = ep_idx == ep
        order = np.argsort(fr_idx[m])
        s = np.atleast_2d(states[m][order])
        a = np.atleast_2d(actions[m][order])
        # terminal success = whether the episode ever hit success (or last frame).
        ts = float(np.asarray(succ[m]).any()) if has_success else float("nan")
        feats = compute_features(s, a, terminal_success=ts)
        feats["episode_index"] = int(ep)
        rows.append(feats)

    df = pd.DataFrame(rows).sort_values("episode_index").reset_index(drop=True)
    cols = ["episode_index"] + FEATURE_COLS + ["terminal_success"]
    return df[cols]


if __name__ == "__main__":
    df = build_feature_table()
    print(f"Built feature table for {len(df)} episodes\n")
    print(df.describe().round(3))
    print("\nFirst few rows:")
    print(df.head().round(3).to_string(index=False))

    out = "results/episode_features.csv"
    df.to_csv(out, index=False)
    print(f"\nSaved -> {out}")