"""DemoGuard — compute the FILTERED episode list: polluted minus detector-flagged.

Scientifically honest filtering: we drop the episodes the Isolation Forest flags
on the POLLUTED dataset (not the known ground-truth corrupted ids). The kept list
is written for the filtered training run, and we also report how well the
detector's flags matched the true corruption (for the writeup).

Run from repo root:
    python -m src.experiment.make_filtered_list
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.load import load_dataset, REPO_ID
from src.features.trajectory_features import compute_features
from src.scoring.isoforest import score_episodes

POLLUTED_ROOT = "outputs/datasets/pusht_polluted"
GROUND_TRUTH = "results/corrupted_episode_ids.json"
KEPT_JSON = "results/filtered_keep_episodes.json"
CONTAMINATION = 0.25


def feature_table_from_root(root: str) -> pd.DataFrame:
    ds = load_dataset(REPO_ID, ) if root is None else None
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    ds = LeRobotDataset(REPO_ID, root=root)
    hf = ds.hf_dataset.with_format("numpy")
    ep_idx = np.asarray(hf["episode_index"])
    fr_idx = np.asarray(hf["frame_index"])
    states = np.asarray(hf["observation.state"])
    actions = np.asarray(hf["action"])
    rows = []
    for ep in np.unique(ep_idx):
        m = ep_idx == ep
        order = np.argsort(fr_idx[m])
        s = np.atleast_2d(states[m][order])
        a = np.atleast_2d(actions[m][order])
        feats = compute_features(s, a)
        feats["episode_index"] = int(ep)
        rows.append(feats)
    return pd.DataFrame(rows).sort_values("episode_index").reset_index(drop=True)


if __name__ == "__main__":
    # 1) score the POLLUTED dataset with the detector
    df = feature_table_from_root(POLLUTED_ROOT)
    scored = score_episodes(df, contamination=CONTAMINATION, random_state=0)
    scored = scored.sort_values("episode_index").reset_index(drop=True)

    flagged = set(scored.loc[scored["flagged"], "episode_index"].astype(int).tolist())
    all_eps = scored["episode_index"].astype(int).tolist()
    kept = [e for e in all_eps if e not in flagged]

    # 2) compare flags vs the true corruption (for reporting only)
    gt = set(json.load(open(GROUND_TRUTH))["corrupted_episodes"])
    tp = len(flagged & gt); fp = len(flagged - gt); fn = len(gt - flagged)
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0

    Path("results").mkdir(exist_ok=True)
    json.dump({"keep_episodes": kept, "dropped_episodes": sorted(flagged),
               "n_total": len(all_eps), "n_dropped": len(flagged)},
              open(KEPT_JSON, "w"), indent=2)

    print(f"Detector flagged {len(flagged)} episodes; keeping {len(kept)}.")
    print(f"Flagged-vs-true: precision {prec:.3f}, recall {rec:.3f} "
          f"(TP={tp}, FP={fp}, FN={fn})")
    print(f"Saved keep-list -> {KEPT_JSON}")