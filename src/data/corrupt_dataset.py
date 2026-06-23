"""DemoGuard — Phase 4b: write a 'polluted' copy of the PushT dataset.

Copies the cached LeRobotDataset to a fresh, writable folder, then rewrites
ONLY the `observation.state` and `action` columns for a chosen set of episodes
using the same corruption as src/data/corrupt.py. Videos and metadata untouched,
episode lengths preserved, so all dataset indices stay valid.

Run from repo root:
    python -m src.data.corrupt_dataset
"""
from __future__ import annotations

import glob
import json
import os
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.load import load_dataset, REPO_ID
from src.data.corrupt import corrupt_episode

OUT_ROOT = Path("outputs/datasets/pusht_polluted")
CORRUPT_IDS_JSON = "results/corrupted_episode_ids.json"


def _resolve_parquets(root: Path):
    return sorted(glob.glob(str(root / "data" / "**" / "*.parquet"), recursive=True))


def write_polluted_dataset(repo_id: str = REPO_ID, corrupt_fraction: float = 0.25,
                           seed: int = 0, strength: float = 1.0,
                           out_root: Path = OUT_ROOT) -> dict:
    ds = load_dataset(repo_id)
    src_root = Path(ds.root)

    # 1) Fresh writable copy (resolve symlinks so we never touch the HF blob store).
    if out_root.exists():
        shutil.rmtree(out_root)
    out_root.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(src_root, out_root, symlinks=False)
    print(f"Copied dataset -> {out_root}")

    # 2) Choose which episodes to corrupt (same RNG scheme as corrupt.py).
    parquets = _resolve_parquets(out_root)
    if not parquets:
        raise FileNotFoundError(f"No parquet files under {out_root}/data")
    all_eps = set()
    for pqf in parquets:
        all_eps.update(pd.read_parquet(pqf, columns=["episode_index"])
                       ["episode_index"].unique().tolist())
    episodes = np.array(sorted(int(e) for e in all_eps))
    rng = np.random.default_rng(seed)
    n_corrupt = int(round(len(episodes) * corrupt_fraction))
    corrupt_set = set(int(e) for e in rng.choice(episodes, size=n_corrupt, replace=False))
    print(f"Corrupting {len(corrupt_set)}/{len(episodes)} episodes.")

    # 3) Rewrite only state/action for corrupted episodes, per parquet file.
    for pqf in parquets:
        df = pd.read_parquet(pqf)

        # Work on object-dtype copies so each cell holds one length-2 array.
        state_col = df["observation.state"].to_list()
        action_col = df["action"].to_list()
        fr_idx_all = df["frame_index"].to_numpy()
        ep_idx_all = df["episode_index"].to_numpy()

        changed = False
        for ep in sorted(set(ep_idx_all.tolist()) & corrupt_set):
            rows = np.where(ep_idx_all == ep)[0]
            order = np.argsort(fr_idx_all[rows])
            rows_sorted = rows[order]

            s = np.stack([np.asarray(state_col[r], dtype=float) for r in rows_sorted])
            a = np.stack([np.asarray(action_col[r], dtype=float) for r in rows_sorted])

            s_c, a_c = corrupt_episode(s, a, np.random.default_rng(seed + int(ep)),
                                       strength=strength)

            for k, r in enumerate(rows_sorted):
                state_col[r] = s_c[k].astype(np.float32)
                action_col[r] = a_c[k].astype(np.float32)
            changed = True

        if changed:
            df["observation.state"] = pd.Series(state_col, index=df.index, dtype=object)
            df["action"] = pd.Series(action_col, index=df.index, dtype=object)
            df.to_parquet(pqf, index=False)
            print(f"  rewrote {os.path.relpath(pqf, out_root)}")

    # 4) Save ground-truth ids.
    os.makedirs("results", exist_ok=True)
    payload = {
        "repo_id": repo_id,
        "corrupt_fraction": corrupt_fraction,
        "seed": seed,
        "strength": strength,
        "out_root": str(out_root),
        "corrupted_episodes": sorted(corrupt_set),
        "all_episodes": [int(e) for e in episodes],
    }
    with open(CORRUPT_IDS_JSON, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\nSaved ground truth -> {CORRUPT_IDS_JSON}")
    return payload


if __name__ == "__main__":
    info = write_polluted_dataset()
    print(f"\nPolluted dataset ready at: {info['out_root']}")
    print(f"Corrupted {len(info['corrupted_episodes'])} episodes; "
          f"ids saved for the filtered-training step.")