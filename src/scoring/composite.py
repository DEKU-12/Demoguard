"""DemoGuard — Phase 3: composite scorer (Isolation Forest + autoencoder).

Builds a labeled corrupted feature table (reusing Phase 4's corruption), scores
it three ways — Isolation Forest alone, autoencoder alone, and a blend — and
reports ROC-AUC / average-precision for each against the known labels. This is
the before/after that shows whether the second detector helps.

CPU-only. Run from repo root:
    python -m src.scoring.composite
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score

from src.data.load import load_dataset, REPO_ID
from src.data.corrupt import corrupt_episode
from src.features.trajectory_features import compute_features
from src.scoring.isoforest import score_episodes
from src.scoring.autoencoder import resample_episode, train_autoencoder

CORRUPT_FRACTION = 0.25
SEED = 0
AE_EPOCHS = 400
BLEND = 0.5  # weight on autoencoder vs isolation forest


def _nrm(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    rng = np.ptp(x)
    return (x - x.min()) / (rng + 1e-12)


def build_labeled(repo_id: str = REPO_ID, corrupt_fraction: float = CORRUPT_FRACTION,
                  seed: int = SEED, strength: float = 1.0):
    """Return (feature_df with is_corrupted, list of (state,action) trajectories)."""
    ds = load_dataset(repo_id)
    hf = ds.hf_dataset.with_format("numpy")
    ep_idx = np.asarray(hf["episode_index"])
    fr_idx = np.asarray(hf["frame_index"])
    states = np.asarray(hf["observation.state"])
    actions = np.asarray(hf["action"])

    episodes = np.unique(ep_idx)
    rng = np.random.default_rng(seed)
    n_corrupt = int(round(len(episodes) * corrupt_fraction))
    corrupt_set = set(int(e) for e in rng.choice(episodes, size=n_corrupt, replace=False))

    feat_rows, trajs, labels = [], [], []
    for ep in episodes:
        m = ep_idx == ep
        order = np.argsort(fr_idx[m])
        s = np.atleast_2d(states[m][order])
        a = np.atleast_2d(actions[m][order])
        is_c = int(ep) in corrupt_set
        if is_c:
            s, a = corrupt_episode(s, a, np.random.default_rng(seed + int(ep)), strength=strength)
        f = compute_features(s, a); f["episode_index"] = int(ep); f["is_corrupted"] = int(is_c)
        feat_rows.append(f); trajs.append((s, a)); labels.append(int(is_c))

    df = pd.DataFrame(feat_rows).sort_values("episode_index").reset_index(drop=True)
    # keep trajs aligned to sorted df order
    order = np.argsort([int(e) for e in episodes])
    trajs = [trajs[i] for i in order]
    labels = np.array([labels[i] for i in order])
    return df, trajs, labels


if __name__ == "__main__":
    df, trajs, y = build_labeled()
    print(f"Corrupted {int(y.sum())}/{len(y)} episodes.\n")

    # Isolation Forest anomaly score
    scored = score_episodes(df, contamination=CORRUPT_FRACTION, random_state=SEED)
    scored = scored.sort_values("episode_index").reset_index(drop=True)
    iso_anom = _nrm(-scored["quality"].to_numpy())

    # Autoencoder anomaly score (reconstruction error)
    X = np.stack([resample_episode(s, a) for (s, a) in trajs])
    _, err = train_autoencoder(X, epochs=AE_EPOCHS, seed=SEED)
    ae_anom = _nrm(err)

    # Composite
    comp = BLEND * ae_anom + (1 - BLEND) * iso_anom

    print("Detector comparison (ROC-AUC / average-precision vs known labels):")
    for name, score in [("isolation_forest", iso_anom),
                        ("autoencoder", ae_anom),
                        (f"composite({BLEND:.1f})", comp)]:
        auc = roc_auc_score(y, score)
        ap = average_precision_score(y, score)
        print(f"  {name:18s}  AUC {auc:.3f}   AP {ap:.3f}")

    out = df.copy()
    out["iso_anomaly"] = iso_anom
    out["ae_anomaly"] = ae_anom
    out["composite"] = comp
    out.to_csv("results/composite_scores.csv", index=False)
    print("\nSaved -> results/composite_scores.csv")