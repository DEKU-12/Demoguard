
from __future__ import annotations
import os
from typing import List, Tuple
import numpy as np
import matplotlib
matplotlib.use("Agg")  # writes a PNG without needing a display; fine for scripts
import matplotlib.pyplot as plt

REPO_ID = "lerobot/pusht"
PLOTS_DIR = os.path.join("results", "plots")


def load_dataset(repo_id: str = REPO_ID, episodes: List[int] | None = None):
    """Load a LeRobotDataset from the Hub (cached locally after first call)."""
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    if episodes is not None:
        return LeRobotDataset(repo_id, episodes=episodes)
    return LeRobotDataset(repo_id)


def dataset_summary(dataset) -> dict:
    return {
        "num_episodes": int(dataset.num_episodes),
        "num_frames": int(len(dataset)),
        "fps": float(getattr(dataset, "fps", float("nan"))),
        "features": list(dataset.meta.features.keys()),
    }


def episode_trajectory(repo_id: str, ep: int,
                       state_key: str = "observation.state",
                       action_key: str = "action") -> Tuple[np.ndarray, np.ndarray]:
    """Return (state, action) arrays of shape (T, dim) for one episode."""
    ds = load_dataset(repo_id, episodes=[ep])
    states, actions = [], []
    for i in range(len(ds)):
        f = ds[i]
        states.append(np.asarray(f[state_key]).reshape(-1))
        actions.append(np.asarray(f[action_key]).reshape(-1))
    return np.stack(states), np.stack(actions)


def plot_episode(state: np.ndarray, action: np.ndarray, ep: int,
                 out_dir: str = PLOTS_DIR) -> str:
    """Plot agent path (first two state dims) + actions over time. Saves a PNG."""
    os.makedirs(out_dir, exist_ok=True)
    agent_xy = state[:, :2]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))

    ax1.plot(agent_xy[:, 0], agent_xy[:, 1], "-", lw=1.5, alpha=0.8)
    ax1.scatter(*agent_xy[0], c="green", s=60, zorder=5, label="start")
    ax1.scatter(*agent_xy[-1], c="red", s=60, zorder=5, label="end")
    ax1.set_title(f"Episode {ep} — agent path")
    ax1.set_xlabel("x"); ax1.set_ylabel("y")
    ax1.legend(); ax1.set_aspect("equal", adjustable="datalim")

    t = np.arange(action.shape[0])
    for d in range(min(action.shape[1], 2)):
        ax2.plot(t, action[:, d], lw=1.2, label=f"action[{d}]")
    ax2.set_title(f"Episode {ep} — actions over time")
    ax2.set_xlabel("frame"); ax2.set_ylabel("action value")
    ax2.legend()

    fig.tight_layout()
    out_path = os.path.join(out_dir, f"episode_{ep:03d}.png")
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return out_path


if __name__ == "__main__":
    EP = 0
    print(f"Loading {REPO_ID} (first call downloads + caches; pusht is small)...")
    ds = load_dataset(REPO_ID)
    print("Summary:", dataset_summary(ds))

    print(f"\nExtracting trajectory for episode {EP}...")
    state, action = episode_trajectory(REPO_ID, EP)
    print(f"  state  shape: {state.shape}")
    print(f"  action shape: {action.shape}")

    path = plot_episode(state, action, EP)
    print(f"\nSaved plot -> {path}")
    print("Phase 0 done-when: open that PNG and you should see the agent's path.")