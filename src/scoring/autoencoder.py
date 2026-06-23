
from __future__ import annotations

from typing import Tuple

import numpy as np
import torch
import torch.nn as nn

RESAMPLE_T = 50          # fixed timesteps per episode after resampling
N_CHANNELS = 4           # state(x,y) + action(x,y)
INPUT_DIM = RESAMPLE_T * N_CHANNELS


def resample_episode(state: np.ndarray, action: np.ndarray, T: int = RESAMPLE_T) -> np.ndarray:
    """Resample (state[:, :2], action[:, :2]) to T timesteps via linear interp.

    Returns a flat vector of length T*4, per-channel z-normalized so the AE
    learns trajectory SHAPE rather than absolute position/scale.
    """
    s = np.asarray(state, dtype=float)[:, :2]
    a = np.asarray(action, dtype=float)[:, :2]
    n = s.shape[0]
    xp = np.linspace(0.0, 1.0, n)
    xq = np.linspace(0.0, 1.0, T)
    chans = []
    for arr in (s, a):
        for d in range(2):
            r = np.interp(xq, xp, arr[:, d])
            r = (r - r.mean()) / (r.std() + 1e-6)  # shape, not scale
            chans.append(r)
    return np.concatenate(chans).astype(np.float32)  # (T*4,)


class TrajAutoencoder(nn.Module):
    def __init__(self, input_dim: int = INPUT_DIM, latent: int = 16):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128), nn.ReLU(),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, latent),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent, 64), nn.ReLU(),
            nn.Linear(64, 128), nn.ReLU(),
            nn.Linear(128, input_dim),
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))


def train_autoencoder(X: np.ndarray, epochs: int = 300, lr: float = 1e-3,
                      seed: int = 0, verbose: bool = False) -> Tuple[TrajAutoencoder, np.ndarray]:
    """Train AE on X (n, INPUT_DIM); return (model, per-row reconstruction error)."""
    torch.manual_seed(seed)
    model = TrajAutoencoder(input_dim=X.shape[1])
    Xt = torch.from_numpy(X.astype(np.float32))
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    model.train()
    for ep in range(epochs):
        opt.zero_grad()
        out = model(Xt)
        loss = loss_fn(out, Xt)
        loss.backward()
        opt.step()
        if verbose and (ep + 1) % 50 == 0:
            print(f"  epoch {ep+1}/{epochs}  loss {loss.item():.4f}")
    model.eval()
    with torch.no_grad():
        recon = model(Xt)
        err = ((recon - Xt) ** 2).mean(dim=1).numpy()  # per-episode recon error
    return model, err


def ae_anomaly_scores(states_actions, epochs: int = 300, seed: int = 0) -> np.ndarray:
    """Convenience: list of (state, action) -> normalized AE anomaly score [0,1]."""
    X = np.stack([resample_episode(s, a) for (s, a) in states_actions])
    _, err = train_autoencoder(X, epochs=epochs, seed=seed)
    lo, hi = err.min(), err.max()
    return (err - lo) / (hi - lo) if hi > lo else np.zeros_like(err)


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    def smooth(T):
        steps = np.cumsum(rng.normal(0, 1, (T, 2)), axis=0)
        k = np.ones(5) / 5
        pos = np.vstack([np.convolve(steps[:, d], k, mode="same") for d in range(2)]).T
        return pos, pos + rng.normal(0, 0.05, (T, 2))
    data = [smooth(int(rng.integers(80, 200))) for _ in range(100)]
    scores = ae_anomaly_scores(data, epochs=100)
    print(f"AE scores: mean {scores.mean():.3f}, min {scores.min():.3f}, max {scores.max():.3f}")
    print("autoencoder.py OK")