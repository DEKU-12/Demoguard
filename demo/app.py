"""DemoGuard — interactive demo (Streamlit).

Browse the PushT demonstrations, see each one's quality scores (Isolation
Forest / autoencoder / composite), whether DemoGuard flagged it, and how it
compares to the known corruption labels. Plus a summary of the detector AUCs
and the downstream filtered-vs-polluted policy comparison.

Run from repo root (env active):
    streamlit run demo/app.py
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.metrics import roc_auc_score, average_precision_score

SCORES_CSV = ROOT / "results" / "composite_scores.csv"
FLAG_FRACTION = 0.25  # flag the worst 25% by composite score (matches the experiment)

# Downstream policy results (avg_max_reward over 50 eval episodes, 20k-step ACT).
# Edit here if you re-run with different numbers / more seeds.
DOWNSTREAM = {
    "clean (206 eps)": 0.344,
    "polluted (206, 52 corrupted)": 0.333,
    "filtered (154, detector-cleaned)": 0.362,
}

SCORE_COLS = {
    "iso_anomaly": "Isolation Forest",
    "ae_anomaly": "Autoencoder",
    "composite": "Composite",
}


@st.cache_data
def load_scores() -> pd.DataFrame:
    df = pd.read_csv(SCORES_CSV).sort_values("episode_index").reset_index(drop=True)
    if "flagged" not in df and "composite" in df:
        k = int(round(len(df) * FLAG_FRACTION))
        cutoff = df["composite"].nlargest(k).min() if k > 0 else float("inf")
        df["flagged"] = df["composite"] >= cutoff
    return df


@st.cache_resource
def load_trajectories():
    """Lazily load per-episode (state, action). Cached. None if unavailable."""
    try:
        from src.data.load import load_dataset, REPO_ID
        ds = load_dataset(REPO_ID)
        hf = ds.hf_dataset.with_format("numpy")
        ep_idx = np.asarray(hf["episode_index"])
        fr_idx = np.asarray(hf["frame_index"])
        states = np.asarray(hf["observation.state"])
        actions = np.asarray(hf["action"])
        out = {}
        for ep in np.unique(ep_idx):
            m = ep_idx == ep
            order = np.argsort(fr_idx[m])
            out[int(ep)] = (np.atleast_2d(states[m][order]),
                            np.atleast_2d(actions[m][order]))
        return out
    except Exception:
        return None


def compute_aucs(df: pd.DataFrame) -> pd.DataFrame:
    if "is_corrupted" not in df or df["is_corrupted"].sum() == 0:
        return pd.DataFrame()
    y = df["is_corrupted"].to_numpy()
    rows = []
    for col, label in SCORE_COLS.items():
        if col in df:
            rows.append({
                "Detector": label,
                "ROC-AUC": round(roc_auc_score(y, df[col]), 3),
                "Avg Precision": round(average_precision_score(y, df[col]), 3),
            })
    return pd.DataFrame(rows)


def plot_trajectory(state, action, ep, flagged, corrupted):
    fig, ax = plt.subplots(figsize=(5, 5))
    xy = np.asarray(state)[:, :2]
    ax.plot(xy[:, 0], xy[:, 1], "-", lw=1.5, alpha=0.8)
    ax.scatter(*xy[0], c="green", s=70, zorder=5, label="start")
    ax.scatter(*xy[-1], c="red", s=70, zorder=5, label="end")
    status = []
    if flagged:
        status.append("FLAGGED")
    if corrupted == 1:
        status.append("corrupted (truth)")
    title = f"Episode {ep}" + (f" — {', '.join(status)}" if status else "")
    ax.set_title(title)
    ax.set_xlabel("x"); ax.set_ylabel("y")
    ax.set_aspect("equal", adjustable="datalim"); ax.legend()
    return fig


def main():
    st.set_page_config(page_title="DemoGuard", layout="wide")
    st.title("DemoGuard — demonstration quality scoring")
    st.caption("Unsupervised detection of low-quality robot demonstrations in the "
               "LeRobot PushT dataset.")

    if not SCORES_CSV.exists():
        st.error(f"Missing {SCORES_CSV}. Run `python -m src.scoring.composite` first.")
        return

    df = load_scores()
    tab_summary, tab_explore = st.tabs(["Summary", "Episode explorer"])

    with tab_summary:
        st.subheader("Detector performance")
        st.caption("Measured against known injected corruption (ground truth).")
        aucs = compute_aucs(df)
        if not aucs.empty:
            st.dataframe(aucs, hide_index=True, use_container_width=True)
            best = aucs.loc[aucs["ROC-AUC"].idxmax()]
            st.success(f"Best detector: {best['Detector']} "
                       f"(ROC-AUC {best['ROC-AUC']}, AP {best['Avg Precision']})")

        st.subheader("Downstream policy comparison")
        st.caption("avg_max_reward in gym-pusht (higher is better). Single seed, "
                   "20k-step ACT — directional, not statistically tight.")
        ds = pd.DataFrame({"condition": list(DOWNSTREAM),
                           "avg_max_reward": list(DOWNSTREAM.values())})
        fig, ax = plt.subplots(figsize=(7, 3))
        colors = ["#888", "#c44", "#3a3"]
        ax.barh(ds["condition"], ds["avg_max_reward"], color=colors)
        ax.set_xlabel("avg_max_reward"); ax.invert_yaxis()
        for i, v in enumerate(ds["avg_max_reward"]):
            ax.text(v + 0.003, i, f"{v:.3f}", va="center")
        st.pyplot(fig)
        st.markdown("Filtering the detector-flagged demos **recovered and "
                    "slightly exceeded** the clean baseline, and beat the "
                    "polluted set.")

    with tab_explore:
        st.subheader("Browse episodes")
        c1, c2 = st.columns([1, 2])

        with c1:
            only_flagged = st.checkbox("Show only flagged", value=False)
            view = df[df["flagged"]] if (only_flagged and "flagged" in df) else df
            sort_by = st.selectbox("Sort by", ["composite", "ae_anomaly",
                                               "iso_anomaly", "episode_index"])
            view = view.sort_values(sort_by, ascending=(sort_by == "episode_index"))
            ep = st.selectbox("Episode", view["episode_index"].astype(int).tolist())

        row = df[df["episode_index"] == ep].iloc[0]
        flagged = bool(row.get("flagged", False))
        corrupted = int(row.get("is_corrupted", 0))

        with c2:
            m1, m2, m3 = st.columns(3)
            m1.metric("Isolation Forest", f"{row.get('iso_anomaly', float('nan')):.3f}")
            m2.metric("Autoencoder", f"{row.get('ae_anomaly', float('nan')):.3f}")
            m3.metric("Composite", f"{row.get('composite', float('nan')):.3f}")
            st.write(f"**Flagged by DemoGuard:** {'yes' if flagged else 'no'}  |  "
                     f"**Actually corrupted:** {'yes' if corrupted else 'no'}")

        trajs = load_trajectories()
        if trajs and ep in trajs:
            s, a = trajs[ep]
            st.pyplot(plot_trajectory(s, a, ep, flagged, corrupted))
        else:
            st.info("Trajectory preview needs the PushT dataset cached locally. "
                    "Scores above come from results/composite_scores.csv.")

        st.subheader("All episodes")
        show_cols = [c for c in ["episode_index", "iso_anomaly", "ae_anomaly",
                                 "composite", "flagged", "is_corrupted"] if c in df]
        st.dataframe(df[show_cols].round(3), hide_index=True, use_container_width=True,
                     height=300)


if __name__ == "__main__":
    main()