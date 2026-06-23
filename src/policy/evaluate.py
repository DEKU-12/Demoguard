"""DemoGuard — evaluate a trained checkpoint in gym-pusht and log to MLflow."""
from __future__ import annotations

import argparse
import ast
import os
import subprocess
import sys
from pathlib import Path
from typing import Optional


def _extract_metrics(stdout: str) -> dict:
    """Pull the dict printed after 'Overall Aggregated Metrics:'."""
    marker = "Overall Aggregated Metrics:"
    i = stdout.find(marker)
    if i == -1:
        raise ValueError("Could not find aggregated metrics in eval output.")
    j = stdout.find("{", i)
    depth = 0
    for k in range(j, len(stdout)):
        if stdout[k] == "{":
            depth += 1
        elif stdout[k] == "}":
            depth -= 1
            if depth == 0:
                return ast.literal_eval(stdout[j:k + 1])
    raise ValueError("Unbalanced braces while parsing metrics.")


def evaluate_checkpoint(checkpoint: str, n_episodes: int = 50, batch_size: int = 10,
                        device: str = "mps", env_type: str = "pusht",
                        run_name: Optional[str] = None,
                        extra_params: Optional[dict] = None,
                        log_mlflow: bool = True) -> dict:
    cmd = [
        "lerobot-eval",
        f"--policy.path={checkpoint}",
        f"--env.type={env_type}",
        f"--eval.n_episodes={n_episodes}",
        f"--eval.batch_size={batch_size}",
        f"--policy.device={device}",
    ]
    env = dict(os.environ, PYTORCH_ENABLE_MPS_FALLBACK="1")
    print("Running:", " ".join(cmd))
    proc = subprocess.run(cmd, env=env, text=True, capture_output=True)
    if proc.returncode != 0:
        sys.stderr.write(proc.stdout[-3000:] + "\n" + proc.stderr[-3000:])
        raise RuntimeError(f"lerobot-eval failed (exit {proc.returncode}).")

    metrics = _extract_metrics(proc.stdout)
    success = float(metrics.get("pc_success", float("nan")))
    print(f"\npc_success = {success}%  over {metrics.get('n_episodes')} episodes "
          f"(avg_max_reward = {metrics.get('avg_max_reward'):.3f})")

    if log_mlflow:
        import mlflow
        Path("results").mkdir(exist_ok=True)
        uri = "sqlite:///" + str((Path("results") / "mlflow.db").resolve())
        mlflow.set_tracking_uri(uri)
        mlflow.set_experiment("demoguard-baseline")
        parts = Path(checkpoint).parts
        default_run = (parts[parts.index("checkpoints") - 1]
                       if "checkpoints" in parts else Path(checkpoint).name)
        with mlflow.start_run(run_name=run_name or default_run):
            mlflow.log_params({
                "checkpoint": checkpoint,
                "n_episodes": n_episodes,
                "device": device,
                "env_type": env_type,
                **(extra_params or {}),
            })
            mlflow.log_metrics({
                "pc_success": success,
                "avg_max_reward": float(metrics.get("avg_max_reward", float("nan"))),
                "avg_sum_reward": float(metrics.get("avg_sum_reward", float("nan"))),
            })
        print(f"Logged to MLflow at {uri} (experiment: demoguard-baseline)")
    return metrics


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--n-episodes", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=10)
    p.add_argument("--device", default="mps")
    p.add_argument("--run-name", default=None)
    p.add_argument("--no-mlflow", action="store_true")
    a = p.parse_args()
    evaluate_checkpoint(a.checkpoint, a.n_episodes, a.batch_size, a.device,
                        run_name=a.run_name, log_mlflow=not a.no_mlflow)