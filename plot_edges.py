from __future__ import annotations

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt


def plot_edge_refinement(edge_obs: np.ndarray, edge_refined: np.ndarray, save_path: str | Path | None = None, show: bool = False) -> None:
    """Plot observed and refined multiview edge locations."""
    edge_obs = np.asarray(edge_obs, dtype=float)
    edge_refined = np.asarray(edge_refined, dtype=float)
    if edge_obs.shape != edge_refined.shape or edge_obs.shape[1] < 2:
        raise ValueError("edge_obs and edge_refined must both have shape (N, >=2).")
    plt.figure(figsize=(6, 6))
    plt.scatter(edge_obs[:, 0], edge_obs[:, 1], marker="x", label="observed")
    plt.scatter(edge_refined[:, 0], edge_refined[:, 1], marker="o", label="refined")
    for i in range(edge_obs.shape[0]):
        plt.plot([edge_obs[i, 0], edge_refined[i, 0]], [edge_obs[i, 1], edge_refined[i, 1]], linewidth=1)
        plt.text(edge_obs[i, 0], edge_obs[i, 1], f"v{i+1}")
    plt.axis("equal")
    plt.gca().invert_yaxis()
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    plt.tight_layout()
    if save_path is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=200)
    if show:
        plt.show()
    else:
        plt.close()
