from __future__ import annotations

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt


def plot_curvature_heatmap(points: np.ndarray, curvature: np.ndarray, save_path: str | Path | None = None, show: bool = False) -> None:
    """Scatter-plot a polyline/edge set colored by curvature magnitude."""
    points = np.asarray(points, dtype=float)
    curvature = np.asarray(curvature, dtype=float).reshape(-1)
    if points.ndim != 2 or points.shape[1] != 2 or points.shape[0] != curvature.size:
        raise ValueError("points must have shape (N,2), curvature must have shape (N,).")
    plt.figure(figsize=(6, 5))
    sc = plt.scatter(points[:, 0], points[:, 1], c=np.abs(curvature), s=18)
    plt.colorbar(sc, label="|curvature|")
    plt.axis("equal")
    plt.gca().invert_yaxis()
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Curvature heatmap")
    plt.tight_layout()
    if save_path is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=200)
    if show:
        plt.show()
    else:
        plt.close()
