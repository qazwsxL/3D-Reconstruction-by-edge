#!/usr/bin/env python3
"""
curvel_formation.py

implementation of curvel formation from edge anchor information.

Given an edge anchor:
    gamma = (x, y)
    theta = edge tangent orientation in radians
    kappa = curvature

Construct the local curvel using arc-length parameter s:

    theta(s) = kappa * s + theta

    C_x(s) = 1/kappa * sin(kappa*s + theta) + C0
    C_y(s) = -1/kappa * cos(kappa*s + theta) + C1

with C(0) = (x, y).

If kappa is close to 0, the curvel degenerates to a straight line:
    C(s) = (x, y) + s * (cos(theta), sin(theta))
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Tuple

import numpy as np
import matplotlib.pyplot as plt


EPS_KAPPA = 1e-8


@dataclass
class EdgeAnchor:
    """
    A single edge anchor used to construct a curvel.

    Attributes
    ----------
    x, y:
        Anchor edge location.
    theta:
        Tangent orientation in radians.
    kappa:
        Curvature. Positive/negative sign determines bending direction.
    """
    x: float
    y: float
    theta: float
    kappa: float


def curvel_point(edge: EdgeAnchor, s: float | np.ndarray) -> np.ndarray:
    """
    Evaluate one curvel point C(s), or many points if s is an array.

    Parameters
    ----------
    edge:
        EdgeAnchor containing x, y, theta, kappa.
    s:
        Arc-length parameter. Can be scalar or numpy array.

    Returns
    -------
    np.ndarray
        If s is scalar: shape (2,)
        If s is array: shape (len(s), 2)
    """
    x, y, theta, kappa = edge.x, edge.y, edge.theta, edge.kappa
    s_arr = np.asarray(s, dtype=float)

    if abs(kappa) < EPS_KAPPA:
        xs = x + s_arr * np.cos(theta)
        ys = y + s_arr * np.sin(theta)
    else:
        C0 = x - (1.0 / kappa) * np.sin(theta)
        C1 = y + (1.0 / kappa) * np.cos(theta)

        xs = (1.0 / kappa) * np.sin(kappa * s_arr + theta) + C0
        ys = -(1.0 / kappa) * np.cos(kappa * s_arr + theta) + C1

    pts = np.stack([xs, ys], axis=-1)

    if np.ndim(s) == 0:
        return pts.reshape(2)

    return pts


def generate_curvel(
    edge: EdgeAnchor,
    s_min: float = -5.0,
    s_max: float = 5.0,
    num: int = 200,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate sampled points on a local curvel.

    Returns
    -------
    curve_points:
        Shape (num, 2).
    s_values:
        Shape (num,).
    """
    s_values = np.linspace(s_min, s_max, num)
    curve_points = curvel_point(edge, s_values)
    return curve_points, s_values


def estimate_s_range_from_neighbors(
    anchor_xy: Tuple[float, float],
    neighbor_points: np.ndarray,
    fallback: float = 5.0,
) -> Tuple[float, float]:
    """
    Estimate a reasonable local s range using neighboring edge points.

    This is useful when the code provides forward/backward neighboring edges.
    If neighbor_points contains 7 edges centered at the anchor, this gives
    approximately the local visible arc-length scale.

    Parameters
    ----------
    anchor_xy:
        Anchor location (x, y).
    neighbor_points:
        Array of nearby edge locations, shape (N, 2).
    fallback:
        Used if neighbor_points is empty or invalid.

    Returns
    -------
    (s_min, s_max)
    """
    if neighbor_points is None or len(neighbor_points) == 0:
        return -fallback, fallback

    neighbor_points = np.asarray(neighbor_points, dtype=float)
    anchor = np.asarray(anchor_xy, dtype=float)

    dists = np.linalg.norm(neighbor_points - anchor[None, :], axis=1)
    max_dist = float(np.nanmax(dists))

    if not np.isfinite(max_dist) or max_dist < 1e-6:
        return -fallback, fallback

    return -max_dist, max_dist


def plot_edge_and_curvel(
    edge: EdgeAnchor,
    edge_points: np.ndarray | None = None,
    s_range: Tuple[float, float] = (-5.0, 5.0),
    save_path: str | None = None,
    show: bool = True,
) -> None:
    """
    Plot detected edge points and the parameterized curvel.

    Parameters
    ----------
    edge:
        Anchor edge.
    edge_points:
        Optional raw detected/local neighboring edge points, shape (N, 2).
    s_range:
        Arc-length plotting range.
    save_path:
        If provided, save figure to this path.
    show:
        Whether to display the plot.
    """
    curve, _ = generate_curvel(edge, s_range[0], s_range[1], num=300)

    plt.figure(figsize=(6, 6))

    if edge_points is not None:
        edge_points = np.asarray(edge_points, dtype=float)
        plt.scatter(edge_points[:, 0], edge_points[:, 1], s=18, label="edge / neighboring edgels")

    plt.plot(curve[:, 0], curve[:, 1], linewidth=2, label="parameterized curvel")
    plt.scatter([edge.x], [edge.y], s=60, marker="x", label="anchor")

    # Draw local tangent direction at anchor.
    tangent_len = 0.2 * max(abs(s_range[0]), abs(s_range[1]), 1.0)
    tx = tangent_len * np.cos(edge.theta)
    ty = tangent_len * np.sin(edge.theta)
    plt.arrow(edge.x, edge.y, tx, ty, head_width=0.08 * tangent_len, length_includes_head=True)

    plt.axis("equal")
    plt.gca().invert_yaxis()  # image coordinate convention; remove if using Cartesian coordinates
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title(f"Curvel from anchor: theta={edge.theta:.3f}, kappa={edge.kappa:.5f}")
    plt.legend()
    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=200)

    if show:
        plt.show()
    else:
        plt.close()


def build_multiview_C_vector(edges: List[EdgeAnchor], s_values: Iterable[float]) -> np.ndarray:
    """
    Build the multiview vector C = [C1(s1)^T, C2(s2)^T, ..., 1]^T.

    This is the vector that replaces the original stacked 2D point vector
    in the edge-based triangulation formulation.

    Parameters
    ----------
    edges:
        List of EdgeAnchor, one per view.
    s_values:
        Arc-length values, one per view.

    Returns
    -------
    np.ndarray
        Shape (2N + 1,).
    """
    s_values = list(s_values)
    if len(edges) != len(s_values):
        raise ValueError("edges and s_values must have the same length.")

    C = []
    for edge, s in zip(edges, s_values):
        pt = curvel_point(edge, s)
        C.extend([float(pt[0]), float(pt[1])])

    C.append(1.0)
    return np.asarray(C, dtype=float)


def demo() -> None:
    """
    A small synthetic demo.

    Replace this part with real data:
        edge = EdgeAnchor(x, y, theta, kappa)
        edge_points = local 7 neighboring edge points
    """
    edge = EdgeAnchor(
        x=100.0,
        y=80.0,
        theta=np.deg2rad(35.0),
        kappa=0.035,
    )

    # Simulate local neighboring edge points from the same curvel.
    s_neighbors = np.linspace(-6, 6, 7)
    edge_points = curvel_point(edge, s_neighbors)

    # Add tiny noise to mimic detected edgels.
    rng = np.random.default_rng(0)
    edge_points = edge_points + rng.normal(scale=0.15, size=edge_points.shape)

    s_range = estimate_s_range_from_neighbors((edge.x, edge.y), edge_points)

    plot_edge_and_curvel(
        edge=edge,
        edge_points=edge_points,
        s_range=s_range,
        save_path="curvel_demo.png",
        show=True,
    )

    # Example multiview C vector with two views.
    edges = [
        EdgeAnchor(x=100.0, y=80.0, theta=np.deg2rad(35.0), kappa=0.035),
        EdgeAnchor(x=130.0, y=78.0, theta=np.deg2rad(31.0), kappa=0.030),
    ]
    s_values = [0.0, 0.0]
    C = build_multiview_C_vector(edges, s_values)
    print("Example multiview C vector:")
    print(C)


if __name__ == "__main__":
    demo()
