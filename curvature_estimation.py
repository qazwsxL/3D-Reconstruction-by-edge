from __future__ import annotations

import numpy as np


def estimate_curvature_from_three_points(p_prev: np.ndarray, p: np.ndarray, p_next: np.ndarray) -> float:
    """Estimate signed curvature from three 2D points using the circumcircle formula."""
    p_prev = np.asarray(p_prev, dtype=float).reshape(2)
    p = np.asarray(p, dtype=float).reshape(2)
    p_next = np.asarray(p_next, dtype=float).reshape(2)
    a = p - p_prev
    b = p_next - p
    c = p_next - p_prev
    denom = np.linalg.norm(a) * np.linalg.norm(b) * np.linalg.norm(c)
    if denom < 1e-12:
        return 0.0
    signed_area2 = float(np.cross(a, b))
    return 2.0 * signed_area2 / denom


def estimate_curvatures_polyline(points: np.ndarray) -> np.ndarray:
    """Estimate curvature at each point of a 2D polyline; endpoints copy nearest interior value."""
    points = np.asarray(points, dtype=float)
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError("points must have shape (N, 2).")
    n = points.shape[0]
    if n < 3:
        return np.zeros(n, dtype=float)
    kappa = np.zeros(n, dtype=float)
    for i in range(1, n - 1):
        kappa[i] = estimate_curvature_from_three_points(points[i - 1], points[i], points[i + 1])
    kappa[0] = kappa[1]
    kappa[-1] = kappa[-2]
    return kappa


def estimate_orientations_polyline(points: np.ndarray) -> np.ndarray:
    """Estimate tangent orientation theta at each point of a 2D polyline."""
    points = np.asarray(points, dtype=float)
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError("points must have shape (N, 2).")
    n = points.shape[0]
    theta = np.zeros(n, dtype=float)
    for i in range(n):
        if i == 0:
            d = points[1] - points[0]
        elif i == n - 1:
            d = points[-1] - points[-2]
        else:
            d = points[i + 1] - points[i - 1]
        theta[i] = np.arctan2(d[1], d[0])
    return theta
