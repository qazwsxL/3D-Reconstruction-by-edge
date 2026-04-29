from __future__ import annotations

import numpy as np


def skew(v: np.ndarray) -> np.ndarray:
    """Return the 3x3 cross-product matrix [v]_x."""
    v = np.asarray(v, dtype=float).reshape(3)
    return np.array([
        [0.0, -v[2], v[1]],
        [v[2], 0.0, -v[0]],
        [-v[1], v[0], 0.0],
    ], dtype=float)


def camera_center(P: np.ndarray) -> np.ndarray:
    """Return homogeneous camera center C satisfying P C = 0."""
    P = np.asarray(P, dtype=float)
    if P.shape != (3, 4):
        raise ValueError("P must have shape (3, 4).")
    _, _, Vt = np.linalg.svd(P)
    C = Vt[-1]
    if abs(C[-1]) < 1e-12:
        raise ValueError("Camera center has near-zero homogeneous coordinate.")
    return C / C[-1]


def fundamental_from_projections(P_i: np.ndarray, P_j: np.ndarray) -> np.ndarray:
    """Compute F_ij such that x_j.T @ F_ij @ x_i = 0."""
    P_i = np.asarray(P_i, dtype=float)
    P_j = np.asarray(P_j, dtype=float)
    C_i = camera_center(P_i)
    e_j = P_j @ C_i
    F = skew(e_j) @ P_j @ np.linalg.pinv(P_i)
    norm = np.linalg.norm(F)
    return F / norm if norm > 0 else F


def project(P: np.ndarray, X_h: np.ndarray) -> np.ndarray:
    """Project a homogeneous 3D point into inhomogeneous image coordinates."""
    P = np.asarray(P, dtype=float)
    X_h = np.asarray(X_h, dtype=float).reshape(4)
    x = P @ X_h
    if abs(x[2]) < 1e-12:
        raise ValueError("Projected point has near-zero depth/homogeneous coordinate.")
    return x[:2] / x[2]


def pairwise_fundamentals(cameras: list[np.ndarray]) -> dict[tuple[int, int], np.ndarray]:
    """Build all ordered pairwise fundamental matrices for 1-based view indices."""
    F_dict: dict[tuple[int, int], np.ndarray] = {}
    for i, P_i in enumerate(cameras, start=1):
        for j, P_j in enumerate(cameras, start=1):
            if i == j:
                continue
            F_dict[(i, j)] = fundamental_from_projections(P_i, P_j)
    return F_dict
