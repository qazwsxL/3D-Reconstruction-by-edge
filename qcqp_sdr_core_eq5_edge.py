from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Literal, Sequence

import numpy as np


def _require_cvxpy():
    try:
        import cvxpy as cp
    except ImportError as exc:
        raise ImportError(
            "cvxpy is required to solve the SDR. Install it, or use the "
            "matrix-construction helpers without calling SDRSolver/solve_sdr."
        ) from exc
    return cp


# ============================================================
# Core algebraic construction
# ============================================================


def symmetrize(M: np.ndarray) -> np.ndarray:
    """Return the symmetric part of a square matrix."""
    M = np.asarray(M, dtype=float)
    return 0.5 * (M + M.T)



def build_objective_matrix_from_observation(obs: np.ndarray) -> np.ndarray:
    """
    Build the homogeneous least-squares objective matrix M for
        z = [u, 1]
    with objective
        ||u - obs[:-1]||^2 = z^T M z.

    obs must already be stacked in the same coordinate convention as z,
    and obs[-1] must be 1.
    """
    obs = np.asarray(obs, dtype=float).reshape(-1)
    if obs.ndim != 1 or obs.size < 3:
        raise ValueError("obs must be a 1D vector of length at least 3.")
    if not np.isclose(obs[-1], 1.0):
        raise ValueError("obs[-1] must be 1 for homogeneous lifting.")

    d = obs.size
    M = np.zeros((d, d), dtype=float)
    M[:-1, :-1] = np.eye(d - 1)
    M[:-1, -1] = -obs[:-1]
    M[-1, :-1] = -obs[:-1]
    M[-1, -1] = float(np.dot(obs[:-1], obs[:-1]))
    return symmetrize(M)



def build_global_epipolar_constraint(Fij: np.ndarray, i: int, j: int, n_views: int) -> np.ndarray:
    """
    Build the homogeneous quadratic constraint matrix E_ij such that
        z^T E_ij z = gamma_j^T F_ij gamma_i,
    where
        z = [x1, y1, x2, y2, ..., xN, yN, 1]^T,
        gamma_k = [x_k, y_k, 1]^T.

    The convention is the same as in qcqp_sdr_noisy_dynamic_reduced.ipynb:
    Fij maps a point in view i to an epipolar line in view j.

    i, j are 1-based view indices.
    """
    Fij = np.asarray(Fij, dtype=float)
    if Fij.shape != (3, 3):
        raise ValueError("Fij must have shape (3, 3).")
    if not (1 <= i <= n_views and 1 <= j <= n_views):
        raise ValueError("i and j must be in {1, ..., n_views}.")
    if i == j:
        raise ValueError("i and j must be different.")

    d = 2 * n_views + 1
    E = np.zeros((d, d), dtype=float)

    xi = 2 * (i - 1)
    yi = xi + 1
    xj = 2 * (j - 1)
    yj = xj + 1
    c = d - 1

    f11, f12, f13 = Fij[0, 0], Fij[0, 1], Fij[0, 2]
    f21, f22, f23 = Fij[1, 0], Fij[1, 1], Fij[1, 2]
    f31, f32, f33 = Fij[2, 0], Fij[2, 1], Fij[2, 2]

    # Bilinear terms between view j and view i.
    E[xj, xi] += 0.5 * f11
    E[xi, xj] += 0.5 * f11

    E[xj, yi] += 0.5 * f12
    E[yi, xj] += 0.5 * f12

    E[yj, xi] += 0.5 * f21
    E[xi, yj] += 0.5 * f21

    E[yj, yi] += 0.5 * f22
    E[yi, yj] += 0.5 * f22

    # Linear terms involving the constant coordinate.
    E[xj, c] += 0.5 * f13
    E[c, xj] += 0.5 * f13

    E[yj, c] += 0.5 * f23
    E[c, yj] += 0.5 * f23

    E[xi, c] += 0.5 * f31
    E[c, xi] += 0.5 * f31

    E[yi, c] += 0.5 * f32
    E[c, yi] += 0.5 * f32

    # Constant term.
    E[c, c] += f33
    return symmetrize(E)



def _fundamental_for_order(
    F_dict: dict[tuple[int, int], np.ndarray],
    i: int,
    j: int,
) -> np.ndarray:
    """
    Return F_ij under the convention gamma_j^T F_ij gamma_i = 0.

    If only F_ji is present, use F_ij = F_ji^T.
    """
    if (i, j) in F_dict:
        return np.asarray(F_dict[(i, j)], dtype=float)
    if (j, i) in F_dict:
        return np.asarray(F_dict[(j, i)], dtype=float).T
    raise KeyError(f"No fundamental matrix for pair ({i}, {j}) or ({j}, {i}).")



def reduced_constraint_pairs(n_views: int, anchor_pair: tuple[int, int] = (1, 2)) -> list[tuple[int, int]]:
    """
    Pair pattern for the reduced epipolar constraints used by Eq. (5).

    For anchors (p, q), this returns
        (q, p), and for every i not in {p, q}: (i, p), (i, q).

    With anchors (1, 2), the result is
        (2,1), (3,1), (3,2), ..., (N,1), (N,2),
    which is the 2N - 3 reduced constraint set.
    """
    if n_views < 2:
        raise ValueError("n_views must be at least 2.")
    p, q = anchor_pair
    if not (1 <= p <= n_views and 1 <= q <= n_views and p != q):
        raise ValueError(f"Invalid anchor_pair={anchor_pair} for n_views={n_views}.")

    pairs: list[tuple[int, int]] = [(q, p)]
    for i in range(1, n_views + 1):
        if i in (p, q):
            continue
        pairs.append((i, p))
        pairs.append((i, q))
    return pairs



def select_fundamental_matrices(
    F_dict: dict[tuple[int, int], np.ndarray],
    n_views: int,
    *,
    constraint_mode: Literal["reduced", "full"] = "reduced",
    anchor_pair: tuple[int, int] = (1, 2),
) -> dict[tuple[int, int], np.ndarray]:
    """
    Select full or reduced pairwise fundamental matrices.

    Returns an ordered dict-like plain dict whose insertion order is the
    constraint order. Each returned F_ij satisfies gamma_j^T F_ij gamma_i = 0.
    """
    if constraint_mode not in {"reduced", "full"}:
        raise ValueError("constraint_mode must be 'reduced' or 'full'.")

    if constraint_mode == "reduced":
        pairs = reduced_constraint_pairs(n_views, anchor_pair=anchor_pair)
    else:
        pairs = [(i, j) for i in range(1, n_views + 1) for j in range(i + 1, n_views + 1)]

    return {(i, j): _fundamental_for_order(F_dict, i, j) for i, j in pairs}



def stack_constraints_from_pairwise_F(F_dict: dict[tuple[int, int], np.ndarray], n_views: int) -> np.ndarray:
    """
    Convert pairwise fundamental matrices into a stack of homogeneous
    quadratic constraints A with shape (m, d, d).
    """
    mats = []
    for (i, j), Fij in F_dict.items():
        mats.append(build_global_epipolar_constraint(Fij, i, j, n_views))
    if not mats:
        raise ValueError("F_dict is empty. No constraints were built.")
    return np.stack(mats, axis=0)



def build_normalization_matrix(d: int) -> np.ndarray:
    """
    Build E so that trace(E X) = 1 enforces X[-1, -1] = 1.
    """
    if d < 2:
        raise ValueError("d must be at least 2.")
    E = np.zeros((d, d), dtype=float)
    E[-1, -1] = 1.0
    return E


# ============================================================
# Edge-correspondence / Eq. (5) construction
# ============================================================


@dataclass(frozen=True)
class EdgeCorrespondence:
    """
    One N-view edge correspondence.

    Parameters
    ----------
    xy:
        Array of shape (N, 2), with one 2D detected edge location per view.
    theta:
        Tangent orientation per view in radians. A tangent vector is
        [cos(theta_i), sin(theta_i)].
    kappa:
        Optional curvature per view. It is used for exact curvel evaluation
        after recovering arc-lengths. The SDR itself uses a first-order curvel
        linearization, so that the formulation remains a QCQP/SDR.
    view_ids, edge_ids:
        Optional metadata from an edge-correspondence data file.
    """

    xy: np.ndarray
    theta: np.ndarray
    kappa: np.ndarray | None = None
    view_ids: tuple[int, ...] | None = None
    edge_ids: tuple[int, ...] | None = None

    def __post_init__(self):
        xy = np.asarray(self.xy, dtype=float)
        theta = np.asarray(self.theta, dtype=float).reshape(-1)
        if xy.ndim != 2 or xy.shape[1] != 2:
            raise ValueError("xy must have shape (N, 2).")
        if theta.shape != (xy.shape[0],):
            raise ValueError("theta must have shape (N,).")
        object.__setattr__(self, "xy", xy)
        object.__setattr__(self, "theta", theta)
        if self.kappa is not None:
            kappa = np.asarray(self.kappa, dtype=float).reshape(-1)
            if kappa.shape != (xy.shape[0],):
                raise ValueError("kappa must have shape (N,) when provided.")
            object.__setattr__(self, "kappa", kappa)
        if self.view_ids is not None:
            if len(self.view_ids) != xy.shape[0]:
                raise ValueError("view_ids must have length N when provided.")
            object.__setattr__(self, "view_ids", tuple(int(v) for v in self.view_ids))
        if self.edge_ids is not None:
            if len(self.edge_ids) != xy.shape[0]:
                raise ValueError("edge_ids must have length N when provided.")
            object.__setattr__(self, "edge_ids", tuple(int(e) for e in self.edge_ids))

    @property
    def n_views(self) -> int:
        return int(self.xy.shape[0])

    @classmethod
    def from_stacked(
        cls,
        stacked_xy: np.ndarray,
        theta: np.ndarray,
        kappa: np.ndarray | None = None,
        *,
        view_ids: Sequence[int] | None = None,
        edge_ids: Sequence[int] | None = None,
    ) -> "EdgeCorrespondence":
        """Create an EdgeCorrespondence from [x1,y1,...,xN,yN,1]."""
        xy = unstack_edge_points(stacked_xy)
        return cls(xy=xy, theta=theta, kappa=kappa, view_ids=view_ids, edge_ids=edge_ids)


@dataclass(frozen=True)
class Eq5SDRProblem:
    """
    Matrices for the SDR obtained from Eq. (5) with curvel linearization.

    The optimized homogeneous variable is
        y = [s_1, ..., s_N, 1]^T,
    and the first-order curvel/edge map gives
        C_linear = H y = [C_1(s_1)^T, ..., C_N(s_N)^T, 1]^T.
    """

    M: np.ndarray
    A: np.ndarray
    E: np.ndarray
    H: np.ndarray
    edge: EdgeCorrespondence
    F_selected: dict[tuple[int, int], np.ndarray]
    constraint_mode: str
    anchor_pair: tuple[int, int]



def stack_edge_points(xy: np.ndarray) -> np.ndarray:
    """Return [x1, y1, ..., xN, yN, 1]^T from an (N, 2) array."""
    xy = np.asarray(xy, dtype=float)
    if xy.ndim != 2 or xy.shape[1] != 2:
        raise ValueError("xy must have shape (N, 2).")
    return np.r_[xy.reshape(-1), 1.0]



def unstack_edge_points(stacked: np.ndarray) -> np.ndarray:
    """Return an (N, 2) array from [x1, y1, ..., xN, yN, 1]^T."""
    stacked = np.asarray(stacked, dtype=float).reshape(-1)
    if stacked.size < 5 or stacked.size % 2 != 1:
        raise ValueError("stacked must have length 2N+1 with N >= 2.")
    if not np.isclose(stacked[-1], 1.0):
        raise ValueError("stacked[-1] must be 1.")
    return stacked[:-1].reshape(-1, 2)



def tangent_vectors(theta: np.ndarray) -> np.ndarray:
    """Return unit tangent vectors [cos(theta), sin(theta)]."""
    theta = np.asarray(theta, dtype=float).reshape(-1)
    return np.column_stack([np.cos(theta), np.sin(theta)])



def normal_vectors(theta: np.ndarray) -> np.ndarray:
    """Return left normal vectors [-sin(theta), cos(theta)]."""
    theta = np.asarray(theta, dtype=float).reshape(-1)
    return np.column_stack([-np.sin(theta), np.cos(theta)])



def curvel_points(
    xy: np.ndarray,
    theta: np.ndarray,
    kappa: np.ndarray | float | None,
    s: np.ndarray,
    *,
    curvature_eps: float = 1e-10,
) -> np.ndarray:
    """
    Evaluate the local curve C_i(s_i) from Eq. (4).

    The implementation uses the constants implied by C_i(0)=gamma_i and
    tangent angle theta_i:
        x_i(s) = x_i + [sin(k_i s + theta_i) - sin(theta_i)] / k_i
        y_i(s) = y_i - [cos(k_i s + theta_i) - cos(theta_i)] / k_i
    and falls back to the straight line gamma_i + s_i t_i when k_i ~= 0.
    """
    xy = np.asarray(xy, dtype=float)
    theta = np.asarray(theta, dtype=float).reshape(-1)
    s = np.asarray(s, dtype=float).reshape(-1)
    if xy.ndim != 2 or xy.shape[1] != 2:
        raise ValueError("xy must have shape (N, 2).")
    if theta.shape != (xy.shape[0],) or s.shape != (xy.shape[0],):
        raise ValueError("theta and s must have shape (N,).")

    if kappa is None:
        kappa_arr = np.zeros_like(theta)
    else:
        kappa_arr = np.asarray(kappa, dtype=float) + np.zeros_like(theta)

    out = np.empty_like(xy, dtype=float)
    t = tangent_vectors(theta)
    small = np.abs(kappa_arr) <= curvature_eps
    out[small] = xy[small] + s[small, None] * t[small]

    idx = ~small
    if np.any(idx):
        k = kappa_arr[idx]
        th = theta[idx]
        ss = s[idx]
        out[idx, 0] = xy[idx, 0] + (np.sin(k * ss + th) - np.sin(th)) / k
        out[idx, 1] = xy[idx, 1] - (np.cos(k * ss + th) - np.cos(th)) / k
    return out



def curvel_orientations(theta: np.ndarray, kappa: np.ndarray | float | None, s: np.ndarray) -> np.ndarray:
    """Evaluate theta_i(s_i) = theta_i + kappa_i s_i from Eq. (3)."""
    theta = np.asarray(theta, dtype=float).reshape(-1)
    s = np.asarray(s, dtype=float).reshape(-1)
    if theta.shape != s.shape:
        raise ValueError("theta and s must have the same shape.")
    if kappa is None:
        return theta.copy()
    return theta + np.asarray(kappa, dtype=float) * s



def build_curvel_affine_map(edge: EdgeCorrespondence | np.ndarray, theta: np.ndarray | None = None) -> np.ndarray:
    """
    Build H for the first-order Eq. (4) curvel linearization.

    y = [s_1, ..., s_N, 1]^T
    H y = [x_1 + s_1 cos(theta_1), y_1 + s_1 sin(theta_1), ...,
           x_N + s_N cos(theta_N), y_N + s_N sin(theta_N), 1]^T.

    This is the QCQP-compatible first-order form of Eq. (5): it keeps one
    scalar arc-length variable per view and restricts each refined edge point
    to move along the local curvel tangent at the observed edge.
    """
    if isinstance(edge, EdgeCorrespondence):
        xy = edge.xy
        theta_arr = edge.theta
    else:
        if theta is None:
            raise ValueError("theta must be provided when edge is not an EdgeCorrespondence.")
        xy = np.asarray(edge, dtype=float)
        theta_arr = np.asarray(theta, dtype=float).reshape(-1)

    if xy.ndim != 2 or xy.shape[1] != 2:
        raise ValueError("xy must have shape (N, 2).")
    n = xy.shape[0]
    if theta_arr.shape != (n,):
        raise ValueError("theta must have shape (N,).")

    H = np.zeros((2 * n + 1, n + 1), dtype=float)
    t = tangent_vectors(theta_arr)
    for i in range(n):
        H[2 * i, i] = t[i, 0]
        H[2 * i + 1, i] = t[i, 1]
        H[2 * i, -1] = xy[i, 0]
        H[2 * i + 1, -1] = xy[i, 1]
    H[-1, -1] = 1.0
    return H



def project_quadratic_through_affine(Q: np.ndarray, H: np.ndarray) -> np.ndarray:
    """
    Given C = H y, convert C^T Q C into y^T (H^T Q H) y.
    """
    Q = np.asarray(Q, dtype=float)
    H = np.asarray(H, dtype=float)
    if Q.ndim != 2 or Q.shape[0] != Q.shape[1]:
        raise ValueError("Q must be square.")
    if H.ndim != 2 or H.shape[0] != Q.shape[0]:
        raise ValueError("H must have shape (Q.shape[0], k).")
    return symmetrize(H.T @ Q @ H)



def build_eq5_sdr_problem(
    edge: EdgeCorrespondence,
    F_dict: dict[tuple[int, int], np.ndarray],
    *,
    constraint_mode: Literal["reduced", "full"] = "reduced",
    anchor_pair: tuple[int, int] = (1, 2),
    objective_matrix_C: np.ndarray | None = None,
) -> Eq5SDRProblem:
    """
    Build the SDR matrices for the multiview edge-based triangulation Eq. (5).

    Eq. (5) uses
        C = [C_1(s_1)^T, ..., C_N(s_N)^T, 1]^T
    in the same epipolar QCQP used for point triangulation. To keep the problem
    a QCQP/SDR, this function uses the first-order curvel approximation
        C_i(s_i) = gamma_i + s_i [cos(theta_i), sin(theta_i)]^T.

    The returned matrices are in the arc-length variable
        y = [s_1, ..., s_N, 1]^T.
    """
    if not isinstance(edge, EdgeCorrespondence):
        raise TypeError("edge must be an EdgeCorrespondence instance.")
    n = edge.n_views

    C_obs = stack_edge_points(edge.xy)
    if objective_matrix_C is None:
        objective_matrix_C = build_objective_matrix_from_observation(C_obs)
    objective_matrix_C = symmetrize(np.asarray(objective_matrix_C, dtype=float))
    if objective_matrix_C.shape != (2 * n + 1, 2 * n + 1):
        raise ValueError(f"objective_matrix_C must have shape {(2 * n + 1, 2 * n + 1)}.")

    H = build_curvel_affine_map(edge)
    M_s = project_quadratic_through_affine(objective_matrix_C, H)

    F_selected = select_fundamental_matrices(
        F_dict,
        n_views=n,
        constraint_mode=constraint_mode,
        anchor_pair=anchor_pair,
    )
    A_C = stack_constraints_from_pairwise_F(F_selected, n)
    A_s = np.stack([project_quadratic_through_affine(Ak, H) for Ak in A_C], axis=0)
    E_s = build_normalization_matrix(n + 1)

    return Eq5SDRProblem(
        M=M_s,
        A=A_s,
        E=E_s,
        H=H,
        edge=edge,
        F_selected=F_selected,
        constraint_mode=constraint_mode,
        anchor_pair=anchor_pair,
    )



def recover_edge_solution_from_arc_lengths(problem: Eq5SDRProblem, y: np.ndarray) -> dict:
    """
    Recover edge locations/orientations from y = [s, 1].

    Returns both the linearized locations used by the SDR and, if kappa is
    supplied, the exact Eq. (4) curvel evaluation at the recovered s values.
    """
    y = np.asarray(y, dtype=float).reshape(-1)
    n = problem.edge.n_views
    if y.shape != (n + 1,):
        raise ValueError(f"y must have shape ({n + 1},).")
    if not np.isclose(y[-1], 1.0):
        y = y / y[-1]

    s = y[:-1].copy()
    C_linear = problem.H @ y
    xy_linear = unstack_edge_points(C_linear)
    xy_curvel = curvel_points(problem.edge.xy, problem.edge.theta, problem.edge.kappa, s)
    theta_curvel = curvel_orientations(problem.edge.theta, problem.edge.kappa, s)
    C_curvel = stack_edge_points(xy_curvel)

    return {
        "s": s,
        "C_linear_stacked": C_linear,
        "xy_linear": xy_linear,
        "C_curvel_stacked": C_curvel,
        "xy_curvel": xy_curvel,
        "theta_curvel": theta_curvel,
    }



def solve_edge_correspondence_eq5_sdr(
    edge: EdgeCorrespondence,
    F_dict: dict[tuple[int, int], np.ndarray],
    *,
    constraint_mode: Literal["reduced", "full"] = "reduced",
    anchor_pair: tuple[int, int] = (1, 2),
    solver: str = "MOSEK",
    verbose: bool = False,
    **solver_kwargs,
) -> dict:
    """
    Convenience wrapper: build and solve the Eq. (5) edge SDR.

    Returns
    -------
    dict with keys:
        problem, X_opt, y_rec, edge_solution, status, metrics.
    """
    problem = build_eq5_sdr_problem(
        edge,
        F_dict,
        constraint_mode=constraint_mode,
        anchor_pair=anchor_pair,
    )
    result = solve_sdr(problem.M, problem.A, problem.E, solver=solver, verbose=verbose, **solver_kwargs)
    edge_solution = recover_edge_solution_from_arc_lengths(problem, result["z_rec"])
    return {
        "problem": problem,
        "X_opt": result["X_opt"],
        "y_rec": result["z_rec"],
        "edge_solution": edge_solution,
        "status": result["status"],
        "metrics": result["metrics"],
    }


# ============================================================
# Optional MATLAB correspondence helpers
# ============================================================


@dataclass(frozen=True)
class LoadedMatCorrespondences:
    """Lightweight container returned by load_edge_correspondence_mat."""

    raw: dict
    arrays: dict[str, np.ndarray]
    candidate_keys: dict[str, list[str]]



def load_edge_correspondence_mat(path: str | Path) -> LoadedMatCorrespondences:
    """
    Load a MATLAB edge-correspondence .mat file and expose likely arrays.

    The email describes a file named index_pairs_3D_2D_0006.mat, but the exact
    field names are not fixed in this Python project. This helper therefore
    loads the .mat file and reports candidate numeric arrays rather than
    assuming a brittle schema. Use the reported keys to construct an
    EdgeCorrespondence with EdgeCorrespondence(xy=..., theta=..., kappa=...).
    """
    try:
        import scipy.io as sio
    except ImportError as exc:
        raise ImportError("scipy is required to read MATLAB .mat files.") from exc

    raw = sio.loadmat(str(path), squeeze_me=True, struct_as_record=False)
    arrays: dict[str, np.ndarray] = {}
    for key, value in raw.items():
        if key.startswith("__"):
            continue
        try:
            arr = np.asarray(value)
        except Exception:
            continue
        if np.issubdtype(arr.dtype, np.number):
            arrays[key] = arr

    candidate_keys = {
        "xy_like_Nx2": [],
        "xy_like_Nx3": [],
        "angle_like_1d": [],
        "index_like_int": [],
    }
    for key, arr in arrays.items():
        squeezed = np.squeeze(arr)
        if squeezed.ndim == 2 and squeezed.shape[-1] == 2:
            candidate_keys["xy_like_Nx2"].append(key)
        if squeezed.ndim == 2 and squeezed.shape[-1] == 3:
            candidate_keys["xy_like_Nx3"].append(key)
        if squeezed.ndim == 1 and np.issubdtype(squeezed.dtype, np.floating):
            candidate_keys["angle_like_1d"].append(key)
        if np.issubdtype(squeezed.dtype, np.integer):
            candidate_keys["index_like_int"].append(key)

    return LoadedMatCorrespondences(raw=raw, arrays=arrays, candidate_keys=candidate_keys)


# ============================================================
# Solver
# ============================================================


class SDRSolver:
    """
    Solve
        minimize    trace(M X)
        subject to  trace(A_k X) = 0,   k = 1, ..., m
                    trace(E X) = 1
                    X >= 0  (PSD)
    """

    def __init__(self, M: np.ndarray, A: np.ndarray, E: np.ndarray | None = None):
        M = symmetrize(np.asarray(M, dtype=float))
        A = np.asarray(A, dtype=float)

        if M.ndim != 2 or M.shape[0] != M.shape[1]:
            raise ValueError("M must be square.")

        if A.ndim == 2:
            A = A[None, :, :]
        if A.ndim != 3 or A.shape[1:] != M.shape:
            raise ValueError("A must have shape (m, d, d) or (d, d).")

        if E is None:
            E = build_normalization_matrix(M.shape[0])
        E = symmetrize(np.asarray(E, dtype=float))
        if E.shape != M.shape:
            raise ValueError("E must have the same shape as M.")

        self.M = M
        self.A = np.asarray([symmetrize(Ak) for Ak in A])
        self.E = E
        self.d = M.shape[0]

        cp = _require_cvxpy()
        self.X = cp.Variable((self.d, self.d), PSD=True)
        constraints = [cp.trace(self.E @ self.X) == 1]
        for Ak in self.A:
            constraints.append(cp.trace(Ak @ self.X) == 0)

        self.problem = cp.Problem(cp.Minimize(cp.trace(self.M @ self.X)), constraints)

    def solve(self, solver: str = "MOSEK", verbose: bool = False, **solver_kwargs):
        self.problem.solve(solver=solver, verbose=verbose, **solver_kwargs)
        if self.X.value is None:
            raise RuntimeError(f"Solver failed. Status: {self.problem.status}")
        return self.solution_matrix()

    def solution_matrix(self) -> np.ndarray:
        return symmetrize(np.asarray(self.X.value, dtype=float))


# ============================================================
# Recovery
# ============================================================



def recover_rank1_vector(X: np.ndarray, atol: float = 1e-10) -> np.ndarray:
    """
    Recover z from X using the dominant eigenpair and normalize so z[-1] = 1.
    """
    X = symmetrize(np.asarray(X, dtype=float))
    evals, evecs = np.linalg.eigh(X)
    order = np.argsort(evals)[::-1]
    evals = evals[order]
    evecs = evecs[:, order]

    lam1 = max(float(evals[0]), 0.0)
    z = np.sqrt(lam1) * evecs[:, 0]
    if z[-1] < 0:
        z = -z
    if abs(z[-1]) <= atol:
        raise ValueError("Recovered vector has near-zero homogeneous coordinate.")
    return z / z[-1]


# ============================================================
# Metrics
# ============================================================



def objective_value(M: np.ndarray, X: np.ndarray) -> float:
    return float(np.trace(np.asarray(M, dtype=float) @ np.asarray(X, dtype=float)))



def constraint_residuals(A: np.ndarray, X: np.ndarray) -> np.ndarray:
    A = np.asarray(A, dtype=float)
    if A.ndim == 2:
        A = A[None, :, :]
    X = np.asarray(X, dtype=float)
    return np.array([float(np.trace(Ak @ X)) for Ak in A], dtype=float)



def rank1_metrics(X: np.ndarray) -> dict:
    """
    Generic SDR tightness metrics useful for both point and edge formulations.
    """
    X = symmetrize(np.asarray(X, dtype=float))
    evals = np.linalg.eigvalsh(X)
    evals = np.sort(np.maximum(evals, 0.0))[::-1]

    lam1 = float(evals[0]) if evals.size > 0 else 0.0
    lam2 = float(evals[1]) if evals.size > 1 else 0.0
    traceX = float(np.sum(evals))

    return {
        "lambda1": lam1,
        "lambda2": lam2,
        "lambda2_over_lambda1": lam2 / lam1 if lam1 > 0 else np.inf,
        "rank1_gap": lam1 - lam2,
        "trace": traceX,
    }



def summarize_solution(M: np.ndarray, A: np.ndarray, X: np.ndarray) -> dict:
    """
    Minimal formulation-level metrics only.
    No geometry-specific validation is included.
    """
    resid = constraint_residuals(A, X)
    out = {
        "objective": objective_value(M, X),
        "constraint_max_abs": float(np.max(np.abs(resid))) if resid.size else 0.0,
        "constraint_mean_abs": float(np.mean(np.abs(resid))) if resid.size else 0.0,
    }
    out.update(rank1_metrics(X))
    return out


# ============================================================
# Minimal convenience wrapper
# ============================================================



def solve_sdr(
    M: np.ndarray,
    A: np.ndarray,
    E: np.ndarray | None = None,
    solver: str = "MOSEK",
    verbose: bool = False,
    **solver_kwargs,
) -> dict:
    sdr = SDRSolver(M=M, A=A, E=E)
    X_opt = sdr.solve(solver=solver, verbose=verbose, **solver_kwargs)
    z_rec = recover_rank1_vector(X_opt)
    metrics = summarize_solution(sdr.M, sdr.A, X_opt)

    return {
        "X_opt": X_opt,
        "z_rec": z_rec,
        "status": sdr.problem.status,
        "metrics": metrics,
    }
