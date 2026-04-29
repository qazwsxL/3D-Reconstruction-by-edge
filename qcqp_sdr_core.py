from __future__ import annotations

import numpy as np
import cvxpy as cp


# ============================================================
# Core algebraic construction
# ============================================================


def symmetrize(M: np.ndarray) -> np.ndarray:
    M = np.asarray(M, dtype=float)
    return 0.5 * (M + M.T)


def edge_sdr_variable_dim(n_views: int) -> int:
    """
    Dimension of the lifted edge-SDR variable in equation (15):

        x = [gamma_1 - Delta gamma_1,
             ...,
             gamma_N - Delta gamma_N,
             1] in R^(3N+1).
    """
    n_views = int(n_views)
    if n_views < 1:
        raise ValueError("n_views must be positive.")
    return 3 * n_views + 1


def stack_edge_observations(edge_points: np.ndarray) -> np.ndarray:
    """
    Stack observed homogeneous edge points into the edge-SDR variable layout:

        [gamma_1, gamma_2, ..., gamma_N, 1]

    where each gamma_i is length-3 and typically has gamma_i[2] = 1.
    """
    edge_points = np.asarray(edge_points, dtype=float)
    if edge_points.ndim != 2 or edge_points.shape[1] != 3:
        raise ValueError("edge_points must have shape (n_views, 3).")

    stacked = edge_points.reshape(-1)
    return np.concatenate([stacked, np.array([1.0], dtype=float)])


def extract_edge_view_from_stacked(x: np.ndarray, i: int) -> np.ndarray:
    """
    Extract the i-th homogeneous edge point from the stacked edge-SDR variable.

    x is expected to follow
        [gamma_1, gamma_2, ..., gamma_N, 1]
    with each gamma_i occupying 3 consecutive entries.
    """
    x = np.asarray(x, dtype=float).reshape(-1)
    if x.size < 4 or (x.size - 1) % 3 != 0:
        raise ValueError("x must have length 3N+1.")

    n_views = (x.size - 1) // 3
    if not (1 <= i <= n_views):
        raise ValueError(f"i must be in {{1, ..., {n_views}}}.")

    start = 3 * (i - 1)
    return x[start:start + 3].copy()


def recover_edge_perturbations(x: np.ndarray, edge_obs: np.ndarray) -> np.ndarray:
    """
    Recover Delta gamma_i = gamma_i(obs) - gamma_i(rec) for each view.

    Returns an array of shape (n_views, 3).
    """
    x = np.asarray(x, dtype=float).reshape(-1)
    edge_obs = np.asarray(edge_obs, dtype=float)

    if x.size < 4 or (x.size - 1) % 3 != 0:
        raise ValueError("x must have length 3N+1.")
    n_views = (x.size - 1) // 3

    if edge_obs.shape != (n_views, 3):
        raise ValueError(f"edge_obs must have shape ({n_views}, 3).")

    x_blocks = x[:-1].reshape(n_views, 3)
    return edge_obs - x_blocks


def recover_edge_points(x: np.ndarray) -> np.ndarray:
    """
    Recover the refined homogeneous edge points from the stacked edge-SDR vector.

    Returns an array of shape (n_views, 3).
    """
    x = np.asarray(x, dtype=float).reshape(-1)
    if x.size < 4 or (x.size - 1) % 3 != 0:
        raise ValueError("x must have length 3N+1.")
    return x[:-1].reshape(-1, 3).copy()


def edge_points_to_stacked_variable(edge_points: np.ndarray) -> np.ndarray:
    """
    Convert refined homogeneous edge points into the edge-SDR stacked variable
    layout [gamma_1, ..., gamma_N, 1].
    """
    edge_points = np.asarray(edge_points, dtype=float)
    if edge_points.ndim != 2 or edge_points.shape[1] != 3:
        raise ValueError("edge_points must have shape (n_views, 3).")
    return np.concatenate([edge_points.reshape(-1), np.array([1.0], dtype=float)])


def build_edge_objective_vector(thetas: np.ndarray) -> np.ndarray:
    """
    Build the linear objective vector c from equation (10):

        c = [-sin(theta_1), cos(theta_1), 0, ..., -sin(theta_N), cos(theta_N), 0, 0]^T
    """
    thetas = np.asarray(thetas, dtype=float).reshape(-1)
    if thetas.size < 1:
        raise ValueError("thetas must be non-empty.")

    c = np.zeros(edge_sdr_variable_dim(thetas.size), dtype=float)
    for idx, theta in enumerate(thetas):
        base = 3 * idx
        c[base] = -np.sin(theta)
        c[base + 1] = np.cos(theta)
    return c


def build_edge_objective_matrix_from_orientations(thetas: np.ndarray) -> np.ndarray:
    """
    Build the lifted symmetric objective matrix M~ from equations (11)-(13).

    This corresponds to the linear term c^T x after dropping constants that
    depend only on the observed edge locations gamma_i. The dropped constant
    does not affect the optimizer.
    """
    c = build_edge_objective_vector(thetas)
    d = c.size
    M = np.zeros((d, d), dtype=float)
    M[:-1, -1] = 0.5 * c[:-1]
    M[-1, :-1] = 0.5 * c[:-1]
    return symmetrize(M)


def build_edge_epipolar_constraint(Fij: np.ndarray, i: int, j: int, n_views: int) -> np.ndarray:
    """
    Build the lifted edge epipolar matrix A~_ij such that

        x^T A~_ij x = (gamma_i - Delta gamma_i)^T Fij (gamma_j - Delta gamma_j)

    for the edge-SDR variable

        x = [gamma_1 - Delta gamma_1,
             ...,
             gamma_N - Delta gamma_N,
             1]^T.

    The trailing homogeneous lift coordinate is not used by this bilinear form,
    so the last row and column are zero.
    """
    Fij = np.asarray(Fij, dtype=float)
    if Fij.shape != (3, 3):
        raise ValueError("Fij must have shape (3, 3).")
    if not (1 <= i <= n_views and 1 <= j <= n_views):
        raise ValueError("i and j must be in {1, ..., n_views}.")
    if i == j:
        raise ValueError("i and j must be different.")

    d = edge_sdr_variable_dim(n_views)
    A = np.zeros((d, d), dtype=float)

    bi = 3 * (i - 1)
    bj = 3 * (j - 1)

    A[bi:bi + 3, bj:bj + 3] += 0.5 * Fij
    A[bj:bj + 3, bi:bi + 3] += 0.5 * Fij.T
    return symmetrize(A)


def stack_edge_constraints_from_pairwise_F(F_dict: dict[tuple[int, int], np.ndarray], n_views: int) -> np.ndarray:
    """
    Convert pairwise fundamental matrices into the edge-SDR constraint stack
    with shape (m, 3N+1, 3N+1).
    """
    mats = []
    for (i, j), Fij in F_dict.items():
        mats.append(build_edge_epipolar_constraint(Fij, i, j, n_views))
    if not mats:
        raise ValueError("F_dict is empty. No constraints were built.")
    return np.stack(mats, axis=0)


def build_edge_homogeneous_coordinate_constraints(n_views: int) -> np.ndarray:
    """
    Build linear lifted constraints enforcing that each refined edge point keeps
    its homogeneous third coordinate equal to 1:

        x_{3i} = 1,  i = 1, ..., N

    in the edge-SDR stacked variable

        x = [gamma_1 - Delta gamma_1,
             ...,
             gamma_N - Delta gamma_N,
             1].

    Each constraint is encoded as

        trace(B_i X) = 0

    with x^T B_i x = x_{3i} - x_last.
    """
    d = edge_sdr_variable_dim(n_views)
    mats = []
    last = d - 1
    for i in range(n_views):
        k = 3 * i + 2
        B = np.zeros((d, d), dtype=float)
        B[k, last] = 0.5
        B[last, k] = 0.5
        B[last, last] = -1.0
        mats.append(symmetrize(B))
    return np.stack(mats, axis=0)


def build_edge_box_constraints(edge_obs: np.ndarray, trust_radius_px: float) -> np.ndarray:
    """
    Build linear lifted inequality constraints that keep each refined x/y
    coordinate within a symmetric pixel box around its observation:

        obs_x - tau <= x_i <= obs_x + tau
        obs_y - tau <= y_i <= obs_y + tau

    Each scalar inequality is encoded as trace(B X) <= 0.
    """
    edge_obs = np.asarray(edge_obs, dtype=float)
    if edge_obs.ndim != 2 or edge_obs.shape[1] != 3:
        raise ValueError("edge_obs must have shape (n_views, 3).")
    if trust_radius_px <= 0:
        raise ValueError("trust_radius_px must be positive.")

    n_views = edge_obs.shape[0]
    d = edge_sdr_variable_dim(n_views)
    last = d - 1
    mats = []

    def linear_ineq_matrix(index: int, coeff: float, offset: float) -> np.ndarray:
        # Encodes coeff * x[index] + offset <= 0 as x^T B x <= 0
        B = np.zeros((d, d), dtype=float)
        B[index, last] = 0.5 * coeff
        B[last, index] = 0.5 * coeff
        B[last, last] = offset
        return symmetrize(B)

    for i in range(n_views):
        xi = 3 * i
        yi = xi + 1
        x_obs = float(edge_obs[i, 0])
        y_obs = float(edge_obs[i, 1])
        tau = float(trust_radius_px)

        mats.append(linear_ineq_matrix(xi, +1.0, -(x_obs + tau)))
        mats.append(linear_ineq_matrix(xi, -1.0, +(x_obs - tau)))
        mats.append(linear_ineq_matrix(yi, +1.0, -(y_obs + tau)))
        mats.append(linear_ineq_matrix(yi, -1.0, +(y_obs - tau)))

    return np.stack(mats, axis=0)


def build_edge_sdr_problem_matrices(
    thetas: np.ndarray,
    F_dict: dict[tuple[int, int], np.ndarray],
    enforce_homogeneous_blocks: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Convenience constructor for equation (15).

    Returns
        M_tilde, A_tilde_stack, E
    where
        minimize    trace(M_tilde X)
        subject to  trace(A_tilde[k] X) = 0
                    trace(E X) = 1
                    X >= 0.
    """
    thetas = np.asarray(thetas, dtype=float).reshape(-1)
    n_views = thetas.size
    if n_views < 1:
        raise ValueError("thetas must be non-empty.")

    M = build_edge_objective_matrix_from_orientations(thetas)
    A = stack_edge_constraints_from_pairwise_F(F_dict, n_views)
    if enforce_homogeneous_blocks:
        A = np.concatenate(
            [A, build_edge_homogeneous_coordinate_constraints(n_views)],
            axis=0,
        )
    E = build_normalization_matrix(edge_sdr_variable_dim(n_views))
    return M, A, E



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
    M[-1, -1] = float(np.dot(obs, obs))
    return symmetrize(M)



def build_global_epipolar_constraint(Fij: np.ndarray, i: int, j: int, n_views: int) -> np.ndarray:
    """
    Build the homogeneous quadratic constraint matrix E_ij such that
        z^T E_ij z = gamma_j^T F_ij gamma_i,
    where
        z = [x1, y1, x2, y2, ..., xN, yN, 1]^T.

    This function is reusable for both point-SDR and edge-SDR as long as the
    variable layout stays the same and only the observation / objective changes.

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

    E[xj, xi] += 0.5 * f11
    E[xi, xj] += 0.5 * f11

    E[xj, yi] += 0.5 * f12
    E[yi, xj] += 0.5 * f12

    E[yj, xi] += 0.5 * f21
    E[xi, yj] += 0.5 * f21

    E[yj, yi] += 0.5 * f22
    E[yi, yj] += 0.5 * f22

    E[xj, c] += 0.5 * f13
    E[c, xj] += 0.5 * f13

    E[yj, c] += 0.5 * f23
    E[c, yj] += 0.5 * f23

    E[xi, c] += 0.5 * f31
    E[c, xi] += 0.5 * f31

    E[yi, c] += 0.5 * f32
    E[c, yi] += 0.5 * f32

    E[c, c] += f33
    return symmetrize(E)



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
# Solver
# ============================================================


class SDRSolver:
    """
    Solve
        minimize    trace(M X)
        subject to  trace(A_k X) = 0,   k = 1, ..., m
                    trace(B_l X) <= 0,  l = 1, ..., q
                    trace(E X) = 1
                    X >= 0  (PSD)
    """

    def __init__(self, M: np.ndarray, A: np.ndarray, E: np.ndarray | None = None,
                 B: np.ndarray | None = None):
        M = symmetrize(np.asarray(M, dtype=float))
        A = np.asarray(A, dtype=float)

        if M.ndim != 2 or M.shape[0] != M.shape[1]:
            raise ValueError("M must be square.")

        if A.ndim == 2:
            A = A[None, :, :]
        if A.ndim != 3 or A.shape[1:] != M.shape:
            raise ValueError("A must have shape (m, d, d) or (d, d).")

        if B is None:
            B = np.zeros((0,) + M.shape, dtype=float)
        else:
            B = np.asarray(B, dtype=float)
            if B.ndim == 2:
                B = B[None, :, :]
            if B.ndim != 3 or B.shape[1:] != M.shape:
                raise ValueError("B must have shape (q, d, d) or (d, d).")

        if E is None:
            E = build_normalization_matrix(M.shape[0])
        E = symmetrize(np.asarray(E, dtype=float))
        if E.shape != M.shape:
            raise ValueError("E must have the same shape as M.")

        self.M = M
        self.A = np.asarray([symmetrize(Ak) for Ak in A])
        self.B = np.asarray([symmetrize(Bk) for Bk in B])
        self.E = E
        self.d = M.shape[0]

        self.X = cp.Variable((self.d, self.d), PSD=True)
        constraints = [cp.trace(self.E @ self.X) == 1]
        for Ak in self.A:
            constraints.append(cp.trace(Ak @ self.X) == 0)
        for Bk in self.B:
            constraints.append(cp.trace(Bk @ self.X) <= 0)

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
        "constraint_max_abs": float(np.max(np.abs(resid))),
        "constraint_mean_abs": float(np.mean(np.abs(resid))),
    }
    out.update(rank1_metrics(X))
    return out


# ============================================================
# Minimal convenience wrapper
# ============================================================



def solve_sdr(M: np.ndarray, A: np.ndarray, E: np.ndarray | None = None,
              B: np.ndarray | None = None,
              solver: str = "MOSEK", verbose: bool = False, **solver_kwargs) -> dict:
    sdr = SDRSolver(M=M, A=A, E=E, B=B)
    X_opt = sdr.solve(solver=solver, verbose=verbose, **solver_kwargs)
    z_rec = recover_rank1_vector(X_opt)
    metrics = summarize_solution(sdr.M, sdr.A, X_opt)

    return {
        "X_opt": X_opt,
        "z_rec": z_rec,
        "status": sdr.problem.status,
        "metrics": metrics,
        "ineq_count": int(sdr.B.shape[0]),
    }


def solve_and_evaluate_eq15(
    edge_obs: np.ndarray,
    thetas: np.ndarray,
    F_dict: dict[tuple[int, int], np.ndarray],
    solver: str = "SCS",
    verbose: bool = False,
    enforce_homogeneous_blocks: bool = True,
    trust_radius_px: float | None = None,
    **solver_kwargs,
) -> dict:
    """
    Solve the equation (15) edge-based SDR:

        minimize    trace(M_tilde X)
        subject to  trace(A_tilde_ij X) = 0
                    X[-1, -1] = 1
                    X >= 0

    Parameters
    ----------
    edge_obs : (n_views, 3) array
        Observed homogeneous edge points gamma_i.
    thetas : (n_views,) array
        Edge orientations theta_i in radians.
    F_dict : dict[(i, j) -> (3, 3) array]
        Pairwise fundamental matrices indexed by 1-based local view indices.
    solver : str
        CVXPY solver name. Defaults to ``SCS`` for portability.

    Returns
    -------
    dict containing:
        edge_obs, thetas, M, A, E, X_opt, z_rec, edge_rec, delta_rec,
        status, metrics, and n_constraints.
    """
    edge_obs = np.asarray(edge_obs, dtype=float)
    if edge_obs.ndim != 2 or edge_obs.shape[1] != 3:
        raise ValueError("edge_obs must have shape (n_views, 3).")

    thetas = np.asarray(thetas, dtype=float).reshape(-1)
    n_views = edge_obs.shape[0]
    if thetas.shape != (n_views,):
        raise ValueError("thetas must have shape (n_views,).")

    M, A, E = build_edge_sdr_problem_matrices(
        thetas,
        F_dict,
        enforce_homogeneous_blocks=enforce_homogeneous_blocks,
    )
    B = None
    if trust_radius_px is not None:
        B = build_edge_box_constraints(edge_obs, trust_radius_px)

    sol = solve_sdr(M, A, E=E, B=B, solver=solver, verbose=verbose, **solver_kwargs)

    z_rec = np.asarray(sol["z_rec"], dtype=float)
    edge_rec = recover_edge_points(z_rec)
    delta_rec = recover_edge_perturbations(z_rec, edge_obs)

    delta_xy = delta_rec[:, :2]
    eval_out = {
        "edge_obs": edge_obs,
        "thetas": thetas,
        "M": M,
        "A": A,
        "E": E,
        "X_opt": sol["X_opt"],
        "z_rec": z_rec,
        "edge_rec": edge_rec,
        "delta_rec": delta_rec,
        "status": sol["status"],
        "metrics": sol["metrics"],
        "n_constraints": int(A.shape[0]),
        "n_ineq_constraints": int(0 if B is None else B.shape[0]),
        "trust_radius_px": trust_radius_px,
        "mean_delta_norm_xy": float(np.mean(np.linalg.norm(delta_xy, axis=1))),
        "max_delta_norm_xy": float(np.max(np.linalg.norm(delta_xy, axis=1))),
    }
    return eval_out


def _edge_refinement_residual_vector(
    xy_flat: np.ndarray,
    edge_obs: np.ndarray,
    thetas: np.ndarray,
    F_dict: dict[tuple[int, int], np.ndarray],
    normal_weight: float,
    epipolar_weight: float,
    tangent_weight: float,
) -> np.ndarray:
    """
    Residual stack for post-SDR nonlinear refinement.

    Variables are the refined image coordinates [x1, y1, ..., xN, yN].
    """
    xy = np.asarray(xy_flat, dtype=float).reshape(-1, 2)
    n_views = xy.shape[0]
    edge_obs = np.asarray(edge_obs, dtype=float)
    thetas = np.asarray(thetas, dtype=float).reshape(-1)

    if edge_obs.shape != (n_views, 3):
        raise ValueError(f"edge_obs must have shape ({n_views}, 3).")
    if thetas.shape != (n_views,):
        raise ValueError(f"thetas must have shape ({n_views},).")

    residuals = []
    sqrt_n = float(np.sqrt(max(normal_weight, 0.0)))
    sqrt_e = float(np.sqrt(max(epipolar_weight, 0.0)))
    sqrt_t = float(np.sqrt(max(tangent_weight, 0.0)))

    for i in range(n_views):
        delta = edge_obs[i, :2] - xy[i]
        theta = thetas[i]
        # Normal direction used in equation (6): [sin(theta), -cos(theta)].
        normal_resid = delta[0] * np.sin(theta) - delta[1] * np.cos(theta)
        if sqrt_n > 0:
            residuals.append(sqrt_n * normal_resid)

        if sqrt_t > 0:
            tangent_resid = delta[0] * np.cos(theta) + delta[1] * np.sin(theta)
            residuals.append(sqrt_t * tangent_resid)

    if sqrt_e > 0:
        for (i, j), Fij in F_dict.items():
            gi = np.array([xy[i - 1, 0], xy[i - 1, 1], 1.0], dtype=float)
            gj = np.array([xy[j - 1, 0], xy[j - 1, 1], 1.0], dtype=float)
            residuals.append(sqrt_e * float(gi @ np.asarray(Fij, dtype=float) @ gj))

    return np.asarray(residuals, dtype=float)


def refine_eq15_solution(
    edge_obs: np.ndarray,
    thetas: np.ndarray,
    F_dict: dict[tuple[int, int], np.ndarray],
    edge_init: np.ndarray | None = None,
    normal_weight: float = 1.0,
    epipolar_weight: float = 1.0e6,
    tangent_weight: float = 1.0e-2,
    trust_radius_px: float | None = 5.0,
    **least_squares_kwargs,
) -> dict:
    """
    Refine an edge-based solution in the original image coordinates using
    nonlinear least squares.

    The refinement objective balances:
    - normal displacement residuals
    - epipolar residuals
    - a weak tangent displacement penalty to prevent drift along the edge

    Parameters
    ----------
    edge_obs : (n_views, 3) array
        Observed homogeneous edge points.
    thetas : (n_views,) array
        Edge orientations in radians.
    F_dict : dict[(i, j) -> (3, 3) array]
        Pairwise epipolar matrices using 1-based local indices.
    edge_init : (n_views, 3) array, optional
        Initial refined edge points. Defaults to edge_obs.
    trust_radius_px : float, optional
        If provided, bounds each refined x/y coordinate to stay within
        +/- trust_radius_px of the observation.
    """
    from scipy.optimize import least_squares

    edge_obs = np.asarray(edge_obs, dtype=float)
    if edge_obs.ndim != 2 or edge_obs.shape[1] != 3:
        raise ValueError("edge_obs must have shape (n_views, 3).")
    thetas = np.asarray(thetas, dtype=float).reshape(-1)
    n_views = edge_obs.shape[0]
    if thetas.shape != (n_views,):
        raise ValueError("thetas must have shape (n_views,).")

    if edge_init is None:
        xy0 = edge_obs[:, :2].copy()
    else:
        edge_init = np.asarray(edge_init, dtype=float)
        if edge_init.shape != (n_views, 3):
            raise ValueError(f"edge_init must have shape ({n_views}, 3).")
        xy0 = edge_init[:, :2].copy()

    kwargs = dict(least_squares_kwargs)
    kwargs.setdefault("method", "trf")
    kwargs.setdefault("max_nfev", 2000)
    kwargs.setdefault("ftol", 1e-10)
    kwargs.setdefault("xtol", 1e-10)
    kwargs.setdefault("gtol", 1e-10)

    bounds = (-np.inf, np.inf)
    if trust_radius_px is not None:
        tau = float(trust_radius_px)
        lower = (edge_obs[:, :2] - tau).reshape(-1)
        upper = (edge_obs[:, :2] + tau).reshape(-1)
        bounds = (lower, upper)
        xy0 = np.clip(xy0.reshape(-1), lower, upper).reshape(n_views, 2)

    result = least_squares(
        _edge_refinement_residual_vector,
        x0=xy0.reshape(-1),
        bounds=bounds,
        args=(edge_obs, thetas, F_dict, normal_weight, epipolar_weight, tangent_weight),
        **kwargs,
    )

    xy_refined = result.x.reshape(n_views, 2)
    edge_refined = np.column_stack([xy_refined, np.ones(n_views, dtype=float)])
    delta_refined = edge_obs - edge_refined
    residual_vec = _edge_refinement_residual_vector(
        result.x,
        edge_obs,
        thetas,
        F_dict,
        normal_weight,
        epipolar_weight,
        tangent_weight,
    )

    epi_residuals = []
    for (i, j), Fij in F_dict.items():
        epi_residuals.append(float(edge_refined[i - 1] @ np.asarray(Fij, dtype=float) @ edge_refined[j - 1]))

    return {
        "success": bool(result.success),
        "status": int(result.status),
        "message": str(result.message),
        "nfev": int(result.nfev),
        "cost": float(result.cost),
        "optimality": float(result.optimality),
        "edge_refined": edge_refined,
        "delta_refined": delta_refined,
        "z_refined": edge_points_to_stacked_variable(edge_refined),
        "mean_delta_norm_xy": float(np.mean(np.linalg.norm(delta_refined[:, :2], axis=1))),
        "max_delta_norm_xy": float(np.max(np.linalg.norm(delta_refined[:, :2], axis=1))),
        "max_abs_epipolar_residual": float(np.max(np.abs(epi_residuals))) if epi_residuals else 0.0,
        "mean_abs_epipolar_residual": float(np.mean(np.abs(epi_residuals))) if epi_residuals else 0.0,
        "residual_l2": float(np.linalg.norm(residual_vec)),
        "normal_weight": float(normal_weight),
        "epipolar_weight": float(epipolar_weight),
        "tangent_weight": float(tangent_weight),
        "trust_radius_px": trust_radius_px,
    }
