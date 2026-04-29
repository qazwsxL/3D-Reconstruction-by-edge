# Curvel-SDR: Multiview Edge-Based Triangulation

This repository implements a curvel-constrained multiview edge triangulation pipeline. It combines local edge/curvel modeling, epipolar QCQP constraints, and semidefinite relaxation (SDR) for globally recoverable edge refinement.

## Core idea

Given an edge correspondence across multiple views, each observed edge sample has:

- location `gamma_i = (x_i, y_i)`,
- tangent orientation `theta_i`,
- optional curvature `kappa_i`.

The curvel model parameterizes the local edge curve by arc length `s_i`. The multiview vector

```text
C = [C_1(s_1)^T, C_2(s_2)^T, ..., C_N(s_N)^T, 1]^T
```

replaces the freely moving point vector in standard multiview triangulation. Epipolar consistency gives a QCQP, which is relaxed to an SDP by lifting `X = yy^T` or `X = xx^T`.

## Repository layout

```text
curvel-sdr/
├── curvel_sdr/
│   ├── core/                 # Eq.5/Eq.15 QCQP + SDR construction
│   ├── curvel/               # curvel formation and curvature estimation
│   ├── solvers/              # solver wrappers
│   └── visualization/        # plotting helpers
├── experiments/
│   ├── synthetic/            # synthetic sanity checks
│   └── multiview/            # dataset runner template
├── notebooks/                # original exploratory notebook
├── docs/                     # project formulation PDF
└── data/                     # optional local data folder
```

## Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

For higher-quality SDP solving, install MOSEK and use `--solver MOSEK`. The examples also work with SCS.

## Run the synthetic Eq. (5) SDR demo

```bash
python experiments/synthetic/run_synthetic_eq5.py --solver SCS
```

This creates a three-view synthetic edge correspondence, solves the curvel-SDR problem, prints rank-1 and residual metrics, and saves a before/after refinement plot.

You can also run the original standalone test:

```bash
python experiments/synthetic/test_eq5_edge_sdr.py --solve --solver SCS
```

## Important files

- `curvel_sdr/core/qcqp_sdr_core_eq5_edge.py`: Eq. (5) curvel/arc-length formulation and SDR construction.
- `curvel_sdr/core/qcqp_sdr_core.py`: Eq. (15)-style lifted edge SDR with homogeneous edge blocks and optional box constraints.
- `curvel_sdr/curvel/curvel_formation.py`: local curvel construction from `(x, y, theta, kappa)`.
- `experiments/synthetic/test_eq5_edge_sdr.py`: original synthetic sanity test.

## Minimal Python usage

```python
import numpy as np
from curvel_sdr.core.qcqp_sdr_core_eq5_edge import EdgeCorrespondence, solve_edge_correspondence_eq5_sdr

edge = EdgeCorrespondence(
    xy=np.array([[0.1, 0.2], [0.0, 0.2], [0.15, 0.1]]),
    theta=np.array([0.2, -0.4, 1.1]),
    kappa=None,
)

# F_dict maps 1-based view pairs (i, j) to F_ij satisfying x_j.T @ F_ij @ x_i = 0.
result = solve_edge_correspondence_eq5_sdr(edge, F_dict, solver="SCS")
print(result["edge_solution"]["xy_linear"])
```

## Notes

- The Eq. (5) implementation uses a first-order curvel approximation in order to preserve a QCQP/SDR form.
- If curvature `kappa` is provided, the recovered arc lengths can also be evaluated using the exact curvel formula for visualization and analysis.
- The uploaded notebook is preserved in `notebooks/` for experiment history, while the package code is organized for reuse.
