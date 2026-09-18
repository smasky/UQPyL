"""Shared covariance validation and rank-aware ensemble gain calculation."""
import numpy as np


def validateCovariance(r, nObs):
    if r is None:
        return np.zeros((nObs, nObs), dtype=float)
    matrix = np.asarray(r, dtype=float)
    if matrix.shape != (nObs, nObs):
        raise ValueError("Observation error covariance R must have shape (n_valid_obs, n_valid_obs).")
    if not np.all(np.isfinite(matrix)):
        raise ValueError("Observation error covariance R must be finite.")
    scale = np.max(np.abs(matrix)) if matrix.size else 0.0
    tolerance = max(1, nObs) * np.finfo(float).eps * scale * 10
    if np.max(np.abs(matrix-matrix.T), initial=0.0) > tolerance:
        raise ValueError("Observation error covariance R must be symmetric.")
    matrix = (matrix + matrix.T) * 0.5
    eigenvalues, vectors = np.linalg.eigh(matrix)
    if np.any(eigenvalues < -tolerance):
        raise ValueError("Observation error covariance R must be positive semidefinite.")
    # Remove only negative eigenvalues within the roundoff tolerance.
    if np.any(eigenvalues < 0):
        matrix = (vectors * np.maximum(eigenvalues, 0)) @ vectors.T
    return matrix


def validateRegularization(lam):
    value = np.asarray(lam, dtype=float)
    if value.ndim != 0 or not np.isfinite(value) or value < 0:
        raise ValueError("lam must be a finite nonnegative scalar.")
    return float(value)


def ensembleGain(crossCovariance, simulationCovariance, r, lam=0.0):
    """Solve at full numerical rank; otherwise apply the symmetric pseudoinverse.

    No observation noise or ridge is added beyond the supplied R and lam.
    Eigenvalues <= dimension * machine epsilon * largest eigenvalue are dropped.
    """
    matrix = simulationCovariance + r + lam*np.eye(len(r))
    matrix = (matrix + matrix.T) * 0.5
    if not np.all(np.isfinite(matrix)) or not np.all(np.isfinite(crossCovariance)):
        raise ValueError("Ensemble covariances must be finite.")
    values, vectors = np.linalg.eigh(matrix)
    cutoff = len(matrix)*np.finfo(float).eps*np.max(np.abs(values), initial=0.0)
    if np.any(values < -10*cutoff):
        raise ValueError("Ensemble covariance must be positive semidefinite.")
    active = values > cutoff
    rank = int(np.count_nonzero(active))
    if rank == len(matrix):
        gain = np.linalg.solve(matrix, crossCovariance.T).T
        solver = "solve"
    else:
        basis = vectors[:, active]
        gain = ((crossCovariance @ basis) / values[active]) @ basis.T
        solver = "pinv"
    return gain, {"solver": solver, "rank": rank, "dimension": len(matrix), "cutoff": float(cutoff)}
