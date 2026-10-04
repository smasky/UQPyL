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
    if np.max(np.abs(matrix - matrix.T), initial=0.0) > tolerance:
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
    matrix = simulationCovariance + r + lam * np.eye(len(r))
    matrix = (matrix + matrix.T) * 0.5
    if not np.all(np.isfinite(matrix)) or not np.all(np.isfinite(crossCovariance)):
        raise ValueError("Ensemble covariances must be finite.")
    values, vectors = np.linalg.eigh(matrix)
    cutoff = len(matrix) * np.finfo(float).eps * np.max(np.abs(values), initial=0.0)
    if np.any(values < -10 * cutoff):
        raise ValueError("Ensemble covariance must be positive semidefinite.")
    active = values > cutoff
    rank = int(np.count_nonzero(active))
    # Reuse the rank-check decomposition for both full and deficient rank.
    basis = vectors[:, active]
    gain = ((crossCovariance @ basis) / values[active]) @ basis.T
    solver = "eigh" if rank == len(matrix) else "pinv"
    return gain, {"solver": solver, "rank": rank, "dimension": len(matrix), "cutoff": float(cutoff)}


def anomalyGain(dX, dY, r=None, lam=0.0):
    """Use thin SVD for zero noise when observations outnumber ensemble members.

    dX and dY are centered ensemble matrices. The SVD branch applies the same
    observation-space eigenvalue cutoff, including the explicit ridge lam.
    Supplied covariance matrices and smaller observation spaces retain the
    general dense solver; small systems do not benefit from the SVD workspace.
    """
    scale = 1.0 / (len(dY) - 1)
    if r is None and dY.shape[1] <= len(dY):
        r = np.zeros((dY.shape[1], dY.shape[1]), dtype=float)
    if r is not None:
        return ensembleGain(dX.T @ dY * scale, dY.T @ dY * scale, r, lam)
    if not np.all(np.isfinite(dX)) or not np.all(np.isfinite(dY)):
        raise ValueError("Ensemble covariances must be finite.")
    rootScale = np.sqrt(scale)
    left, singular, right = np.linalg.svd(dY * rootScale, full_matrices=False)
    with np.errstate(over="ignore"):
        eigenvalues = singular**2 + lam
    if not np.all(np.isfinite(eigenvalues)):
        raise ValueError("Ensemble covariances must be finite.")
    nObs = dY.shape[1]
    cutoff = nObs * np.finfo(float).eps * np.max(eigenvalues, initial=lam)
    active = eigenvalues > cutoff
    # Cxy has no component in the observation nullspace. A positive ridge can
    # nevertheless make those directions count toward the system's rank.
    rank = int(np.count_nonzero(active)) + (nObs - len(singular)) * int(lam > cutoff)
    weights = singular[active] / eigenvalues[active]
    gain = ((dX.T * rootScale @ left[:, active]) * weights) @ right[active]
    if not np.all(np.isfinite(gain)):
        raise ValueError("Ensemble gain must be finite.")
    return gain, {"solver": "svd", "rank": rank, "dimension": nObs, "cutoff": float(cutoff)}


def squareRootAnalysis(X, Y, obs, r):
    """Kalman mean and symmetric square-root anomalies (before box projection)."""
    dX, dY = X - X.mean(axis=0), Y - Y.mean(axis=0)
    if r is None or not np.any(r):
        # With exact observations the transform is an orthogonal projection;
        # its square root is itself. Keep the thin-SVD observation path.
        gain, info = anomalyGain(dX, dY, None)
        return X + (obs - Y) @ gain.T, info
    count = len(X)
    centering = np.eye(count) - 1.0 / count
    memberGain, info = anomalyGain(centering, dY, r)
    gain = dX.T @ memberGain
    mean = X.mean(axis=0) + (obs - Y.mean(axis=0)) @ gain.T
    covariance = np.eye(count) - memberGain @ dY.T
    values, vectors = np.linalg.eigh((covariance + covariance.T) * 0.5)
    tolerance = 100 * count * np.finfo(float).eps
    if np.min(values) < -tolerance:
        raise ValueError("Square-root ensemble covariance is numerically indefinite.")
    transform = (vectors * np.sqrt(np.maximum(values, 0))) @ vectors.T
    anomalies = transform @ dX
    # Remove roundoff in the constant-member direction.
    return mean + anomalies - anomalies.mean(axis=0), info


def regressionSensitivity(X, Y, scales, previous, referenceNorm):
    """Regress responses on prior-scaled parameters, retaining unresolved slopes.

    Retaining the previous slope in lost directions extends the regression to
    collapsed hard-observation ensembles; it is exact for linear models.
    The rank cutoff is anchored to the initial ensemble to reject roundoff
    spread after a zero-noise update.
    """
    anomalies = (X - X.mean(axis=0)) / scales
    responses = Y - Y.mean(axis=0)
    left, singular, right = np.linalg.svd(anomalies, full_matrices=False)
    cutoff = max(anomalies.shape) * np.finfo(float).eps * max(referenceNorm, singular.max(initial=0))
    active = singular > cutoff
    correction = (right[active].T / singular[active]) @ (left[:, active].T @ (responses - anomalies @ previous))
    return previous + correction, int(np.count_nonzero(active))
