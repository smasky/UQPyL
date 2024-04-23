import numpy as np
from scipy.stats import norm

def morris_generate_trajectory(N, k, r=0, levels=4):
    """
    Generate Morris trajectories for sensitivity analysis.

    Parameters:
    - N (int): Number of trajectories.
    - k (int): Number of parameters.
    - r (int): Number of levels for each parameter (0 for extended method).
    - levels (int): Number of grid points per parameter (default: 4).

    Returns:
    - X (np.ndarray): Trajectories matrix with shape (N * (k + 1), k).
    """
    delta = 1.0 / (levels - 1)
    B = np.zeros((N, k + 1, k))  # Base trajectories matrix

    for i in range(N):
        base = np.random.rand(k)  # Random start within [0, 1] for each parameter
        B[i, 0, :] = base
        for j in range(1, k + 1):
            B[i, j, :] = B[i, j - 1, :]
            if r == 0:
                # Extended Morris method: symmetric changes
                B[i, j, j - 1] += np.random.choice([-1, 1]) * delta
            else:
                # Original Morris method: only positive increments
                B[i, j, j - 1] += delta

    # Reshape to (N * (k + 1), k) and ensure all values are within [0, 1]
    X = B.reshape(N * (k + 1), k)
    X = np.clip(X, 0, 1)  # Ensure values are not outside the boundaries

    return X

X=morris_generate_trajectory(10, 4, 1, 4)
a=1