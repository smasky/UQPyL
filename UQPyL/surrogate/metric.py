import numpy as np
from scipy.stats import kendalltau
from ._numeric import centeredColumns, squaredSum


def _pairedOutputs(trueY, predY):
    arrays = []
    for values in (trueY, predY):
        values = np.asarray(values, dtype=float)
        if values.ndim == 1:
            values = values.reshape(-1, 1)
        if values.ndim != 2 or not values.size or not np.all(np.isfinite(values)):
            raise ValueError("Metrics require nonempty finite sample matrices.")
        arrays.append(values)
    if arrays[0].shape != arrays[1].shape:
        raise ValueError("True and predicted outputs must have matching sample and output counts.")
    return arrays


def r_square(true_Y: np.ndarray, pre_Y: np.ndarray) -> float:
    """
    R2-score
    """
    true_Y, pre_Y = _pairedOutputs(true_Y, pre_Y)
    centered, powers, _ = centeredColumns(true_Y)
    denominator, denominatorPower = squaredSum(centered, powers)
    if denominator == 0:
        # Retain undefined-score warnings and the caller's NumPy errstate
        # policy for truly constant targets (NaN if exact, otherwise -inf).
        return 1 - np.divide(0.0 if np.array_equal(true_Y, pre_Y) else 1.0, 0.0)
    residualPowers = np.frexp(np.maximum(np.max(np.abs(true_Y), axis=0), np.max(np.abs(pre_Y), axis=0)))[1]
    residual = np.ldexp(true_Y, -residualPowers) - np.ldexp(pre_Y, -residualPowers)
    numerator, numeratorPower = squaredSum(residual, residualPowers)
    return 1 - np.ldexp(numerator / denominator, numeratorPower - denominatorPower)


def nse(true_Y: np.ndarray, pre_Y: np.ndarray) -> float:
    """
    NSE
    """
    return r_square(true_Y, pre_Y)


def mse(true_Y: np.ndarray, pre_Y: np.ndarray) -> np.ndarray:
    """
    Mean square error
    """
    true_Y, pre_Y = _pairedOutputs(true_Y, pre_Y)
    return np.mean(np.square(true_Y - pre_Y), axis=0)


def rank_score(true_Y: np.ndarray, pre_Y: np.ndarray) -> float:
    """Mean per-output Kendall tau-b; constant columns contribute zero."""
    trueY, predY = _pairedOutputs(true_Y, pre_Y)
    if len(trueY) < 2:
        raise ValueError("Rank scoring requires at least two samples.")
    scores = []
    for actual, predicted in zip(trueY.T, predY.T):
        if np.all(actual == actual[0]) or np.all(predicted == predicted[0]):
            scores.append(0.0)
        else:
            scores.append(float(kendalltau(actual, predicted, variant="b").statistic))
    return float(np.mean(scores))


def sort_score(true_Y: np.ndarray, pre_Y: np.ndarray) -> int:
    """
    Sort_score
    """

    t_idx = np.argsort(true_Y.ravel())
    p_idx = np.argsort(pre_Y.ravel())

    return int(np.sum(np.abs(t_idx - p_idx)))
