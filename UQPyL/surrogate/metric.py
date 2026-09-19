import numpy as np
from scipy.stats import kendalltau

def _pairedOutputs(trueY, predY):
    arrays = []
    for values in (trueY, predY):
        values = np.asarray(values, dtype=float)
        if values.ndim == 1:
            values = values.reshape(-1, 1)
        if values.ndim != 2 or not values.size or not np.all(np.isfinite(values)):
            raise ValueError('Metrics require nonempty finite sample matrices.')
        arrays.append(values)
    if arrays[0].shape != arrays[1].shape:
        raise ValueError('True and predicted outputs must have matching sample and output counts.')
    return arrays


def r_square(true_Y: np.ndarray, pre_Y: np.ndarray) -> float:
    """
    R2-score
    """
    true_Y, pre_Y = _pairedOutputs(true_Y, pre_Y)
    SSR = np.sum(np.square(true_Y-pre_Y))
    mean_Y = np.mean(true_Y, axis=0)
    SST = np.sum(np.square(true_Y-mean_Y))
    
    return 1-SSR/SST

def nse(true_Y: np.ndarray, pre_Y: np.ndarray) -> float:
    """
    NSE
    """
    true_Y, pre_Y = _pairedOutputs(true_Y, pre_Y)
    return 1-np.sum(np.square(true_Y-pre_Y))/np.sum(np.square(true_Y-np.mean(true_Y, axis=0)))

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
        raise ValueError('Rank scoring requires at least two samples.')
    scores = []
    for actual, predicted in zip(trueY.T, predY.T):
        if np.all(actual == actual[0]) or np.all(predicted == predicted[0]):
            scores.append(0.0)
        else:
            scores.append(float(kendalltau(actual, predicted, variant='b').statistic))
    return float(np.mean(scores))


def sort_score(true_Y: np.ndarray, pre_Y: np.ndarray) -> int:
    """
    Sort_score
    """
    
    t_idx = np.argsort(true_Y.ravel())
    p_idx = np.argsort(pre_Y.ravel())
    
    return int(np.sum(np.abs(t_idx-p_idx)))
