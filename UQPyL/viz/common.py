import numpy as np


COLOR_SET = ["#769FCD", "#F38181", "#A6B1E1", "#B9D7EA", "#A8D8EA", "#BDD2B6", "#FFE2E2"]
MARKER_SET = ["o", "s", "D", "v", "^", "<", ">"]
AREA_FACTOR = {"o": 1.0, "s": 1.0, "D": 0.9, "v": 1.1, "^": 1.1, "<": 1.1, ">": 1.1}


def check_plot_var(var, n, default):
    if var is None:
        var = default
    if isinstance(var, (list, tuple, np.ndarray)):
        return list(var)
    return [var] * n


def smooth_curve(y, window=10):
    y = np.asarray(y, dtype=float)
    if window < 2:
        return y
    box = np.ones(window) / window
    y_smooth = np.convolve(y, box, mode="same")
    y_smooth[: window * 2] = y[: window * 2]
    y_smooth[-window * 2 :] = y[-window * 2 :]
    return y_smooth
