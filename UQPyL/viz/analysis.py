from pathlib import Path
from typing import Optional

import numpy as np
import warnings

from ..analysis.runtime import AnaReader, AnaResult


def _coerce_ana_result(source):
    if isinstance(source, AnaResult):
        return source
    if isinstance(source, AnaReader):
        return source.load_result()
    if isinstance(source, (str, Path)):
        reader = AnaReader(str(source))
        try:
            return reader.load_result()
        finally:
            reader.close()
    raise TypeError("source must be AnaResult, AnaReader, or sqlite path.")


def plot_sa(
    source: dict,
    fontsize=20,
    width: float = 0.20,
    color=None,
    xLabel="Parameters",
    yLabel="Sensitivity Indices",
    title: Optional[str] = None,
    *,
    metric: str = "S1",
    outputIndex: int = 0,
):
    import matplotlib.pyplot as plt

    colorSet = ["#5EABD6", "#E14434", "#03A6A1", "#A7C1A8", "#FFDBB6", "#B7A3E3"]
    colors = colorSet if color is None else list(color)
    nItem = len(source)
    fig, ax = plt.subplots(1, 1, figsize=(16, 8))

    if not source or not colors:
        raise ValueError("source and colors must be nonempty.")
    if isinstance(outputIndex, (bool, np.bool_)) or not isinstance(outputIndex, (int, np.integer)) or outputIndex < 0:
        raise ValueError("outputIndex must be a nonnegative integer.")
    params = None
    for i, (label, item) in enumerate(source.items()):
        result = _coerce_ana_result(item)
        selected = result.getMetric(metric)
        values = np.asarray(selected.values)
        labels = list(selected.colLabels)
        if values.ndim != 2 or outputIndex >= values.shape[0] or len(labels) != values.shape[1]:
            raise ValueError("Metric dimensions and outputIndex must match the stored result.")
        if len(set(labels)) != len(labels):
            raise ValueError("Metric column labels must be unique.")
        if params is None:
            params = labels
        if set(labels) != set(params):
            raise ValueError("Compared sensitivity metrics must have the same column labels.")
        si = values[outputIndex, [labels.index(name) for name in params]].astype(float, copy=True)
        if np.any(~np.isfinite(si)):
            warnings.warn(
                f"plot_sa: {label}/{metric} has non-finite values; those bars are omitted.",
                RuntimeWarning,
                stacklevel=2,
            )
            si[~np.isfinite(si)] = np.nan
        W = width * nItem + 0.4
        x = np.arange(len(si)) * W
        ax.bar(
            x + 0.2 + i * width + 0.5 * width,
            si,
            width,
            label=label,
            color=colors[i % len(colors)],
            edgecolor="black",
            linewidth=1.5,
        )

    W = width * nItem + 0.4
    x = np.arange(len(params)) * W
    ax.set_xlabel(xLabel, fontsize=25, fontweight="bold")
    ax.set_ylabel(yLabel, fontsize=25, fontweight="bold")
    ax.set_xticks(x + W * 0.5, labels=params, fontweight="bold")
    ax.set_xlim(x[0], x[-1] + W)
    ax.legend(fontsize=int(fontsize * 0.9))
    if title is not None:
        ax.set_title(title, fontsize=fontsize)
    plt.show()
    return fig, ax
