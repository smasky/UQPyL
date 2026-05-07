from pathlib import Path
from typing import Optional

import numpy as np

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


def plot_sa(source: dict, fontsize=20, width: float = 0.20, color=None, xLabel="Parameters",
            yLabel="Sensitivity Indices", title: Optional[str] = None):
    import matplotlib.pyplot as plt

    colorSet = ["#5EABD6", "#E14434", "#03A6A1", "#A7C1A8", "#FFDBB6", "#B7A3E3"]
    colors = colorSet if color is None else list(color)
    nItem = len(source)
    fig, ax = plt.subplots(1, 1, figsize=(16, 8))

    first_metric = None
    for i, (label, item) in enumerate(source.items()):
        result = _coerce_ana_result(item)
        metric = result.getMetric("S1")
        first_metric = metric
        si = np.asarray(metric.values).reshape(metric.values.shape[0], -1)[0]
        total = np.sum(si)
        if total != 0:
            si = si / total
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

    if first_metric is None:
        raise ValueError("source is empty.")

    params = list(first_metric.colLabels)
    W = width * nItem + 0.4
    x = np.arange(len(params)) * W
    ax.set_xlabel(xLabel, fontsize=25, fontweight="bold")
    ax.set_ylabel(yLabel, fontsize=25, fontweight="bold")
    ax.set_xticks(x + W * 0.5, labels=params, fontweight="bold")
    ax.set_ylim(0, 1.0)
    ax.set_xlim(x[0], x[-1] + W)
    ax.legend(fontsize=int(fontsize * 0.9))
    if title is not None:
        ax.set_title(title, fontsize=fontsize)
    plt.show()
    return fig, ax
