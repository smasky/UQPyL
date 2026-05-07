from pathlib import Path
from typing import Optional

import numpy as np

from ..inference.runtime import InfReader, InfResult


def _coerce_inf_result(source):
    if isinstance(source, InfResult):
        return source
    if isinstance(source, InfReader):
        return source.load_result()
    if isinstance(source, (str, Path)):
        reader = InfReader(str(source))
        try:
            return reader.load_result()
        finally:
            reader.close()
    raise TypeError("source must be InfResult, InfReader, or sqlite path.")


def plot_infer_trace(source, fontsize=20, burnIn: int = 0, colors=None, linewidth: float = 2.0,
                     xLabel="Iterations", yLabel="Value", subtitle=None, idx: Optional[list] = None, gridOn=True):
    import math
    import matplotlib.pyplot as plt

    result = _coerce_inf_result(source)
    decs = np.asarray(result.decs)
    nChains, nSamples, nDim = decs.shape
    if idx is not None:
        decs = decs[:, :, idx]
        nDim = len(idx)
    nCols = 1 if nDim <= 2 else 2
    nRows = math.ceil(nDim / nCols)
    fig, axes = plt.subplots(nRows, nCols, figsize=(8 * nCols + 1, max(4.5, nRows * 3.5) + 2), sharex=False)
    axes = np.atleast_1d(axes).ravel()
    colors = colors or plt.cm.tab10.colors
    for i in range(nDim):
        ax = axes[i]
        for j in range(nChains):
            ax.plot(np.arange(burnIn, nSamples), decs[j, burnIn:, i].ravel(), lw=linewidth, color=colors[j % len(colors)])
        ax.set_title(subtitle[i] if subtitle is not None else f"Decision Variable {i+1}", fontsize=int(fontsize * 0.9))
        ax.set_xlabel(xLabel, fontsize=int(fontsize * 0.85))
        ax.set_ylabel(yLabel, fontsize=int(fontsize * 0.85))
        if gridOn:
            ax.grid(alpha=0.8, linestyle="--", linewidth=1.5)
    plt.tight_layout()
    plt.show()
    return fig, axes


def _plot_infer_core(decs, *, bins=30, hist=True, kde=True, mode="chains", fontsize=18, gridOn=True, idx=None, legendOn=True):
    import math
    import matplotlib.pyplot as plt
    import seaborn as sns

    nChains, _, nDim = decs.shape
    n_cols = int(math.ceil(np.sqrt(nDim)))
    n_rows = int(math.ceil(nDim / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(9 * n_cols, 8 * n_rows))
    axes = np.atleast_1d(axes).ravel()
    colors = plt.cm.tab10.colors
    handles = []

    for i in range(nDim):
        ax = axes[i]
        if mode == "combined":
            data = decs[:, :, i].ravel()
            if hist:
                sns.histplot(data, bins=bins, stat="density", alpha=0.28, color=colors[0], ax=ax, edgecolor="black")
            if kde:
                sns.kdeplot(data, ax=ax, lw=2, color="black", alpha=0.9)
        else:
            for j in range(nChains):
                c = colors[j % len(colors)]
                if hist:
                    sns.histplot(decs[j, :, i], bins=bins, stat="density", alpha=0.28, color=c, ax=ax, edgecolor="black")
                if kde:
                    sns.kdeplot(decs[j, :, i], ax=ax, lw=2.0, color=c, alpha=0.9)
                if i == 0:
                    handles.append(plt.Line2D([], [], color=c, lw=2.0, label=f"Chain {j+1}"))
        ax.set_title(f"Decision Variable {idx[i]+1}" if idx is not None else f"Decision Variable {i+1}", fontsize=int(fontsize * 0.95))
        if gridOn:
            ax.grid(alpha=0.8, linestyle="--", linewidth=1.5)
    if legendOn and mode == "chains" and handles:
        plt.tight_layout(rect=[0, 0.10, 1, 1])
        fig.legend(handles=handles, ncol=max(1, int(len(handles) / 2.0)), fontsize=int(fontsize * 0.8),
                   frameon=True, loc="lower center", bbox_to_anchor=(0.5, 0.02), bbox_transform=fig.transFigure)
    else:
        plt.tight_layout()
    return fig, axes


def plot_infer_stat(source, fontsize=18, burnIn: int = 0, bins=30, hist=True, kde=True, idx: Optional[list] = None,
                    legendOn=True, gridOn=True):
    import matplotlib.pyplot as plt

    result = _coerce_inf_result(source)
    decs = np.asarray(result.decs)[:, burnIn:, :]
    if idx is not None:
        decs = decs[:, :, idx]
    fig, axes = _plot_infer_core(decs, bins=bins, hist=hist, kde=kde, mode="chains",
                                 fontsize=fontsize, legendOn=legendOn, gridOn=gridOn, idx=idx)
    plt.show()
    return fig, axes


def plot_infer_stat_combined(source, fontsize=18, burnIn: int = 0, bins=30, hist=True, kde=True,
                             showCI: bool = False, CI: float = 0.95, legendOn: bool = False):
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    result = _coerce_inf_result(source)
    decs = np.asarray(result.decs)[:, burnIn:, :]
    fig, axes = _plot_infer_core(decs, bins=bins, hist=hist, kde=kde, mode="combined",
                                 fontsize=fontsize, legendOn=legendOn)
    if showCI:
        alpha_low = (1.0 - CI) / 2.0 * 100.0
        q_low, q_high = alpha_low, 100.0 - alpha_low
        axes = np.atleast_1d(axes).ravel()
        for i in range(decs.shape[2]):
            ax = axes[i]
            data = decs[:, :, i].ravel()
            median = np.median(data)
            ci_low, ci_high = np.percentile(data, [q_low, q_high])
            ax.axvline(median, color="red", lw=2.2)
            ax.axvspan(ci_low, ci_high, color="red", alpha=0.15)
            handles = [
                plt.Line2D([], [], color="red", lw=2.2, label="Median"),
                Patch(facecolor="red", alpha=0.15, label=f"{int(CI*100)}% CI"),
            ]
            ax.legend(handles=handles, fontsize=int(fontsize * 0.9), frameon=True, loc="best")
    plt.show()
    return fig, axes
