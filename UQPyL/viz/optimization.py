from pathlib import Path
from typing import Optional

import numpy as np
from scipy.stats import t

from ..optimization.runtime import OptReader, OptResult
from .common import AREA_FACTOR, COLOR_SET, MARKER_SET, check_plot_var, smooth_curve


def _coerce_opt_result(source):
    if isinstance(source, OptResult):
        return source
    if isinstance(source, OptReader):
        return _build_opt_result_from_reader(source)
    if isinstance(source, (str, Path)):
        reader = OptReader(str(source))
        try:
            return _build_opt_result_from_reader(reader)
        finally:
            reader.close()
    raise TypeError("source must be OptResult, OptReader, or sqlite path.")


def _build_opt_result_from_reader(reader: OptReader):
    run = reader.get_run()
    if run is None:
        raise ValueError("No optimization run found in sqlite database.")
    snapshots = reader.list_snapshots()
    best = reader.load_last_best()

    history = type("History", (), {})()
    history.iterToFEs = [[snap["iter"], snap["fe"]] for snap in snapshots]
    history.bestObjHistory = [snap["bestObj"] for snap in snapshots if snap["bestObj"] is not None]
    history.bestMetricHistory = [snap["hypervolume"] for snap in snapshots if snap["hypervolume"] is not None]
    history.numBestHistory = [snap["paretoSize"] for snap in snapshots if snap["paretoSize"] is not None]

    return OptResult(
        bestDecs=best.decs.copy(),
        bestObjs=best.objs.copy() if best.objs is not None else None,
        bestCons=best.cons.copy() if best.cons is not None else None,
        bestMetric=history.bestMetricHistory[-1] if history.bestMetricHistory else None,
        bestFeasible=True,
        appearFEs=run.get("finalFEs"),
        appearIters=run.get("finalIters"),
        FEs=run.get("finalFEs", 0),
        iters=run.get("finalIters", 0),
        runtime=run.get("runtime", 0.0) or 0.0,
        history=history,
    )


def _history_xy(result: OptResult, x_coord: str):
    pairs = np.asarray(result.history.iterToFEs, dtype=int)
    if pairs.size == 0:
        raise ValueError("Optimization history is empty.")
    x = pairs[:, 0] if x_coord == "iter" else pairs[:, 1]
    if result.bestObjs is not None and result.bestObjs.shape[1] == 1:
        y = np.asarray(result.history.bestObjHistory, dtype=float)
    else:
        y = np.asarray(result.history.bestMetricHistory, dtype=float)
    return x[: len(y)], y


def plot_op_curve(source: dict, xCoord: str = "iter", yLog: bool = False, ySmooth: bool = False, xlim=None,
                  xMajorLocator=None, ylim=None, yMajorLocator=None, gridOn: bool = True, fontsize=20,
                  title="Optimization Curve", xLabel="Iterations", yLabel="Best Objective", color=None,
                  linewidth=2.5, linestyle="--", marker=None, markersize=15, markevery=30,
                  markeredgecolor="black", markeredgewidth=2):
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MultipleLocator

    if xCoord not in {"iter", "fe"}:
        raise ValueError("xCoord must be 'iter' or 'fe'.")

    colors = check_plot_var(color, len(source), COLOR_SET)
    linewidths = check_plot_var(linewidth, len(source), 2.5)
    linestyles = check_plot_var(linestyle, len(source), "--")
    markers = check_plot_var(marker, len(source), MARKER_SET)
    markersizes = check_plot_var(markersize, len(source), 15)
    markeverys = check_plot_var(markevery, len(source), 30)
    markeredgecolors = check_plot_var(markeredgecolor, len(source), "black")
    markeredgewidths = check_plot_var(markeredgewidth, len(source), 2)

    fig, ax = plt.subplots(figsize=(15, 8))
    for i, (label, item) in enumerate(source.items()):
        result = _coerce_opt_result(item)
        x, y = _history_xy(result, xCoord)
        if ySmooth:
            y = smooth_curve(y)
        mark_idx = np.array(list(range(0, len(x), markeverys[i])) + [len(x) - 1])
        plot_attrs = {
            "color": colors[i],
            "linewidth": linewidths[i],
            "linestyle": linestyles[i],
            "marker": markers[i],
            "markersize": markersizes[i] * AREA_FACTOR[markers[i]],
            "markevery": mark_idx,
            "markeredgecolor": markeredgecolors[i],
            "markeredgewidth": markeredgewidths[i],
            "label": label,
        }
        if yLog:
            ax.semilogy(x, y, **plot_attrs)
        else:
            ax.plot(x, y, **plot_attrs)

    if xlim is not None:
        ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)
    if xMajorLocator is not None:
        ax.xaxis.set_major_locator(MultipleLocator(xMajorLocator))
    if yMajorLocator is not None and not yLog:
        ax.yaxis.set_major_locator(MultipleLocator(yMajorLocator))
    if gridOn:
        ax.grid(True, linestyle="--", linewidth=1.5, alpha=0.6)
    if xLabel is not None:
        ax.set_xlabel(xLabel, fontsize=fontsize)
    if yLabel is not None:
        ax.set_ylabel(yLabel, fontsize=fontsize)
    if title is not None:
        ax.set_title(title, fontsize=fontsize)
    ax.legend(fontsize=int(fontsize * 0.9), handlelength=2, markerscale=0.8, ncol=2)
    plt.show()
    return fig, ax


def plot_op_curve_stat(source: dict, xCoord: str = "iter", ci: str = "std", yLog: bool = False, ySmooth: bool = False,
                       fontsize=20, xlim=None, xMajorLocator=None, ylim=None, yMajorLocator=None, gridOn=True,
                       title=None, xLabel="Iter", yLabel="Best Objective", mean_color="#3F72AF", fill_color="#B7C4CF"):
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MultipleLocator

    if len(source) != 1:
        raise ValueError("plot_op_curve_stat currently expects exactly one labeled group.")

    label, items = next(iter(source.items()))
    if not isinstance(items, (list, tuple)):
        items = [items]
    results = [_coerce_opt_result(item) for item in items]
    series = []
    xCoords = None
    for result in results:
        x, y = _history_xy(result, xCoord)
        xCoords = x
        series.append(y)

    values = np.asarray(series, dtype=float)
    obj_mean = np.mean(values, axis=0)
    obj_std = np.std(values, axis=0, ddof=1) if values.shape[0] > 1 else np.zeros_like(obj_mean)
    if ySmooth:
        obj_mean = smooth_curve(obj_mean)
        obj_std = smooth_curve(obj_std)

    if ci == "std":
        delta = obj_std
        ci_label = "±1 std"
    else:
        se = obj_std / np.sqrt(values.shape[0])
        delta = se * t.ppf(0.975, df=max(values.shape[0] - 1, 1))
        ci_label = "95% CI"

    lo = np.maximum(obj_mean - delta, np.finfo(float).tiny)
    hi = obj_mean + delta

    fig, ax = plt.subplots(figsize=(15, 8))
    if yLog:
        ax.semilogy(xCoords, obj_mean, color=mean_color, linewidth=2, label=label)
        ax.fill_between(xCoords, lo, hi, color=fill_color, alpha=0.7, label=ci_label)
    else:
        ax.plot(xCoords, obj_mean, color=mean_color, linewidth=2, label=label)
        ax.fill_between(xCoords, lo, hi, color=fill_color, alpha=0.7, label=ci_label)

    if xlim is not None:
        ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)
    if xMajorLocator is not None:
        ax.xaxis.set_major_locator(MultipleLocator(xMajorLocator))
    if yMajorLocator is not None and not yLog:
        ax.yaxis.set_major_locator(MultipleLocator(yMajorLocator))
    if gridOn:
        ax.grid(True, linestyle="--", linewidth=1.5, alpha=0.6)
    if title is not None:
        ax.set_title(title, fontsize=fontsize)
    if xLabel is not None:
        ax.set_xlabel(xLabel, fontsize=fontsize)
    if yLabel is not None:
        ax.set_ylabel(yLabel, fontsize=fontsize)
    ax.legend(fontsize=int(fontsize * 0.9))
    plt.show()
    return fig, ax


def plot_op_pareto(source, optima: np.ndarray = None, fontsize=20, facecolor="none", edgecolor="#F67280",
                   markersize=200, linewidth=2.5, marker="o", gridOn=True, xlim=None, xMajorLocator=None,
                   ylim=None, yMajorLocator=None, title=None, coordLabels=None):
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MultipleLocator

    result = _coerce_opt_result(source)
    objs = np.asarray(result.bestObjs)
    nO = objs.shape[1]
    if nO not in (2, 3):
        raise ValueError("Pareto plotting only supports 2 or 3 objectives.")

    if nO == 2:
        fig, ax = plt.subplots(figsize=(15, 8))
        ax.scatter(objs[:, 0], objs[:, 1], facecolors=facecolor, edgecolors=edgecolor, s=markersize,
                   linewidths=linewidth, marker=marker, label="Pareto Front", zorder=10)
        if optima is not None:
            ax.plot(optima[0], optima[1], color="#83C5BE", label="Optima", linewidth=3)
        ax.set_xlabel(coordLabels[0] if coordLabels else "Objective 1", fontsize=int(fontsize * 0.9))
        ax.set_ylabel(coordLabels[1] if coordLabels else "Objective 2", fontsize=int(fontsize * 0.9))
        if xlim is not None:
            ax.set_xlim(xlim)
        if ylim is not None:
            ax.set_ylim(ylim)
        if xMajorLocator is not None:
            ax.xaxis.set_major_locator(MultipleLocator(xMajorLocator))
        if yMajorLocator is not None:
            ax.yaxis.set_major_locator(MultipleLocator(yMajorLocator))
    else:
        fig = plt.figure(figsize=(12, 12))
        ax = fig.add_subplot(111, projection="3d")
        ax.scatter(objs[:, 0], objs[:, 1], objs[:, 2], facecolors=facecolor, edgecolors=edgecolor, s=markersize,
                   linewidths=linewidth, marker=marker, label="Pareto Front", zorder=10)
        if coordLabels is not None:
            ax.set_xlabel(coordLabels[0], fontsize=int(fontsize * 0.9))
            ax.set_ylabel(coordLabels[1], fontsize=int(fontsize * 0.9))
            ax.set_zlabel(coordLabels[2], fontsize=int(fontsize * 0.9))
        else:
            ax.set_xlabel("Objective 1", fontsize=int(fontsize * 0.9))
            ax.set_ylabel("Objective 2", fontsize=int(fontsize * 0.9))
            ax.set_zlabel("Objective 3", fontsize=int(fontsize * 0.9))

    if gridOn:
        ax.grid(True, linestyle="--", linewidth=1.5, alpha=0.6)
    if title is not None:
        ax.set_title(title, fontsize=fontsize)
    ax.legend(fontsize=int(fontsize * 0.9))
    plt.show()
    return fig, ax
