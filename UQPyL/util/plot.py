import xarray as xr
import numpy as np
import os, glob, re
from scipy.stats import t




from typing import Optional, List, Union, Literal

# attrs: title, xLabel, yLabel, yLog

colorSet = ["#769FCD", "#F38181", "#A6B1E1", "#B9D7EA", "#A8D8EA", "#BDD2B6", "#FFE2E2"]

markerSet = ["o", "s", "D", "v", "^", "<", ">"]
area_factor = {'o': 1.0, 's': 1.0, 'D': 0.9, 'v': 1.1, '^': 1.1, '<': 1.1, '>': 1.1}



# colorSet = plt.cm.tab10(np.linspace(0, 1, 10))

def _check_plot_var_(var, n, default):
    
    if var is None:
        var = default
    
    if isinstance(var, list) or isinstance(var, np.ndarray):
        return var
    else:
        return [var] * n

def _smooth_curve_(y, window = 10):
    
    if window < 2:
        return y
    
    box = np.ones(window) / window
    y_smooth = np.convolve(y, box, mode='same')
    
    y_smooth[: window * 2] = y[:window * 2]
    y_smooth[-window * 2:] = y[-window * 2:]
    return y_smooth

def _collect_res_(folder: str, prefix: str):
       
    pattern = os.path.join(folder, f"{prefix}_*.nc")
    files = glob.glob(pattern)
    if not files:
        raise FileNotFoundError(f"No files match {pattern}")

    file_tuples = []
    for f in files:
        base = os.path.basename(f)
        m = re.search(rf"{re.escape(prefix)}_(\d+)\.nc$", base)
        if m:
            run_id = int(m.group(1))
            file_tuples.append((run_id, f))
        else:
            print(f"[warn] skip unmatched: {base}")

    file_tuples.sort(key=lambda x: x[0])

    res_dict = {f"run{r}": path for r, path in file_tuples}
    return res_dict

def plot_op_curve(source: dict, xCoord: str = "iter", 
                  yLog: bool = False, ySmooth: bool = False,
                  xlim: Optional[list] = None, xMajorLocator: Optional[int] = None,
                  ylim: Optional[list] = None, yMajorLocator: Optional[int] = None,
                  gridOn: bool = True,
                  fontsize = 20, 
                  title = "Optimization Curve",
                  xLabel = "Iterations", yLabel = "Best Objective",
                  color: Optional[Union[List[str], str]] = None, 
                  linewidth: Optional[Union[List[float], float]] = 2.5, 
                  linestyle: Optional[Union[List[str], str]] = "--",
                  marker: Optional[Union[List[str], str]] = None,
                  markersize: Optional[Union[List[float], float]] = 15,
                  markevery: Optional[Union[List[int], int]] = 30,
                  markeredgecolor: Optional[Union[List[str], str]] = 'black',
                  markeredgewidth: Optional[Union[List[float], float]] = 2):

    import matplotlib.pyplot as plt
    plt.rcParams['font.family'] = 'Arial'
    from matplotlib.ticker import MultipleLocator, LogLocator, NullFormatter
    
    if xCoord not in ["iter", "fe"]:
        raise ValueError(f"xCoord must be 'iter' or 'fe', but got {xCoord}")
    
    # initialize
    colors = _check_plot_var_(color, len(source), colorSet)
    linewidths = _check_plot_var_(linewidth, len(source), 2.5)
    linestyles = _check_plot_var_(linestyle, len(source), "--")
    markers = _check_plot_var_(marker, len(source), markerSet)
    markersizes = _check_plot_var_(markersize, len(source), 15)
    markeverys = _check_plot_var_(markevery, len(source), 30)
    markeredgecolors = _check_plot_var_(markeredgecolor, len(source), 'black')
    markeredgewidths = _check_plot_var_(markeredgewidth, len(source), 2)
    
    fig, ax = plt.subplots(figsize=(15, 8))
    
    for i, (lab, file) in enumerate(source.items()):
        
        ds = xr.open_dataset(file, group= "result")
        objs = ds["bestObjs_Iter"].values.squeeze()
        
        if ySmooth:
            objs = _smooth_curve_(objs)
        
        xCoords = ds.coords[xCoord].values
        
        markeveryList = np.array(list(range(0, len(xCoords), markeverys[i])) + [len(xCoords) - 1])
        
        plot_attrs = {
            "color": colors[i], "linewidth": linewidths[i], "linestyle": linestyles[i], 
            "marker": markers[i], "markersize": markersizes[i] * area_factor[markers[i]], "markevery": markeveryList,
            "markeredgecolor": markeredgecolors[i], "markeredgewidth": markeredgewidths[i],
            "label": lab
        }   # plot attributes
        
        if yLog:
            ax.semilogy(xCoords, objs, **plot_attrs)
        else:
            ax.plot(xCoords, objs, **plot_attrs)
            
    # coords
    if xlim is not None:
        ax.set_xlim(xlim)
        
    if ylim is not None:
        ax.set_ylim(ylim)
        
    if xMajorLocator is not None:
        ax.xaxis.set_major_locator(MultipleLocator(xMajorLocator))
    
    if yMajorLocator is not None and not yLog:
        ax.yaxis.set_major_locator(MultipleLocator(yMajorLocator))
    
    for spine in ax.spines.values():
        spine.set_linewidth(2)
    
    # tick label
    ax.tick_params(axis = 'both', which = 'both', width = 2, length = 4, direction = 'in', labelsize = int(fontsize * 0.9))
    
    # grid
    if gridOn:  
        ax.grid(True, linestyle = '--', linewidth = 1.5, alpha = 0.6)
    
    # label 
    if xLabel is not None:
        ax.set_xlabel(xLabel, fontsize = fontsize)
        
    if yLabel is not None:
        ax.set_ylabel(yLabel, fontsize = fontsize)
    
    if title is not None:
        ax.set_title(title, fontsize = fontsize)
    
    # legend
    ax.legend(fontsize = int(fontsize * 0.9), handlelength = 2, markerscale = 0.8, ncol = 2)
    
    plt.show()


def plot_op_curve_stat(folder: str, prefix: str, idx: list = None, 
                       yLog: bool = False,
                       ySmooth: bool = False,
                       xCoord: Literal["iter", "fe"] = "iter", 
                       ci: Literal["std", "t95"] = "std",
                       fontsize = 20,
                       xlim: Optional[list] = None, xMajorLocator: Optional[int] = None,
                       ylim: Optional[list] = None, yMajorLocator: Optional[int] = None,
                       gridOn: bool = True,
                       title: Optional[str] = None,
                       xLabel: Optional[str] = "Iter", yLabel: Optional[str] = "Best Objective",
                       mean_color: str = "#3F72AF", fill_color: str = "#B7C4CF"):
    
    import matplotlib.pyplot as plt
    plt.rcParams['font.family'] = 'Arial'
    from matplotlib.ticker import MultipleLocator, LogLocator, NullFormatter
    
    name = prefix.split("_")[0]
    
    res_dict = _collect_res_(folder, prefix)
    results = []
    if idx is not None:
        results = [res_dict[i] for i in idx]
    else:
        results = list(res_dict.values())
    
    Objs = []
    for r in results:
        
        ds = xr.open_dataset(r, group= "result")
        xCoords = ds.coords[xCoord].values
        objs = ds["bestObjs_Iter"].values.squeeze()
        Objs.append(objs)
    
    Objs = np.array(Objs)
    
    obj_mean = np.mean(Objs, axis = 0)
    if ySmooth:
        obj_mean = _smooth_curve_(obj_mean)
        
    obj_std = np.std(Objs, axis=0, ddof=1)
    
    if ySmooth:
        obj_std = _smooth_curve_(obj_std)
    
    if ci == "std":
        delta = obj_std
        ci_label = "±1 std"
    elif ci.lower() in ("t95", "95", "ci95", "95ci"):
        se = obj_std / np.sqrt(Objs.shape[0])
        delta = se * t.ppf(0.975, df = Objs.shape[0]-1)  # 95% CI
        ci_label = "95% CI"
        
    lo = np.maximum(obj_mean - delta, np.finfo(float).tiny)  # 避免 semilogy 下界为0
    hi = obj_mean + delta
    
    fig, ax = plt.subplots(figsize=(15, 8))
    
    if yLog:
        ax.semilogy(xCoords, obj_mean, color = mean_color, linewidth = 2, label = name)
        ax.fill_between(xCoords, lo, hi, color = fill_color, alpha = 0.7, label = ci_label)
    else:
        ax.plot(xCoords, obj_mean, color = mean_color, linewidth = 2, label = name)
        ax.fill_between(xCoords, lo, hi, color = fill_color, alpha = 0.7, label = ci_label)

    for spine in ax.spines.values():
        spine.set_linewidth(2)
    
    ax.tick_params(axis = 'both', which = 'both', width = 2, length = 4, direction = 'in', labelsize = int(fontsize * 0.9))
    
    if xlim is not None:
        ax.set_xlim(xlim)
        
    if ylim is not None:
        ax.set_ylim(ylim)
        
    if xMajorLocator is not None:
        ax.xaxis.set_major_locator(MultipleLocator(xMajorLocator))
        
    if yMajorLocator is not None and not yLog:
        ax.yaxis.set_major_locator(MultipleLocator(yMajorLocator))
    
    if gridOn:
        ax.grid(True, linestyle = '--', linewidth = 1.5, alpha = 0.6)
    
    if title is not None:
        ax.set_title(title, fontsize = fontsize)
    
    if xLabel is not None:
        ax.set_xlabel(xLabel, fontsize = fontsize)
        
    if yLabel is not None:
        ax.set_ylabel(yLabel, fontsize = fontsize)
    
    ax.legend(fontsize = int(fontsize * 0.9), handlelength = 2, markerscale = 0.8)
    plt.show()

def plot_op_pareto(filepath: str, optima: np.ndarray = None, fontsize = 20,
                   facecolor: str = "none", edgecolor: str = "#F67280",
                   markersize: float = 200, linewidth: float = 2.5,
                   marker: str = "o",
                   gridOn: bool = True, 
                   xlim: Optional[list] = None, xMajorLocator: Optional[int] = None,
                   ylim: Optional[list] = None, yMajorLocator: Optional[int] = None,
                   title: Optional[str] = None, coordLabels: Optional[List[str]] = None):
    
    import matplotlib.pyplot as plt
    plt.rcParams['font.family'] = 'Arial'
    from matplotlib.ticker import MultipleLocator
    from mpl_toolkits.mplot3d import Axes3D
    
    ds = xr.open_dataset(filepath, group= "result")
    objs = ds["bestObjs"].values
    
    nO = objs.shape[1]
    
    if nO == 2:
        
        fig, ax = plt.subplots(figsize=(15, 8))
        
        ax.scatter(objs[:, 0], objs[:, 1], 
                   facecolors = facecolor, edgecolors = edgecolor, s = markersize,
                   linewidths = linewidth, marker = marker, label = "Pareto Front", zorder = 10)
        
        if optima is not None:
            ax.plot(optima[0], optima[1], color = "#83C5BE", label = "Optima", linewidth = 3)
        
        if coordLabels is not None:
            ax.set_xlabel(coordLabels[0], fontsize = int(fontsize * 0.9))
            ax.set_ylabel(coordLabels[1], fontsize = int(fontsize * 0.9))
        else:
            ax.set_xlabel("Objective 1", fontsize = int(fontsize * 0.9))
            ax.set_ylabel("Objective 2", fontsize = int(fontsize * 0.9))

        # coords
        if xlim is not None:
            ax.set_xlim(xlim)
            
        if ylim is not None:
            ax.set_ylim(ylim)
            
        if xMajorLocator is not None:
            ax.xaxis.set_major_locator(MultipleLocator(xMajorLocator))
            
        if yMajorLocator is not None:
            ax.yaxis.set_major_locator(MultipleLocator(yMajorLocator))
    
    elif nO == 3:
        
        fig = plt.figure(figsize=(12, 12))
        ax = fig.add_subplot(111, projection='3d')
        
        if optima is not None:
            
            if optima[0].shape[0] == 1 or optima[1].shape[0] == 1:
                ax.plot(optima[0], optima[1], optima[2], color = "#83C5BE", label = "Optima", linewidth = 3)
            else:
                ax.plot_surface(optima[0], optima[1], optima[2],
                            color = "#83C5BE", edgecolor = 'none', alpha = 0.8)
        
        ax.scatter(objs[:, 0], objs[:, 1], objs[:, 2], 
                   facecolors = facecolor, edgecolors = edgecolor, s = markersize,
                   linewidths = linewidth, marker = marker, label = "Pareto Front", zorder = 10)
        
        if coordLabels is not None:
            ax.set_xlabel(coordLabels[0], fontsize = int(fontsize * 0.9))
            ax.set_ylabel(coordLabels[1], fontsize = int(fontsize * 0.9))
            ax.set_zlabel(coordLabels[2], fontsize = int(fontsize * 0.9))
        else:
            ax.set_xlabel("Objective 1", fontsize = int(fontsize * 0.9))
            ax.set_ylabel("Objective 2", fontsize = int(fontsize * 0.9))
            ax.set_zlabel("Objective 3", fontsize = int(fontsize * 0.9))

        ax.view_init(elev=30, azim=45)
        
     # tick label
    ax.tick_params(axis = 'both', which = 'both', width = 2, length = 4, direction = 'in', labelsize = int(fontsize * 0.9))
    
    for spine in ax.spines.values():
        spine.set_linewidth(2)
    
    # grid
    if gridOn:  
        ax.grid(True, linestyle = '--', linewidth = 1.5, alpha = 0.6)
        
    ax.legend(fontsize = int(fontsize * 0.9), handlelength = 2, markerscale = 0.8)
    
    if title is not None:
        ax.set_title(title, fontsize = fontsize)
    
    plt.show()

def plot_sa(source: dict, 
            fontsize = 20, 
            width: float = 0.20,
            color: Optional[Union[List[str], str]] = None,
            xLabel: str = "Parameters",
            yLabel: str = "Sensitivity Indices",
            title: Optional[str] = None):
    
    import matplotlib.pyplot as plt
    plt.rcParams['font.family'] = 'Arial'
    from matplotlib.ticker import MultipleLocator, LogLocator, NullFormatter
    
    colorSet = ['#5EABD6', '#E14434', '#03A6A1', '#A7C1A8', '#FFDBB6', '#B7A3E3']
    
    if color is None:
        colors = colorSet
    else:
        colors = _check_plot_var_(color, len(source), colorSet)
    
    colors = ['#5EABD6', '#E14434', '#03A6A1', '#A7C1A8', '#FFDBB6', '#B7A3E3']

    nItem = len(source)
    
    W = width * nItem + 0.4
    
    fig, ax = plt.subplots(1, 1, figsize = (16, 8))
    
    for i, (lab, file) in enumerate(source.items()):
        
        ds = xr.open_dataset(file)
        
        si = ds["S1"].values.ravel()
        
        si = si / np.sum(si)
        
        x = np.arange(len(si)) * W
        
        ax.bar(x + 0.2 + i * width + 0.5 * width, si, width, 
                         label = lab, color = colors[i],
                         edgecolor = 'black', linewidth = 1.5)
    
    if len(source) > 1:
        for i in range(1, len(si)):
            ax.axvline(x = x[i], color = 'black', linestyle = '--', linewidth = 2)
        
    ax.set_xlabel(xLabel, fontsize=25, fontweight='bold')
    
    params = [f"P{i+1}" for i in range(len(si))]
    
    offset = W * 0.5
    
    ax.set_xticks(x + offset, labels = params, fontweight = 'bold')
    
    ax.tick_params(axis = 'x', labelsize = 20)
    
    ax.tick_params(axis = 'y', labelsize = 20)
    
    ax.set_ylabel(yLabel, fontsize = 25, fontweight = 'bold')
    
    ax.set_yticks(np.arange(0, 1.01, 0.1))
    ax.tick_params(axis='y', labelsize=20)
    for label in ax.get_yticklabels():
        label.set_fontweight('bold')
    
    ax.set_ylim(0, 1.0)
    
    ax.set_xlim(x[0], x[-1] + W)
    
    for spine in ax.spines.values():
        spine.set_linewidth(2)
    
    # tick label
    ax.tick_params(axis = 'both', which = 'both', width = 2, length = 4, direction = 'in', labelsize = int(fontsize * 0.9))
    
    ax.tick_params(axis = 'x', length = 0) 
    
    ax.legend(fontsize = int(fontsize * 0.9))
    
    if title is not None:
        ax.set_title(title, fontsize = fontsize)
    
    plt.show()

def plot_surrogate(name: str, yPred: np.ndarray, yTrue: np.ndarray, 
                   fontsize = 20, markersize: float = 300, 
                   xLabel: Optional[str] = "True Value", yLabel: Optional[str] = "Predicted Value",
                   title: Optional[str] = None,
                   ylim: Optional[list] = None, yMajorLocator: Optional[int] = None):
    
    from .metric import r_square, mse
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MultipleLocator
    plt.rcParams['font.family'] = 'Arial'
    
    yTrue = yTrue.ravel()
    yPred = yPred.ravel()
    
    yMax = np.max(np.concatenate([yTrue, yPred])) * 1.1
    yMin = np.min(np.concatenate([yTrue, yPred])) * 0.9
    
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    
    r2 = r_square(yTrue, yPred)
    rmse = np.sqrt(mse(yTrue, yPred))
    
    colors = ['#F08080' if pred > true else '#4682B4' for pred, true in zip(yPred, yTrue)]
    
    ax.scatter(yTrue, yPred, c = colors, s= markersize, alpha = 1.0,
               edgecolor = 'black', linewidth = 1.5)
    
    ax.scatter([], [], c = '#F08080', s = markersize, alpha = 1.0,
               edgecolor = 'black', linewidth = 1.5, label = 'Overestimated')
    
    ax.scatter([], [], c = '#4682B4', s = markersize, alpha = 1.0,
               edgecolor = 'black', linewidth = 1.5, label = 'Underestimated')
    
    ax.legend(loc = 'lower right', frameon = True, fancybox = True, shadow = True,
              fontsize = int(fontsize * 0.9), ncol = 1)

    if ylim is not None:
        ax.set_xlim(ylim)
        ax.set_ylim(ylim)
        ax.plot([ylim[0], ylim[1]], [ylim[0], ylim[1]], '--', color = "#C34C50", lw = 5)
    else:
        ax.set_xlim(yMin, yMax)
        ax.set_ylim(yMin, yMax)
        ax.plot([yMin, yMax], [yMin, yMax], '--', color = "#C34C50", lw = 5)

    if yMajorLocator is not None:
        ax.yaxis.set_major_locator(MultipleLocator(yMajorLocator))
        ax.xaxis.set_major_locator(MultipleLocator(yMajorLocator))
        
    for spine in ax.spines.values():
        spine.set_linewidth(2)
        spine.set_color('black')

    ax.set_xlabel(xLabel, fontweight='bold', fontsize = fontsize)
    ax.set_ylabel(yLabel, fontweight='bold', fontsize = fontsize)

    if title is not None:
        ax.set_title(title, fontweight = 'bold', fontsize = fontsize)

    letter = chr(97)  # a, b, c, d
    ax.text(0.5, -0.15, f"({letter}) {name}", fontweight = 'bold',
            transform=ax.transAxes, ha='center', va='top', fontsize = int(fontsize * 0.9))

    # 添加性能指标
    ax.text(0.05, 0.95, f"$R^2$ = {r2:.3f}, RMSE = {rmse:.3f}",
            transform=ax.transAxes, ha='left', va='top', fontsize = int(fontsize * 0.9),
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
    
    # tick label
    ax.tick_params(axis = 'both', which = 'both', width = 2, length = 4, direction = 'in', labelsize = int(fontsize * 0.9))
    
    plt.show()


def plot_infer_trace(filepath: str, 
                     fontsize = 20, 
                     burnIn: int = 0,
                     colors: Optional[Union[List[str]]] = None,
                     linewidth: float = 2.0,
                     xLabel: Optional[str] = "Iterations",
                     yLabel: Optional[str] = "Value",
                     subtitle: Optional[Union[List[str]]] = None,
                     idx: Optional[list] = None,
                     gridOn: bool = True):

    import xarray as xr
    import matplotlib.pyplot as plt
    import math

    plt.rcParams['font.family'] = 'Arial'

    ds = xr.open_dataset(filepath, group="posterior")
    decs = ds["decs"].values

    nChains, nSamples, nDim = decs.shape

    if idx is not None:
        decs = decs[:, :, idx]
        nDim = len(idx)
    
    nCols = 1 if nDim <= 2 else 2
    nRows = math.ceil(nDim / nCols)

    base_height = 3.5
    base_width = 9 if nCols == 1 else 8
    fig_height = max(4.5, nRows * base_height) + 2
    fig_width  = base_width * nCols + 1

    fig, axes = plt.subplots(nRows, nCols, figsize=(fig_width, fig_height), sharex=False)
    axes = axes.flatten() if nDim > 1 else [axes]

    if colors is None:
        colors = plt.cm.tab10.colors
    else:
        colors = _check_plot_var_(colors, nChains, plt.cm.tab10.colors)

    for i in range(nDim):
        ax = axes[i]
        for j in range(nChains):
            ax.plot(np.arange(burnIn, nSamples), decs[j, burnIn:, i].ravel(),
                    lw = linewidth,
                    color = colors[j % len(colors)],
                    label = f"Chain {j+1}")

        if subtitle is not None:
            ax.set_title(subtitle[i], fontsize=int(fontsize * 0.9))
        else:
            ax.set_title(
                f"Decision Variable {i+1}" if idx is None else f"Decision Variable {idx[i]+1}",
                fontsize=int(fontsize * 0.9)
            )
            
        ax.set_xlabel(xLabel, fontsize=int(fontsize * 0.85))
        ax.set_ylabel(yLabel, fontsize=int(fontsize * 0.85))
        
        ax.tick_params(axis='both', which='both', width=1.6, length=4,
                       direction='in', labelsize=int(fontsize * 0.8))
        
        for spine in ax.spines.values():
            spine.set_linewidth(2)
            spine.set_color('black')
        
        if gridOn:
            ax.grid(alpha = 0.8, linestyle = '--', linewidth = 1.5)

        ax.set_xlim(burnIn, nSamples)
        
    for k in range(nDim, len(axes)):
        fig.delaxes(axes[k])

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels,
               loc  = 'lower center',
               ncol = int(nChains / 2),
               fontsize = int(fontsize * 0.85),
               frameon = True,
               bbox_to_anchor = (0.5, 0.01))

    plt.tight_layout(rect=[0, 0.1, 1, 1])
    
    plt.show()
    
    return fig, axes


def plot_infer_stat(filepath: str, 
                    fontsize = 18, 
                    burnIn: int = 0,
                    bins = 30, hist = True, kde = True,
                    idx: Optional[List[int]] = None,
                    legendOn = True,
                    gridOn = True):
    
    import xarray as xr
    import matplotlib.pyplot as plt
    
    plt.rcParams['font.family'] = 'Arial'
    
    ds = xr.open_dataset(filepath, group="posterior")
    
    decs = ds["decs"].values[:, burnIn:, :]
    
    if idx is not None:
        decs = decs[:, :, idx]
    
    fig, axes = _plot_infer_core(
        decs, bins = bins, hist = hist, kde = kde, mode = "chains",
        fontsize = fontsize, legendOn = legendOn, gridOn = gridOn, idx = idx
    )
    
    plt.show()
    
    return fig, axes


def plot_infer_stat_combined(filepath: str, fontsize = 18, burnIn: int = 0,
                             bins = 30, 
                             hist = True, 
                             kde = True,
                             showCI: bool = False, CI: float = 0.95,
                             legendOn: bool = False):
    import xarray as xr
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
     
    plt.rcParams['font.family'] = 'Arial'
    
    ds = xr.open_dataset(filepath, group="posterior")
    
    decs = ds["decs"].values[:, burnIn:, :]
    
    fig, axes = _plot_infer_core(
        decs, bins = bins, hist = hist, kde = kde, mode = "combined",
        fontsize = fontsize, legendOn = legendOn
    )
    
    if showCI:
      
        alpha_low = (1.0 - CI) / 2.0 * 100.0
        q_low, q_high = alpha_low, 100.0 - alpha_low

        ci_color = "red"
        line_width = 2.2
        span_alpha = 0.15
        
        import numpy as _np
        axes = _np.atleast_1d(axes).ravel()

        nDim = decs.shape[2]
        for i in range(nDim):
            ax = axes[i]
            data = decs[:, :, i].ravel()

            median = np.median(data)
            ci_low, ci_high = np.percentile(data, [q_low, q_high])

            ax.axvline(median, color=ci_color, lw=line_width)
            ax.axvspan(ci_low, ci_high, color=ci_color, alpha=span_alpha)

            handles = [
                plt.Line2D([], [], color = ci_color, lw = line_width, label = 'Median'),
                Patch(facecolor = ci_color, alpha = span_alpha, label = f'{int(CI*100)}% CI')
            ]
            ax.legend(handles = handles, fontsize = int(fontsize*0.9), frameon = True, loc = 'best')
                   
    plt.show()
    
    return fig, axes

def _plot_infer_core(decs, *, 
                     bins = 30, 
                     hist = True, 
                     kde = True,
                     mode = "chains", 
                     fontsize = 18,
                     gridOn = True,
                     idx: Optional[List[int]] = None,
                     legendOn = True):
        
    import numpy as np, math
    import matplotlib.pyplot as plt, seaborn as sns

    nChains, nSamples, nDim = decs.shape
    
    n_cols = int(math.ceil(np.sqrt(nDim)))
    n_rows = int(math.ceil(nDim / n_cols))
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize = ( 9 * n_cols, 8 * n_rows) )
    
    axes = np.atleast_1d(axes).ravel()

    colors = plt.cm.tab10.colors
    handles = []

    for i in range(nDim):
        
        ax = axes[i]
        
        if mode == "combined":
            
            if idx is not None:
                data = decs[:, :, idx[i]].ravel()
            else:
                data = decs[:, :, i].ravel()
            
            if hist:
                sns.histplot(data, bins = bins, stat = 'density', alpha = 0.28,
                             color = colors[0], ax = ax, edgecolor = 'black')
            if kde:
                sns.kdeplot(data, ax = ax, lw = 2, color = 'black', alpha = 0.9)
                
        else:  # mode == 'chains'
            
            for j in range(nChains):
                c = colors[j % len(colors)]
                if hist:
                    sns.histplot(decs[j, :, i], bins = bins, stat = 'density',
                                 alpha = 0.28, color = c, ax = ax, edgecolor = 'black')
                if kde:
                    sns.kdeplot(decs[j, :, i], ax = ax, lw = 2.0, color = c, alpha = 0.9)
                if i == 0 and j < len(colors):
                    handles.append(plt.Line2D([], [], color = c, lw = 2.0, label = f"Chain {j+1}"))

        ax.set_title(f"Decision Variable {idx[i]+1}" if idx is not None else f"Decision Variable {i+1}", fontsize = int(fontsize*0.95))
        ax.set_xlabel("Value", fontsize = int(fontsize*0.9))
        ax.set_ylabel("Density", fontsize = int(fontsize*0.9))
        ax.tick_params(labelsize = int(fontsize*0.85))
        
        ax.tick_params(axis = 'both', which = 'both', width = 1.6, length = 4.0,
                       direction = 'in', labelsize = int(fontsize * 0.8))
        
        for spine in ax.spines.values():
            spine.set_linewidth(2)
            spine.set_color('black')
        
        if gridOn:
            ax.grid(alpha = 0.8, linestyle = '--', linewidth = 1.5)

    for k in range(nDim, len(axes)):
        fig.delaxes(axes[k])

    if legendOn and mode == "chains" and handles:
        
        plt.tight_layout(rect=[0, 0.10, 1, 1])
        fig.legend(handles = handles, ncol = int(len(handles) / 2.0),
                    fontsize = int(fontsize*0.8), frameon = True,
                    loc = "lower center", bbox_to_anchor = (0.5, 0.02),
                    bbox_transform = fig.transFigure)
        
    else:
        plt.tight_layout()

    return fig, axes