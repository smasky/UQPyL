import numpy as np


def plot_surrogate(name: str, yPred: np.ndarray, yTrue: np.ndarray, fontsize=20, markersize: float = 300,
                   xLabel="True Value", yLabel="Predicted Value", title=None, ylim=None, yMajorLocator=None):
    from matplotlib.ticker import MultipleLocator
    import matplotlib.pyplot as plt

    from ..surrogate.metric import mse, r_square

    yTrue = np.asarray(yTrue).ravel()
    yPred = np.asarray(yPred).ravel()
    yMax = np.max(np.concatenate([yTrue, yPred])) * 1.1
    yMin = np.min(np.concatenate([yTrue, yPred])) * 0.9
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    r2 = r_square(yTrue, yPred)
    rmse = np.sqrt(mse(yTrue, yPred))
    colors = ["#F08080" if pred > true else "#4682B4" for pred, true in zip(yPred, yTrue)]
    ax.scatter(yTrue, yPred, c=colors, s=markersize, alpha=1.0, edgecolor="black", linewidth=1.5)
    if ylim is not None:
        ax.set_xlim(ylim)
        ax.set_ylim(ylim)
        ax.plot([ylim[0], ylim[1]], [ylim[0], ylim[1]], "--", color="#C34C50", lw=5)
    else:
        ax.set_xlim(yMin, yMax)
        ax.set_ylim(yMin, yMax)
        ax.plot([yMin, yMax], [yMin, yMax], "--", color="#C34C50", lw=5)
    if yMajorLocator is not None:
        ax.yaxis.set_major_locator(MultipleLocator(yMajorLocator))
        ax.xaxis.set_major_locator(MultipleLocator(yMajorLocator))
    ax.set_xlabel(xLabel, fontweight="bold", fontsize=fontsize)
    ax.set_ylabel(yLabel, fontweight="bold", fontsize=fontsize)
    if title is not None:
        ax.set_title(title, fontweight="bold", fontsize=fontsize)
    ax.text(0.05, 0.95, f"$R^2$ = {r2:.3f}, RMSE = {rmse:.3f}", transform=ax.transAxes,
            ha="left", va="top", fontsize=int(fontsize * 0.9),
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
    ax.text(0.5, -0.15, f"(a) {name}", fontweight="bold", transform=ax.transAxes,
            ha="center", va="top", fontsize=int(fontsize * 0.9))
    plt.show()
    return fig, ax
