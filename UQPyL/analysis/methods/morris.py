import numpy as np
import warnings
from typing import Optional

from ..base import AnaIndex, AnalysisABC
from ...problem import ProblemABC as Problem


class Morris(AnalysisABC):
    """
    Morris Method for Sensitivity Analysis
    Screening-oriented sensitivity analysis based on elementary effects.
    Input steps are relative to parameter ranges; effects retain output units.

    Examples:
        >>> from UQPyL.doe import MorrisDesign
        >>> mor_method = Morris()
        >>> X, meta = MorrisDesign(numLevels=4).sampleWithMeta(problem, 100)
        >>> Y = problem.evaluate(X, target="objs").objs
        >>> mor_method.analyze(problem, X, Y, meta=meta, target="objs")

    References:
        [1] Max D. Morris (1991) Factorial Sampling Plans for Preliminary Computational Experiments,
            Technometrics, 33:2, 161-174, doi: 10.2307/1269043
        [2] SALib, https://github.com/SALib/SALib
    """

    name = "Morris"

    def __init__(self, verboseFlag: bool = True, logFlag: bool = False, saveFlag: bool = False):
        """
        Initialize the Morris method for sensitivity analysis.

        Args:
            verboseFlag: Whether to print compact runtime summaries.
            logFlag: Whether to write a log file.
            saveFlag: Whether to persist results to sqlite.
        """
        super().__init__(verboseFlag, logFlag, saveFlag)

    def _inputRanges(self, problem):
        lower = np.asarray(problem.lb, dtype=float).reshape(-1).copy()
        upper = np.asarray(problem.ub, dtype=float).reshape(-1).copy()
        for variable in getattr(problem.space, "idxD", []):
            choices = problem.space._discrete_values(variable)
            lower[variable], upper[variable] = np.min(choices), np.max(choices)
        with np.errstate(over="ignore", invalid="ignore"):
            ranges = upper - lower
        if (
            ranges.shape != (problem.nInput,)
            or not np.all(np.isfinite(ranges))
            or not np.all(np.isfinite(lower))
            or not np.all(np.isfinite(upper))
            or np.any(ranges <= 0)
        ):
            raise ValueError("Morris unit effects require finite, positive parameter ranges.")
        return ranges

    def checkMeta(self, meta):
        if meta.get("designType") != "morris":
            raise ValueError("Morris.analyze() requires Morris metadata with meta['designType'] == 'morris'.")

        self.set("numLevels", meta["numLevels"])

    def _analyzeCore(
        self,
        problem: Problem,
        X: np.ndarray,
        Y: Optional[np.ndarray] = None,
        meta: Optional[dict] = None,
        target: str = "objs",
        index: AnaIndex = "all",
    ) -> None:
        """
        Run Morris analysis on trajectory samples.

        Args:
            problem: Analysis problem.
            X: Morris sample matrix.
            Y: Optional output matrix corresponding to `X`.
            meta: Sampling metadata from `MorrisDesign.sampleWithMeta`.
            target: Semantic label of `Y`, typically `objs` or `cons`.
            index: Output column selection.
        """
        if meta is None:
            raise TypeError(
                "Morris.analyze() requires metadata. "
                "Use `X, meta = MorrisDesign(...).sampleWithMeta(...)` or pass meta explicitly."
            )

        # Set the problem instance for analysis

        Y = self.check_Y(X, Y, target, index)
        numY = Y.shape[1]
        # Diff preserves unsigned/boolean dtypes, so use signed floating
        # calculations while retaining original X/Y in the result.
        calculationX = np.asarray(X, dtype=float)
        calculationY = np.asarray(Y, dtype=float)
        if not np.all(np.isfinite(calculationX)) or not np.all(np.isfinite(calculationY)):
            raise ValueError("Morris requires finite input and output values.")

        nInput = problem.nInput
        inputRanges = self._inputRanges(problem)

        trajectorySize = nInput + 1
        if X.shape[0] % trajectorySize != 0:
            raise ValueError(f"The number of samples must be divisible by {trajectorySize} for Morris analysis.")

        numTrajectory = int(X.shape[0] / trajectorySize)
        if numTrajectory < 2:
            raise ValueError("Morris requires at least two trajectories to estimate the sample standard deviation.")

        mu = np.zeros((numY, nInput))
        mu_star = np.zeros((numY, nInput))
        sigma = np.zeros((numY, nInput))
        S1_norm = np.zeros((numY, nInput))
        statisticStatus = {name: [] for name in ("mu", "mu_star", "sigma")}

        row_label = self.outputLabels
        col_label_1 = problem.xLabels

        for i in range(numY):
            Y_i = calculationY[:, i : i + 1]

            # Initialize an array to store elementary effects
            EE = np.zeros((nInput, numTrajectory))

            # Calculate elementary effects for each trajectory
            for j in range(numTrajectory):
                X_sub = calculationX[j * trajectorySize : (j + 1) * trajectorySize, :]
                Y_sub = Y_i[j * trajectorySize : (j + 1) * trajectorySize, :]

                Y_diff = np.diff(Y_sub, axis=0)
                X_diff = np.diff(X_sub, axis=0)
                changeMask = X_diff != 0
                changeCounts = np.sum(changeMask, axis=1)
                if not np.all(changeCounts == 1):
                    raise ValueError("Each Morris trajectory step must change exactly one variable.")

                changedVars = np.argmax(changeMask, axis=1)
                order = np.full(nInput, -1, dtype=int)
                for stepIndex, varIndex in enumerate(changedVars):
                    if order[varIndex] != -1:
                        raise ValueError("Each Morris trajectory must change each variable exactly once.")
                    order[varIndex] = stepIndex

                if np.any(order < 0):
                    raise ValueError("Each Morris trajectory must include all input variables.")

                delta_diff = np.sum(X_diff, axis=1).reshape(-1, 1)
                delta_diff = delta_diff / inputRanges[changedVars, None]
                ee = Y_diff / delta_diff
                EE[:, j : j + 1] = ee[order]

            if not np.all(np.isfinite(EE)):
                raise ValueError("Morris elementary effects must be finite.")
            # Normalize before reductions to protect the mean and variance
            # from extreme output units, then restore dimensional statistics.
            effectScale = np.max(np.abs(EE))
            scaledEffects = EE / effectScale if effectScale > 0 else EE
            meanAbs = np.mean(np.abs(scaledEffects), axis=1)
            for name, normalized, destination in (
                ("mu", np.mean(scaledEffects, axis=1), mu),
                ("mu_star", meanAbs, mu_star),
                ("sigma", np.std(scaledEffects, axis=1, ddof=1), sigma),
            ):
                with np.errstate(over="ignore", under="ignore"):
                    restored = normalized * effectScale
                overflow = np.isinf(restored)
                underflow = (normalized != 0) & (effectScale > 0) & (restored == 0)
                status = np.full(nInput, "available", dtype=object)
                for mask, reason in ((overflow, "overflow"), (underflow, "underflow")):
                    status[mask] = reason
                    if np.any(mask):
                        warnings.warn(
                            f"Morris {name} {reason} in output units for output {i}, "
                            f"input indices {np.flatnonzero(mask).tolist()}; "
                            "see extra['morris_statistic_status']. S1_norm remains available.",
                            RuntimeWarning,
                            stacklevel=2,
                        )
                destination[i] = restored
                statisticStatus[name].append(status.tolist())
            total = np.sum(meanAbs)
            if total > 0:
                S1_norm[i] = meanAbs / total

        res = [
            ("mu", mu, row_label, col_label_1, "decsDim1"),
            ("mu_star", mu_star, row_label, col_label_1, "decsDim1"),
            ("sigma", sigma, row_label, col_label_1, "decsDim1"),
            ("S1_norm", S1_norm, row_label, col_label_1, "decsDim1"),
        ]

        self.state.extra["morris_effects"] = dict(
            effect_mode="unit",
            effect_units="output",
            input_ranges=inputRanges.tolist(),
        )
        self.state.extra["morris_statistic_status"] = statisticStatus
        self.recordResult(X, Y, res, target=target, meta=meta)

        return None
