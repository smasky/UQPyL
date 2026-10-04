# Delta test
import numpy as np
import warnings
from scipy.spatial import KDTree
from typing import Optional

from ..base import AnaIndex, AnalysisABC
from ...problem import ProblemABC, Problem
from ._variance import scaleOutput, scaleOutputColumns, restoreSquaredOutput


class DeltaTest(AnalysisABC):
    """
    Delta Test
    Non-parametric sensitivity analysis based on nearest-neighbor prediction error.
    The delta estimate is half the mean squared difference to k non-self
    neighbors. S1 is the leave-one-variable-out change, not a Sobol index;
    it has squared output units and can be negative for finite samples.
    Distances use parameter-range-scaled inputs. S1_norm divides by the
    sum of absolute scores before restoring squared units, preserving signs
    and ranking even when raw units underflow. Cutoff ties share the remaining
    neighbor slots uniformly, independently of sample row order.

    Examples:
        >>> from UQPyL.doe import LHS
        >>> delta_method = DeltaTest(nNeighbors=2)
        >>> X = LHS('classic').sample(problem, 1000)
        >>> res = delta_method.analyze(problem, X, target="objs")
        >>> print(res)

    References:
        [1] E. Eirola et al, Using the Delta Test for Variable Selection,
            Artificial Neural Networks, 2008.
        [2] SALib, https://github.com/SALib/SALib
    """

    name = "DeltaTest"

    def __init__(self, nNeighbors: int = 2, verboseFlag: bool = True, logFlag: bool = False, saveFlag: bool = False):
        """
        Initialize the Delta Test method.

        Args:
            nNeighbors: Number of nearest neighbors used by the delta estimate.
            verboseFlag: Whether to print compact runtime summaries.
            logFlag: Whether to write a log file.
            saveFlag: Whether to persist results to sqlite.
        """
        if isinstance(nNeighbors, (bool, np.bool_)) or not isinstance(nNeighbors, (int, np.integer)) or nNeighbors < 1:
            raise ValueError("nNeighbors must be a positive integer.")
        super().__init__(verboseFlag, logFlag, saveFlag)
        self.set("nNeighbors", nNeighbors)

    def _analyzeCore(
        self,
        problem,
        X: np.ndarray,
        Y: Optional[np.ndarray] = None,
        meta: Optional[dict] = None,
        target: str = "objs",
        index: AnaIndex = "all",
    ) -> None:
        """
        Run the Delta Test on the provided samples.

        Args:
            problem: Analysis problem.
            X: Input sample matrix.
            Y: Optional output matrix corresponding to `X`.
            meta: Optional sampling metadata for persistence only.
            target: Semantic label of `Y`, typically `objs` or `cons`.
            index: Output column selection.
        """

        # Set the problem instance for analysis

        # Evaluate the problem if Y is not provided
        Y = self.check_Y(X, Y, target, index)

        nInput = problem.nInput
        if nInput < 2:
            raise ValueError("DeltaTest sensitivity analysis requires at least two inputs.")
        numY = Y.shape[1]
        nNeighbors = self.get("nNeighbors")
        if nNeighbors >= len(X):
            raise ValueError("nNeighbors must be a positive integer smaller than the sample count.")
        scaledX = self._scaleInputs(problem, X)
        constantInputs = np.all(scaledX == scaledX[:1], axis=0)

        S1 = np.zeros((numY, nInput))
        S1_norm = np.zeros((numY, nInput))
        row_label = self.outputLabels
        col_label_1 = problem.xLabels
        diagnostics = []

        for i in range(numY):
            Y_i = Y[:, i : i + 1]
            scaledY, outputScale = scaleOutput(Y_i, returnScale=True)
            base = self._cal_delta(scaledX, scaledY, nNeighbors)
            scaledScores = np.zeros(nInput)
            for j in range(nInput):
                # A constant coordinate carries no distance information; retain
                # its defined zero contribution without another neighbor search.
                if constantInputs[j]:
                    continue
                XSub = np.delete(scaledX, [j], axis=1)
                deltaWithoutVar = self._cal_delta(XSub, scaledY, nNeighbors)
                scaledScores[j] = deltaWithoutVar - base

            if not np.all(np.isfinite(scaledScores)):
                raise ValueError("DeltaTest scores must be finite.")
            # Use a positive denominator even when signed scores cancel or sum
            # to a negative value; first scale to protect the absolute reduction.
            scoreScale = np.max(np.abs(scaledScores))
            if scoreScale > 0:
                relativeScores = scaledScores / scoreScale
                S1_norm[i] = relativeScores / np.sum(np.abs(relativeScores))
            S1[i], underflow = restoreSquaredOutput(scaledScores, outputScale, "DeltaTest")
            diagnostics.append(
                dict(scale_mantissa=outputScale[0], scale_exponent=outputScale[1], raw_underflow=underflow)
            )
            if not np.any(scaledScores > 0) and not np.all(Y_i == Y_i[0]):
                warnings.warn(
                    f"DeltaTest output {row_label[i]} has no positive sensitivity scores; "
                    "the current samples provide no positive variable-importance evidence.",
                    RuntimeWarning,
                    stacklevel=2,
                )

        res = [("S1", S1, row_label, col_label_1, "decsDim1"), ("S1_norm", S1_norm, row_label, col_label_1, "decsDim1")]

        self.state.extra["delta_scaling"] = dict(outputs=diagnostics)
        self.recordResult(X, Y, res, target=target, meta=meta)

        return None

    def findCombEA(
        self,
        problem,
        X: np.ndarray,
        Y: Optional[np.ndarray] = None,
        FEs: int = 10000,
        verboseFlag: bool = True,
        saveFlag: bool = True,
        *,
        seed: Optional[int] = None,
    ):
        """
        Find the best combination using Evolutionary Algorithm.

        Args:
            problem: Analysis problem.
            X: Input sample matrix.
            Y: Optional output matrix corresponding to `X`.
            FEs: Maximum number of function evaluations.
            verboseFlag: Whether the helper GA should print progress.
            saveFlag: Whether the helper GA should persist results.
            seed: Optional seed for reproducible GA subset searches.

        Returns:
            The GA result with finite, scaled squared-output objectives. The
            delta_selection extra records their common physical output scale.
        """
        # Set the problem instance for analysis
        self.setProblem(problem)

        # Retrieve the number of nearest neighbors for analysis
        nNeighbors = self.get("nNeighbors")

        # Evaluate outputs if Y is not provided
        if Y is None:
            Y = self.evaluate(X, target="objs")

        X, Y = self._checkXY(X, Y)
        if nNeighbors >= len(X):
            raise ValueError("nNeighbors must be a positive integer smaller than the sample count.")
        scaledX = self._scaleInputs(problem, X)
        scaledY, outputScale = scaleOutputColumns(Y)
        selectionMeta = self._selectionMeta(outputScale)
        sourceProblem = problem

        @ProblemABC.singleFunc
        def objective(x_):
            """
            Minimize the delta value.

            Args:
                x_: Binary array indicating selected variables.

            Returns:
                The delta value for the selected variables.
            """
            x_ = x_.astype(int)
            Indices = np.where(x_ == 1)[0]
            XSub = scaledX[:, Indices]
            if getattr(ga, "state", None) is not None:
                ga.state.extra["delta_selection"] = selectionMeta

            if np.sum(x_) == 0:
                return np.inf
            else:
                return self._selectionDelta(XSub, scaledY, nNeighbors, outputScale, selectionMeta)

        # Create the optimization problem
        nInput = problem.nInput
        nObj = 1
        ub = [1] * nInput
        lb = [0] * nInput
        varType = [1] * nInput

        problem = Problem(
            nInput=nInput,
            nObj=nObj,
            ub=ub,
            lb=lb,
            varType=varType,
            objFunc=objective,
            optType="min",
            name="DeltaTest scaled subset selection",
            xLabels=sourceProblem.xLabels,
        )
        problem.workDir = getattr(sourceProblem, "workDir", None)

        # Initialize the GA
        from ...optimization.soea import GA

        ga = GA(maxFEs=FEs, verboseFlag=verboseFlag, saveFlag=saveFlag)

        # Run the GA
        res = ga.run(problem) if seed is None else ga.run(problem, seed=seed)

        return res

    def findCombVio(self, problem, X: np.ndarray, Y: Optional[np.ndarray] = None):
        """
        Find the best combination using a brute-force approach.

        Args:
            problem: Analysis problem.
            X: Input sample matrix.
            Y: Optional output matrix corresponding to `X`.

        Returns:
            Labels of the selected variables.
        """

        from itertools import product

        # Set the problem instance for analysis
        self.setProblem(problem)

        nInput = problem.nInput

        # Retrieve the number of nearest neighbors for analysis
        nNeighbors = self.get("nNeighbors")

        # Evaluate outputs if Y is not provided
        if Y is None:
            Y = self.evaluate(X, target="objs")

        X, Y = self._checkXY(X, Y)
        if nNeighbors >= len(X):
            raise ValueError("nNeighbors must be a positive integer smaller than the sample count.")
        scaledX = self._scaleInputs(problem, X)
        scaledY, outputScale = scaleOutputColumns(Y)
        selectionMeta = self._selectionMeta(outputScale)

        # Generate all possible combinations of input variables
        combinations = list(product([0, 1], repeat=nInput))

        # Initialize an array to store objective values for each combination
        objs = np.zeros((len(combinations), 1))

        # Evaluate each combination
        for i in range(len(combinations)):
            x_ = np.array(combinations[i])
            Indices = np.where(x_ == 1)[0]
            XSub = scaledX[:, Indices]

            if np.sum(x_) == 0:
                objs[i] = np.inf
            else:
                objs[i] = self._selectionDelta(XSub, scaledY, nNeighbors, outputScale, selectionMeta)

        # Find the best combination based on the objective values
        best_index = np.argmin(objs)
        best_combination = combinations[best_index]

        # Return the labels of the most sensitive variables
        return [problem.xLabels[i] for i in range(nInput) if best_combination[i] == 1]

    def _selectionMeta(self, outputScale):
        return dict(
            objective_units="scaled_output_squared",
            output_aggregation="half_mean_squared_difference",
            scale_mantissa=outputScale[0],
            scale_exponent=outputScale[1],
            raw_underflow=False,
            raw_overflow=False,
        )

    def _selectionDelta(self, X, Y, nNeighbors, outputScale, metadata):
        value = self._cal_delta(X, Y, nNeighbors)
        try:
            _, underflow = restoreSquaredOutput(value, outputScale, "DeltaTest", warnUnderflow=False)
            flag = "raw_underflow" if underflow else None
        except ValueError:
            flag = "raw_overflow"
        if flag is not None and not metadata[flag]:
            metadata[flag] = True
            warnings.warn(
                f"DeltaTest subset raw objective has {flag.removeprefix('raw_')} in physical squared units; "
                "selection and EA results use finite scaled objectives.",
                RuntimeWarning,
                stacklevel=3,
            )
        return value

    def _scaleInputs(self, problem, X):
        """Scale numeric coordinates once, before selecting variable subsets.

        Continuous/integer inputs use declared bounds. Discrete numeric choices
        use their actual value range, since Space bounds describe their encoding.
        Out-of-bound samples are not clipped; their distances remain distinct.
        """
        values = np.asarray(X, dtype=float)
        lower = np.asarray(problem.lb, dtype=float).reshape(-1).copy()
        upper = np.asarray(problem.ub, dtype=float).reshape(-1).copy()
        for index in getattr(problem.space, "idxD", []):
            choices = problem.space._discrete_values(index)
            lower[index], upper[index] = np.min(choices), np.max(choices)
        if (
            lower.shape != (problem.nInput,)
            or upper.shape != (problem.nInput,)
            or not np.all(np.isfinite(lower))
            or not np.all(np.isfinite(upper))
            or np.any(upper < lower)
        ):
            raise ValueError("DeltaTest distance scaling requires finite, ordered parameter bounds.")
        if not np.all(np.isfinite(values)):
            raise ValueError("DeltaTest inputs must be finite.")
        fixed = upper == lower
        if np.any(values[:, fixed] != lower[fixed]):
            raise ValueError("DeltaTest fixed inputs must match their declared value.")
        scaled = np.zeros_like(values)
        # Keep the ordinary arithmetic path unchanged. If finite coordinates
        # overflow during subtraction, halve both differences before dividing.
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            spread = upper - lower
            offset = values - lower
            unsafe = (~fixed) & (~np.isfinite(spread) | np.any(~np.isfinite(offset), axis=0))
            np.divide(offset, spread, out=scaled, where=~fixed & ~unsafe)
            if np.any(unsafe):
                halfSpread = upper[unsafe] * 0.5 - lower[unsafe] * 0.5
                scaled[:, unsafe] = (values[:, unsafe] * 0.5 - lower[unsafe] * 0.5) / halfSpread
        if not np.all(np.isfinite(scaled)):
            raise ValueError("DeltaTest scaled inputs must be finite.")
        return scaled

    def _cal_delta(self, X: np.ndarray, Y: np.ndarray, nNeighbors: int):
        """
        Calculate the Delta value using KDTree for nearest neighbor search.

        Args:
            X: Input data array.
            Y: Output data array.
            nNeighbors: Number of nearest neighbors to consider.

        Returns:
            The calculated delta value.
        """
        N, nInput = X.shape
        if nInput == 0:
            raise ValueError("DeltaTest requires at least one input column for neighbor search.")
        if (
            isinstance(nNeighbors, (bool, np.bool_))
            or not isinstance(nNeighbors, (int, np.integer))
            or not 1 <= nNeighbors < N
        ):
            raise ValueError("nNeighbors must be a positive integer smaller than the sample count.")

        # Build a KDTree for fast nearest neighbor search
        tree = KDTree(X)

        # One extra non-self neighbor detects a cutoff tie without enumerating
        # all pairwise distances. Roundoff-level differences count as ties.
        distances, neighbors_indices = tree.query(X, k=min(N, nNeighbors + 2))
        tieTolerance = 8 * np.finfo(float).eps * max(1, nInput)

        # With duplicate coordinates, the query row need not be the first hit.
        # Exclude by identity rather than dropping an arbitrary zero-distance hit.
        neighborRows = np.empty((N, nNeighbors), dtype=int)
        tiedRows = []
        for i, row in enumerate(neighbors_indices):
            nonSelf = row != i
            candidates = row[nonSelf]
            candidateDistances = distances[i, nonSelf]
            neighborRows[i] = candidates[:nNeighbors]
            if len(candidates) > nNeighbors:
                cutoff = candidateDistances[nNeighbors - 1]
                if candidateDistances[nNeighbors] - cutoff <= tieTolerance * cutoff:
                    tiedRows.append((i, cutoff))

        rowErrors = np.mean((Y[:, None, :] - Y[neighborRows]) ** 2, axis=(1, 2))
        duplicateStats = None
        if any(cutoff == 0 for _, cutoff in tiedRows):
            # Aggregate identical-coordinate groups once. Querying every member
            # of a large duplicate group would otherwise require quadratic work.
            _, groupIds, groupCounts = np.unique(X, axis=0, return_inverse=True, return_counts=True)
            groupMeans = np.zeros((len(groupCounts), Y.shape[1]))
            np.add.at(groupMeans, groupIds, Y)
            groupMeans /= groupCounts[:, None]
            groupVariances = np.zeros_like(groupMeans)
            np.add.at(groupVariances, groupIds, (Y - groupMeans[groupIds]) ** 2)
            groupVariances /= groupCounts[:, None]
            duplicateStats = groupIds, groupCounts, groupMeans, groupVariances

        for i, cutoff in tiedRows:
            if cutoff == 0:
                groupIds, groupCounts, groupMeans, groupVariances = duplicateStats
                group = groupIds[i]
                count = groupCounts[group]
                if count > nNeighbors + 1:
                    rowErrors[i] = (
                        count / (count - 1) * np.mean(groupVariances[group] + (Y[i] - groupMeans[group]) ** 2)
                    )
                    continue
            radius = np.nextafter(cutoff * (1 + tieTolerance), np.inf)
            candidates = np.asarray(tree.query_ball_point(X[i], radius), dtype=int)
            candidates = candidates[candidates != i]
            candidateDistances = np.linalg.norm(X[candidates] - X[i], axis=1)
            tied = np.isclose(candidateDistances, cutoff, rtol=tieTolerance, atol=0)
            closer = (candidateDistances < cutoff) & ~tied
            squared = np.mean((Y[i] - Y[candidates]) ** 2, axis=1)
            slots = nNeighbors - np.count_nonzero(closer)
            rowErrors[i] = (np.sum(squared[closer]) + slots * np.mean(squared[tied])) / nNeighbors

        # Delta is half the mean squared leave-one-out neighbor difference.
        # The neighbor average already divides by k; do not divide by k again.
        return 0.5 * float(np.mean(rowErrors))
