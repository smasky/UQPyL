# Sobol sensitivity analysis
import numpy as np
import itertools
import warnings
from typing import Optional

from ..base import AnaIndex, AnalysisABC
from ._variance import scaleOutput
from ...problem import ProblemABC as Problem


class Sobol(AnalysisABC):
    """
    Sobol' Sensitivity Analysis
    Variance-based global sensitivity analysis with first, total, and optional second-order indices.

    Examples:
        >>> from UQPyL.doe import SaltelliDesign
        >>> sob_method = Sobol()
        >>> X, meta = SaltelliDesign(secondOrder=True).sampleWithMeta(problem, 512)
        >>> Y = problem.evaluate(X, target="objs").objs
        >>> sob_method.analyze(problem, X, Y, meta=meta, target="objs")

    References:
        [1] I. M. Sobol', Global sensitivity indices for nonlinear mathematical models and their Monte Carlo estimates,
            Mathematics and Computers in Simulation, vol. 55, no. 1, pp. 271–280, Feb. 2001,
            doi: 10.1016/S0378-4754(00)00270-6.
        [2] A. Saltelli et al, Variance based sensitivity analysis of model output. Design and estimator for the total sensitivity index,
            Computer Physics Communications, vol. 181, no. 2, pp. 259–270, Feb. 2010,
            doi: 10.1016/j.cpc.2009.09.018.
        [3] SALib, https://github.com/SALib/SALib
    """

    name = "Sobol"

    def __init__(self, verboseFlag: bool = True, logFlag: bool = False, saveFlag: bool = False):
        """
        Initialize the Sobol' method for sensitivity analysis.

        Args:
            verboseFlag: Whether to print compact runtime summaries.
            logFlag: Whether to write a log file.
            saveFlag: Whether to persist results to sqlite.
        """
        super().__init__(verboseFlag, logFlag, saveFlag)

    def checkMeta(self, meta):
        if meta.get("designType") != "saltelli":
            raise ValueError("Sobol.analyze() requires Saltelli metadata with meta['designType'] == 'saltelli'.")

    @staticmethod
    def _isPositiveInteger(value):
        return isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_)) and value > 0

    @staticmethod
    def _matchesDesign(X, secondOrder):
        """Check copied hybrid coordinates without dropping or reordering rows."""
        nInput = X.shape[1]
        blockSize = 2 * nInput + 2 if secondOrder else nInput + 2
        if len(X) == 0 or len(X) % blockSize != 0:
            return False
        blocks = X.reshape(-1, blockSize, nInput)
        A, B = blocks[:, 0], blocks[:, -1]
        for j in range(nInput):
            AB = blocks[:, j + 1]
            if not (
                np.array_equal(AB[:, j], B[:, j])
                and np.array_equal(AB[:, :j], A[:, :j])
                and np.array_equal(AB[:, j + 1 :], A[:, j + 1 :])
            ):
                return False
            if secondOrder:
                BA = blocks[:, nInput + j + 1]
                if not (
                    np.array_equal(BA[:, j], A[:, j])
                    and np.array_equal(BA[:, :j], B[:, :j])
                    and np.array_equal(BA[:, j + 1 :], B[:, j + 1 :])
                ):
                    return False
        return True

    def _resolveDesign(self, X, meta):
        secondOrder = meta.get("secondOrder")
        n = meta.get("N")
        blockSize = meta.get("blockSize")
        validOrder = isinstance(secondOrder, (bool, np.bool_))
        validN = self._isPositiveInteger(n)
        validBlock = self._isPositiveInteger(blockSize)
        issues = []
        if not validOrder:
            issues.append("metadata secondOrder must be a boolean")
        if not validN:
            issues.append("metadata N must be a positive integer")
        if "blockSize" in meta and not validBlock:
            issues.append("metadata blockSize must be a positive integer")
        if validOrder:
            expectedBlock = 2 * X.shape[1] + 2 if secondOrder else X.shape[1] + 2
            if validBlock and blockSize != expectedBlock:
                issues.append(f"metadata blockSize={blockSize} conflicts with secondOrder={bool(secondOrder)}")
            if validN and len(X) != int(n) * expectedBlock:
                issues.append(
                    f"N={n} and secondOrder={bool(secondOrder)} require {int(n) * expectedBlock} rows; received {len(X)} rows"
                )

        chosen = None
        if not issues:
            chosen = (bool(secondOrder), int(n), expectedBlock)
        else:
            candidates = []
            for candidateOrder in (False, True):
                candidateBlock = 2 * X.shape[1] + 2 if candidateOrder else X.shape[1] + 2
                if self._matchesDesign(X, candidateOrder):
                    candidates.append((candidateOrder, len(X) // candidateBlock, candidateBlock))
            if len(candidates) == 1:
                chosen = candidates[0]
            elif len(candidates) > 1:
                # Use consistent explicit layout fields only when the copied
                # coordinates support more than one complete layout.
                matching = [
                    candidate
                    for candidate in candidates
                    if (not validOrder or candidate[0] == bool(secondOrder))
                    and (not validBlock or candidate[2] == int(blockSize))
                ]
                if len(matching) > 1 and validN:
                    matching = [candidate for candidate in matching if candidate[1] == int(n)]
                if len(matching) == 1:
                    chosen = matching[0]

        diagnostic = dict(
            status="not_estimated" if chosen is None else ("recovered" if issues else "validated"),
            n_samples=len(X),
            effective_n=None if chosen is None else chosen[1],
            effective_block_size=None if chosen is None else chosen[2],
            effective_second_order=None if chosen is None else chosen[0],
            metrics_available=chosen is not None,
            recovery_basis="none" if chosen is None else ("sample_structure" if issues else "metadata"),
            issues=issues,
        )
        self.set("secondOrder", diagnostic["effective_second_order"])
        if issues:
            outcome = (
                f"Using complete sample blocks with N={chosen[1]}, blockSize={chosen[2]}, secondOrder={chosen[0]}."
                if chosen is not None
                else "Cannot determine a complete Saltelli layout. Returning zero placeholders marked not_estimated; "
                "no sensitivity estimates were computed."
            )
            warnings.warn(
                f"Sobol sampling metadata is inconsistent: {'; '.join(issues)}. {outcome}", RuntimeWarning, stacklevel=3
            )
        return diagnostic

    def _recordUnavailable(self, X, Y, meta, target, index):
        if target not in ("objs", "cons"):
            raise ValueError("Target must be 'objs' or 'cons'.")
        # Resolve selected labels without evaluating a model for unusable rows.
        labelValues = np.zeros((len(X), self.problem.nObj if target == "objs" else self.problem.nCons))
        selectedY = self.check_Y(X, labelValues if Y is None else Y, target, index)
        if Y is not None and not np.all(np.isfinite(selectedY)):
            raise ValueError("Sobol requires finite output values.")
        res = [
            (
                name,
                np.zeros((selectedY.shape[1], self.problem.nInput)),
                self.outputLabels,
                self.problem.xLabels,
                "decsDim1",
            )
            for name in ("S1", "S1_norm", "ST", "ST_norm")
        ]
        declaredOrder = meta.get("secondOrder")
        if isinstance(declaredOrder, (bool, np.bool_)) and declaredOrder:
            pairLabels = [f"{a}-{b}" for a, b in itertools.combinations(self.problem.xLabels, 2)]
            res.append(
                ("S2", np.zeros((selectedY.shape[1], len(pairLabels))), self.outputLabels, pairLabels, "decsDim2")
            )
        self.recordResult(X, None if Y is None else selectedY, res, target=target, meta=meta)

    def _analyzeCore(
        self,
        problem: Problem,
        X: np.ndarray,
        Y: Optional[np.ndarray] = None,
        meta: Optional[dict] = None,
        secondOrder: bool = True,
        target: str = "objs",
        index: AnaIndex = "all",
    ) -> None:
        """
        Run Sobol' analysis on Saltelli samples.

        Args:
            problem: Analysis problem.
            X: Sample matrix generated for Saltelli/Sobol analysis.
            Y: Optional output matrix corresponding to `X`.
            meta: Sampling metadata from `SaltelliDesign.sampleWithMeta`.
            secondOrder: Placeholder argument; actual behavior is driven by `meta`.
            target: Semantic label of `Y`, typically `objs` or `cons`.
            index: Output column selection.
        """

        # Set the problem instance for the analysis

        if meta is None:
            raise TypeError(
                "Sobol.analyze() requires metadata. "
                "Use `X, meta = SaltelliDesign(...).sampleWithMeta(...)` or pass meta explicitly."
            )

        if not isinstance(X, np.ndarray):
            raise TypeError("X must be an instance of np.ndarray!")
        if X.ndim != 2 or X.shape[1] != problem.nInput:
            raise ValueError("X must have shape (nSamples, nInput).")
        diagnostic = self._resolveDesign(X, meta)
        self.state.extra["sobol_design"] = diagnostic
        if not diagnostic["metrics_available"]:
            self._recordUnavailable(X, Y, meta, target, index)
            return None

        secondOrder = diagnostic["effective_second_order"]
        nInput = problem.nInput
        n = diagnostic["effective_n"]

        # If Y is not provided, evaluate the problem to obtain Y
        Y = self.check_Y(X, Y, target, index)

        numY = Y.shape[1]

        S1 = np.zeros((numY, nInput))
        ST = np.zeros((numY, nInput))
        S1_norm = np.zeros((numY, nInput))
        ST_norm = np.zeros((numY, nInput))

        row_label = self.outputLabels
        col_label_1 = problem.xLabels

        if secondOrder:
            col_label_2 = [f"{a}-{b}" for a, b in itertools.combinations(problem.xLabels, 2)]
            S2 = np.zeros((numY, len(col_label_2)))

        for i in range(numY):
            Y_i = scaleOutput(Y[:, i : i + 1])
            yStd = float(np.std(Y_i))
            if yStd == 0.0:
                continue

            Y_i = (Y_i - np.mean(Y_i)) / yStd

            # Separate the output values into different arrays for analysis
            A, B, AB, BA = self._separateOutputValues(Y_i, nInput, n, secondOrder)
            if float(np.var(np.r_[A, B])) == 0.0:
                raise ValueError(
                    f"Sobol output {self.outputLabels[i]!r} has zero variance in base A/B samples "
                    "while hybrid samples vary; increase the base sample size."
                )

            # Calculate first-order and total-order sensitivity indices for each input variable
            for j in range(nInput):
                S1[i, j] = self._firstOrder(A, AB[:, j : j + 1], B)
                ST[i, j] = self._totalOrder(A, AB[:, j : j + 1], B)

            s1Total = np.sum(S1[i])
            if np.isclose(s1Total, 0.0):
                S1_norm[i] = 0.0
            else:
                S1_norm[i] = S1[i] / s1Total

            stTotal = np.sum(ST[i])
            if np.isclose(stTotal, 0.0):
                ST_norm[i] = 0.0
            else:
                ST_norm[i] = ST[i] / stTotal

            if secondOrder:
                # Calculate second-order sensitivity indices for each pair of input variables
                pairIndex = 0
                for j in range(nInput):
                    for k in range(j + 1, nInput):
                        S2[i, pairIndex] = self._secondOrder(A, AB[:, j : j + 1], AB[:, k : k + 1], BA[:, j : j + 1], B)
                        pairIndex += 1

        res = [
            ("S1", S1, row_label, col_label_1, "decsDim1"),
            ("S1_norm", S1_norm, row_label, col_label_1, "decsDim1"),
            ("ST", ST, row_label, col_label_1, "decsDim1"),
            ("ST_norm", ST_norm, row_label, col_label_1, "decsDim1"),
        ]
        if secondOrder:
            res.append(("S2", S2, row_label, col_label_2, "decsDim2"))

        self.recordResult(X, Y, res, target=target, meta=meta)

        return None

    def _secondOrder(self, A, AB1, AB2, BA, B):
        """
        Calculate the second-order sensitivity index.

        Args:
            A: Output values for the base sample A.
            AB1: Output values for the first hybrid sample.
            AB2: Output values for the second hybrid sample.
            BA: Output values for the reverse hybrid sample.
            B: Output values for the base sample B.

        Returns:
            The second-order sensitivity index.
        """
        Y = np.r_[A, B]

        Vjk = float(np.mean(BA * AB2 - A * B, axis=0).item() / np.var(Y, axis=0).item())
        Sj = self._firstOrder(A, AB1, B)
        Sk = self._firstOrder(A, AB2, B)

        return Vjk - Sj - Sk

    def _firstOrder(self, A, AB, B):
        """
        Calculate the first-order sensitivity index.

        Args:
            A: Output values for the base sample A.
            AB: Output values for the hybrid sample.
            B: Output values for the base sample B.

        Returns:
            The first-order sensitivity index.
        """
        Y = np.r_[A, B]

        return float(np.mean(B * (AB - A), axis=0).item() / np.var(Y, axis=0).item())

    def _totalOrder(self, A, AB, B):
        """
        Calculate the total-order sensitivity index.

        Args:
            A: Output values for the base sample A.
            AB: Output values for the hybrid sample.
            B: Output values for the base sample B.

        Returns:
            The total-order sensitivity index.
        """
        Y = np.r_[A, B]

        return float(0.5 * np.mean((A - AB) ** 2, axis=0).item() / np.var(Y, axis=0).item())

    def _separateOutputValues(self, Y, d, n, calSecondOrder):
        """
        Separate the output values into different arrays for analysis.

        Args:
            Y: Output vector for a single analyzed target.
            d: Number of input variables.
            n: Base sample size.
            calSecondOrder: Whether second-order blocks are present.

        Returns:
            The separated A, B, AB, and BA blocks.
        """
        AB = np.zeros((n, d))
        BA = np.zeros((n, d)) if calSecondOrder else None

        step = 2 * d + 2 if calSecondOrder else d + 2

        total = Y.shape[0]

        A = Y[0:total:step, :]
        B = Y[(step - 1) : total : step, :]

        for j in range(d):
            AB[:, j] = Y[(j + 1) : total : step, 0]

            if calSecondOrder:
                BA[:, j] = Y[(j + 1 + d) : total : step, 0]

        return A, B, AB, BA
