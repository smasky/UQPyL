# Fourier amplitude sensitivity test
import math
import numpy as np
from typing import Optional

from ..base import AnaIndex, AnalysisABC
from ._variance import scaleOutput
from ...problem import ProblemABC as Problem


class FAST(AnalysisABC):
    """
    Fourier Amplitude Sensitivity Test (FAST)
    Global sensitivity analysis based on Fourier amplitude decomposition.

    Examples:
        >>> fast_method = FAST()
        >>> from UQPyL.doe import FASTDesign
        >>> X, meta = FASTDesign(M=4).sampleWithMeta(problem, 500)
        >>> res = fast_method.analyze(problem, X, meta=meta, target="objs")
        >>> print(res)

    References:
        [1] Cukier et al., A Quantitative Model-Independent Method for Global Sensitivity Analysis of Model Output,
            Technometrics, 41(1):39-56, doi: 10.1063/1.1680571
        [2] A. Saltelli et al., A Quantitative Model-Independent Method for Global Sensitivity Analysis of Model Output,
            Technometrics, vol. 41, no. 1, pp. 39-56, Feb. 1999, doi: 10.1080/00401706.1999.10485594.
        [3] SALib, https://github.com/SALib/SALib
    """

    name = "FAST"

    def __init__(self, verboseFlag: bool = True, logFlag: bool = False, saveFlag: bool = False):
        """
        Initialize the FAST method.

        Args:
            verboseFlag: Whether to print compact runtime summaries.
            logFlag: Whether to write a log file.
            saveFlag: Whether to persist results to sqlite.
        """
        super().__init__(verboseFlag, logFlag, saveFlag)

    def checkMeta(self, meta):
        if meta.get("designType") != "fast":
            raise ValueError("FAST.analyze() requires FAST metadata with meta['designType'] == 'fast'.")

        M, N = meta.get("M"), meta.get("N")
        for name, value in (("M", M), ("N", N)):
            if not isinstance(value, (int, np.integer)) or isinstance(value, (bool, np.bool_)) or value <= 0:
                raise ValueError(f"FAST metadata {name} must be a positive integer.")
        M, N = int(M), int(N)
        if N <= 4 * M**2:
            raise ValueError(f"FAST requires N > 4*M^2; received N={N}, M={M}.")
        blockSize = meta.get("blockSize", N)
        if not isinstance(blockSize, (int, np.integer)) or isinstance(blockSize, (bool, np.bool_)) or blockSize != N:
            raise ValueError("FAST metadata blockSize must equal N.")
        self.set("M", M)

    @staticmethod
    def _computeOrders(outputs: np.ndarray, n: int, M: int, omega: int):
        """
        Compute first-order and total-order FAST indices for one output block.

        This implementation follows the same frequency partition used by SALib.
        """
        f = np.fft.fft(scaleOutput(outputs))
        Sp = np.power(np.absolute(f[np.arange(1, math.ceil(n / 2))]) / n, 2)

        V = 2.0 * np.sum(Sp)
        if V == 0.0:
            return 0.0, 0.0

        D1 = 2.0 * np.sum(Sp[np.arange(1, M + 1, dtype=np.int32) * omega - 1])
        Dt = 2.0 * np.sum(Sp[np.arange(math.floor(omega / 2.0), dtype=np.int32)])

        return float(D1 / V), float(1.0 - Dt / V)

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
        Run FAST on the provided samples.

        Args:
            problem: Analysis problem.
            X: Sample matrix generated for FAST.
            Y: Optional output matrix corresponding to `X`.
            meta: Sampling metadata from `FASTDesign.sampleWithMeta`.
            target: Semantic label of `Y`, typically `objs` or `cons`.
            index: Output column selection.
        """

        if meta is None:
            raise TypeError(
                "FAST.analyze() requires metadata. "
                "Use `X, meta = FASTDesign(...).sampleWithMeta(...)` or pass meta explicitly."
            )

        M = self.get("M")
        nInput = problem.nInput
        n = int(meta["N"])
        if isinstance(X, np.ndarray) and X.ndim == 2 and X.shape[0] != n * nInput:
            raise ValueError(f"FAST requires exactly {n * nInput} rows for N={n} and nInput={nInput}.")

        # Set the problem instance for analysis

        Y = self.check_Y(X, Y, target, index)
        numY = Y.shape[1]

        # Initialize arrays to store sensitivity indices

        # Calculate the base frequency
        omega0 = math.floor((n - 1) / (2 * M))

        S1 = np.zeros((numY, nInput))
        ST = np.zeros((numY, nInput))
        S1_norm = np.zeros((numY, nInput))
        ST_norm = np.zeros((numY, nInput))
        row_label = self.outputLabels
        col_label_1 = problem.xLabels

        for i in range(numY):
            Y_i = Y[:, i : i + 1]

            # Calculate sensitivity indices for each input variable
            for j in range(nInput):
                idx = np.arange(j * n, (j + 1) * n)
                Y_sub = Y_i[idx]
                S1[i, j], ST[i, j] = self._computeOrders(Y_sub.ravel(), n, M, omega0)

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

        res = [
            ("S1", S1, row_label, col_label_1, "decsDim1"),
            ("S1_norm", S1_norm, row_label, col_label_1, "decsDim1"),
            ("ST", ST, row_label, col_label_1, "decsDim1"),
            ("ST_norm", ST_norm, row_label, col_label_1, "decsDim1"),
        ]

        self.recordResult(X, Y, res, target=target, meta=meta)

        return None
