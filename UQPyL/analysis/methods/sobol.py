# Sobol sensitivity analysis
import numpy as np
import itertools
from typing import Optional

from ..base import AnaIndex, AnalysisABC
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
            raise ValueError(
                "Sobol.analyze() requires Saltelli metadata with meta['designType'] == 'saltelli'."
            )

        self.set("secondOrder", meta["secondOrder"])
   
                
    def _analyzeCore(self, problem: Problem, X: np.ndarray, Y: Optional[np.ndarray] = None, meta: Optional[dict] = None,
                      secondOrder: bool = True, target: str = 'objs', index: AnaIndex = 'all') -> None:
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
        self.setProblem(problem)

        if meta is None:
            raise TypeError(
                "Sobol.analyze() requires metadata. "
                "Use `X, meta = SaltelliDesign(...).sampleWithMeta(...)` or pass meta explicitly."
            )

        secondOrder = meta["secondOrder"]

        # If Y is not provided, evaluate the problem to obtain Y
        Y = self.check_Y(X, Y, target, index)
        
        numY = Y.shape[1]
        
        nInput = problem.nInput
        
        X, Y = self.__check_X_Y__(X, Y)
        
        # Determine the number of samples based on whether second-order indices are calculated
        if secondOrder:
            if X.shape[0] % (2 * nInput + 2) != 0:
                raise ValueError(f"The number of samples must be divisible by {2 * nInput + 2}!")
            n = int(X.shape[0] / (2 * nInput + 2))
        else:
            if X.shape[0] % (nInput + 2) != 0:
                raise ValueError(f"The number of samples must be divisible by {nInput + 2}!")
            n = int(X.shape[0] / (nInput + 2))
        
        outputLabel = "obj" if target == 'objs' else "con"
        
        
        S1 = np.zeros((numY, nInput))
        ST = np.zeros((numY, nInput))
        S1_norm = np.zeros((numY, nInput))
        ST_norm = np.zeros((numY, nInput))
        
        row_label = [f"{outputLabel}{i+1}" for i in range(numY)]
        col_label_1 = problem.xLabels
        
        if secondOrder:
            col_label_2 = [f"{a}-{b}" for a, b in itertools.combinations(problem.xLabels, 2)]
            S2 = np.zeros((numY, len(col_label_2)))

        for i in range(numY):
            
            Y_i = Y[:, i:i+1]
            yStd = float(np.std(Y_i))
            if np.isclose(yStd, 0.0):
                continue

            Y_i = (Y_i - np.mean(Y_i)) / yStd
            
            # Separate the output values into different arrays for analysis
            A, B, AB, BA = self._separateOutputValues(Y_i, nInput, n, secondOrder)
        
            # Calculate first-order and total-order sensitivity indices for each input variable
            for j in range(nInput):
                S1[i, j] = self._firstOrder(A, AB[:, j:j + 1], B)
                ST[i, j] = self._totalOrder(A, AB[:, j:j + 1], B)

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
                        S2[i, pairIndex] = self._secondOrder(A, AB[:, j:j + 1], AB[:, k:k + 1], BA[:, j:j + 1], B)
                        pairIndex += 1
        
        res = [
            ('S1', S1, row_label, col_label_1, 'decsDim1'),
            ('S1_norm', S1_norm, row_label, col_label_1, 'decsDim1'),
            ('ST', ST, row_label, col_label_1, 'decsDim1'),
            ('ST_norm', ST_norm, row_label, col_label_1, 'decsDim1'),
        ]
        if secondOrder:
            res.append(('S2', S2, row_label, col_label_2, 'decsDim2'))
         
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
        B = Y[(step - 1):total:step, :]
        
        for j in range(d):
            AB[:, j] = Y[(j + 1):total:step, 0]
            
            if calSecondOrder:
                BA[:, j] = Y[(j + 1 + d):total:step, 0]
        
        return A, B, AB, BA
