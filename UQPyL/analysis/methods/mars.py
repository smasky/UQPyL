import numpy as np
from typing import Optional, Tuple

from ..base import AnaIndex, AnalysisABC
from ...util import MinMaxScaler, Scaler
from ...problem import ProblemABC as Problem

from ...surrogate.mars import MARS as MARSModel

class MARS(AnalysisABC):
    """
    Multivariate Adaptive Regression Splines for Sensitivity Analysis
    Sensitivity analysis based on MARS surrogate refitting.
    
    Examples:
        >>> from UQPyL.doe import LHS
        >>> mars_method = MARS()
        >>> X = LHS('classic').sample(problem, 500)
        >>> Y = problem.evaluate(X, target="objs").objs
        >>> res = mars_method.analyze(problem, X, Y, target="objs")
        >>> print(res)
        
    References:
        [1] J. H. Friedman, Multivariate Adaptive Regression Splines, 
            The Annals of Statistics, vol. 19, no. 1, pp. 1-67, Mar. 1991, 
            doi: 10.1214/aos/1176347963.
        [2] SALib, https://github.com/SALib/SALib
    """
    
    name = "MARS"
    
    def __init__(self, scalers: Tuple[Optional[Scaler], Optional[Scaler]] = (None, None),
                 verboseFlag: bool = True, logFlag: bool = False, saveFlag: bool = False):
        """
        Initialize the MARS method.

        Args:
            scalers: Optional scalers for `X` and `Y`.
            verboseFlag: Whether to print compact runtime summaries.
            logFlag: Whether to write a log file.
            saveFlag: Whether to persist results to sqlite.
        """
        
        # Initialize the base class with provided scalers and flags
        super().__init__(scalers, verboseFlag, logFlag, saveFlag)
    
    def _analyzeCore(self, problem: Problem, X: np.ndarray, Y: Optional[np.ndarray] = None, meta: Optional[dict] = None,
                     target: str = 'objs', index: AnaIndex = 'all') -> None:
        """
        Run MARS-based sensitivity analysis on the provided samples.

        Args:
            problem: Analysis problem.
            X: Input sample matrix.
            Y: Optional output matrix corresponding to `X`.
            meta: Optional sampling metadata for persistence only.
            target: Semantic label of `Y`, typically `objs` or `cons`.
            index: Output column selection.
        """
        # Set the problem instance for analysis
        self.setProblem(problem)
        
        # Evaluate the problem if Y is not provided
        Y = self.check_Y(X, Y, target, index)
        numY = Y.shape[1]
        
        # Scale the input and output data if scalers are provided
        X, Y = self.__check_and_scale_xy__(X, Y)
        nInput = problem.nInput
        
        outputLabel = "obj" if target == "objs" else "con"
        
        S1 = np.zeros((numY, nInput))
        S1_norm = np.zeros((numY, nInput))
        row_label = [f"{outputLabel}{i+1}" for i in range(numY)]
        col_label_1 = problem.xLabels
        
        for i in range(numY):
            
            Y_i = Y[:, i:i+1]
        
            # Main process: Fit the MARS model and calculate sensitivity indices
            mars = MARSModel(scalers=(MinMaxScaler(0, 1), MinMaxScaler(0, 1)))
            mars.fit(X, Y_i)
            base_gcv = mars.gcv_

            # Calculate first-order sensitivity indices for each input variable
            for j in range(nInput):
                X_sub = np.delete(X, [j], axis=1)
                mars = MARSModel(scalers=(MinMaxScaler(0, 1), MinMaxScaler(0, 1)))
                mars.fit(X_sub, Y_i)
                S1[i, j] = np.abs(base_gcv - mars.gcv_)
            
            total = np.sum(S1[i])
            if np.isclose(total, 0.0):
                S1_norm[i] = 0.0
            else:
                S1_norm[i] = S1[i] / total
        
        res = [('S1', S1, row_label, col_label_1, 'decsDim1'), ('S1_norm', S1_norm, row_label, col_label_1, 'decsDim1')]
        
        X, Y = self.__reverse_X_Y__(X, Y)
        
        self.recordResult(X, Y, res, target=target, meta=meta)
        
        return None
