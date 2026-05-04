import numpy as np
from scipy.signal import periodogram
from typing import Optional, Tuple

from ..base import AnaIndex, AnalysisABC
from ...problem import ProblemABC as Problem
from ...util import Scaler

class RBDFAST(AnalysisABC):
    """
    Random Balance Designs Fourier Amplitude Sensitivity Test (RBD-FAST)
    First-order global sensitivity analysis using random balance designs.

    Examples:
        >>> from UQPyL.doe import LHS
        >>> rbd_method = RBDFAST(M=4)
        >>> X = LHS('classic').sample(problem, 500)
        >>> Y = problem.evaluate(X, target="objs").objs
        >>> rbd_method.analyze(problem, X, Y, target="objs")

    References:
        [1] S. Tarantola et al, Random balance designs for the estimation of first order global sensitivity indices, 
            Reliability Engineering & System Safety, vol. 91, no. 6, pp. 717-727, Jun. 2006,
            doi: 10.1016/j.ress.2005.06.003.
        [2] J.-Y. Tissot and C. Prieur, Bias correction for the estimation of sensitivity indices based on random balance designs,
            Reliability Engineering & System Safety, vol. 107, pp. 205-213, Nov. 2012, 
            doi: 10.1016/j.ress.2012.06.010.
    """
    
    name = "RBDFAST"
    
    def __init__(self, scalers: Tuple[Optional[Scaler], Optional[Scaler]] = (None, None), 
                 M: int = 4, 
                 verboseFlag: bool = True, logFlag: bool = False, saveFlag: bool = False):
        """
        Initialize the RBD-FAST method for global sensitivity analysis.

        Args:
            scalers: Optional scalers for `X` and `Y`.
            M: Number of harmonics used in the periodogram estimate.
            verboseFlag: Whether to print compact runtime summaries.
            logFlag: Whether to write a log file.
            saveFlag: Whether to persist results to sqlite.
        """
        # Attribute indicating the types of sensitivity indices calculated
        self.firstOrder = True
        self.secondOrder = False
        self.totalOrder = False
        
        # Initialize the base class with provided scalers and flags
        super().__init__(scalers, verboseFlag, logFlag, saveFlag)
        
        # Set the parameter for the number of harmonics
        self.setParaValue("M", M)
    
    def _analyzeCore(self, problem: Problem, X: np.ndarray, Y: Optional[np.ndarray] = None, meta: Optional[dict] = None,
                     target: str = 'objs', index: AnaIndex = 'all') -> None:
        """
        Run RBD-FAST on the provided samples.

        Args:
            problem: Analysis problem.
            X: Input sample matrix.
            Y: Optional output matrix corresponding to `X`.
            meta: Optional sampling metadata for persistence only.
            target: Semantic label of `Y`, typically `objs` or `cons`.
            index: Output column selection.
        """
        # Retrieve the parameter for the number of harmonics
        M = self.getParaValue('M')
        
        # Set the problem instance for analysis
        self.setProblem(problem)
        
        nInput = problem.nInput
        
        # Evaluate the problem if Y is not provided
        Y = self.check_Y(X, Y, target, index)
        
        # Scale the input and output data if scalers are provided
        X, Y = self.__check_and_scale_xy__(X, Y)
        
        numY = Y.shape[1]
        outputLabel = "obj" if target == "objs" else "con"
        
        # Initialize an array to store first-order sensitivity indices
        
        S1 = np.zeros((numY, nInput))
        row_label = [f"{outputLabel}{i+1}" for i in range(numY)]
        col_label_1 = problem.xLabels
        
        for i in range(numY):
            
            Y_i = Y[:, i:i+1]
            
            # Calculate sensitivity indices for each input variable
            for j in range(nInput):
                idx = np.argsort(X[:, j])
                idx = np.concatenate([idx[::2], idx[1::2][::-1]])
                Y_seq = Y_i[idx]
                
                # Perform periodogram analysis
                _, Pxx = periodogram(Y_seq.ravel())
                V = np.sum(Pxx[1:])
                if np.isclose(V, 0.0):
                    S1[i, j] = 0.0
                    continue
                D1 = np.sum(Pxx[1: M+1])
                S1_sub = D1 / V
                
                # Normalization
                lamb = (2 * M) / Y.shape[0]
                if np.isclose(1 - lamb, 0.0):
                    S1[i, j] = S1_sub
                    continue
                S1_sub = S1_sub - lamb / (1 - lamb) * (1 - S1_sub)
                S1_sub = float(np.clip(S1_sub, 0.0, 1.0))
                
                S1[i, j] = S1_sub
        
        res = [('S1', S1, row_label, col_label_1, 'decsDim1')]
        
        X, Y = self.__reverse_X_Y__(X, Y)
        
        self.recordResult(X, Y, res, target=target, meta=meta)
        
        return None
