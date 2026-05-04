import numpy as np
from typing import Optional, Tuple
from scipy.stats import cramervonmises_2samp

from ..base import AnaIndex, AnalysisABC
from ...problem import ProblemABC as Problem
from ...util import Scaler

class RSA(AnalysisABC):
    """
    Regional Sensitivity Analysis (RSA)
    Sensitivity analysis based on regional output partitioning.

    Examples:
        >>> from UQPyL.doe import LHS
        >>> rsa_method = RSA(nRegion=20)
        >>> X = LHS('classic').sample(problem, 500)
        >>> Y = problem.evaluate(X, target="objs").objs
        >>> res = rsa_method.analyze(problem, X, Y, target="objs")

    References:
        [1] F. Pianosi et al., Sensitivity analysis of environmental models: A systematic review with practical workflow, 
            Environmental Modelling & Software, vol. 79, pp. 214-232, May 2016, 
            doi: 10.1016/j.envsoft.2016.02.008.
        [2] SALib, https://github.com/SALib/SALib
    """
    
    name = "RSA"
    
    def __init__(self, scalers: Tuple[Optional[Scaler], Optional[Scaler]] = (None, None),
                 nRegion: int = 20,
                 verboseFlag: bool = True, logFlag: bool = False, saveFlag: bool = False):
        """
        Initialize the RSA method for sensitivity analysis.

        Args:
            scalers: Optional scalers for `X` and `Y`.
            nRegion: Number of output regions used by RSA.
            verboseFlag: Whether to print compact runtime summaries.
            logFlag: Whether to write a log file.
            saveFlag: Whether to persist results to sqlite.
        """
        
        # Initialize the base class with provided scalers and flags
        super().__init__(scalers, verboseFlag, logFlag, saveFlag)
        self.setParaValue("nRegion", nRegion)

    def _analyzeCore(self, problem: Problem, X: np.ndarray, Y: Optional[np.ndarray] = None, meta: Optional[dict] = None,
                     target: str = 'objs', index: AnaIndex = 'all') -> None:
        """
        Run RSA on the provided samples.

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
        
        nInput = problem.nInput
        
        # Evaluate the problem if Y is not provided
        Y = self.check_Y(X, Y, target, index)
        
        # Scale the input and output data if scalers are provided
        X, Y = self.__check_and_scale_xy__(X, Y)
        
        numY = Y.shape[1]
        nRegion = self.getParaValue("nRegion")
        outputLabel = "obj" if target == "objs" else "con"
        
        S1 = np.zeros((numY, nInput))
        S1_norm = np.zeros((numY, nInput))
        
        row_label = [f"{outputLabel}{i+1}" for i in range(numY)]
        col_label_1 = problem.xLabels
        
        for i in range(numY):
            
            Y_i = Y[:, i:i+1]
        
            # Define the sequence for dividing the input space into regions
            seq = np.linspace(0.0, 1.0, nRegion + 1)
            results = np.full((nRegion, nInput), np.nan)
            X_di = np.empty(X.shape[0])
            
            trr = Y_i.ravel()
            mrr = X_di
            
            # Loop over each input dimension to perform RSA
            for d_i in range(nInput):
                X_di[:] = X[:, d_i]
                
                # Calculate quantiles for dividing the output space
                quants = np.quantile(trr, seq)
                
                # Perform analysis for each region
                b = (quants[0] <= trr) & (trr <= quants[1])
                if self._has_samples(Y_i, b):
                    results[0, d_i] = cramervonmises_2samp(mrr[b].ravel(), mrr[~b].ravel()).statistic
                
                for bin_index in range(1, nRegion):
                    
                    b = (quants[bin_index] < trr) & (trr <= quants[bin_index+1])
                    
                    if self._has_samples(Y_i, b):
                        results[bin_index, d_i] = cramervonmises_2samp(mrr[b].ravel(), mrr[~b].ravel()).statistic
        
            # Calculate the mean sensitivity index for each input factor
            validCounts = np.sum(~np.isnan(results), axis=0)
            sums = np.nansum(results, axis=0)
            results_star = np.divide(
                sums,
                validCounts,
                out=np.zeros_like(sums),
                where=validCounts > 0,
            )
            
            S1[i] = results_star
            total = np.sum(results_star)
            if np.isclose(total, 0.0):
                S1_norm[i] = 0.0
            else:
                S1_norm[i] = results_star / total
        
        res = [('S1', S1, row_label, col_label_1, 'decsDim1'), ('S1_norm', S1_norm, row_label, col_label_1, 'decsDim1')]
        
        X, Y = self.__reverse_X_Y__(X, Y)
        
        self.recordResult(X, Y, res, target=target, meta=meta)
        
        return None
    
    def _has_samples(self, y, sel):
        """
        Check if the selected samples are sufficient for analysis.

        This helper ensures that the selected group is non-empty and
        retains enough variation for the two-sample statistic.

        Args:
            y: Output values for one analyzed target.
            sel: Boolean mask of the selected region.

        Returns:
            Whether the region has enough samples for RSA.
        """
        return (
            (np.count_nonzero(sel) != 0)
            and (len(y[~sel]) != 0)
            and np.unique(y[sel]).size > 1
        )
