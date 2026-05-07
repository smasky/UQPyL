import numpy as np
from typing import Optional

from ..base import AnaIndex, AnalysisABC
from ...problem import ProblemABC as Problem

class Morris(AnalysisABC):
    """
    Morris Method for Sensitivity Analysis
    Screening-oriented sensitivity analysis based on elementary effects.

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

    def checkMeta(self, meta):
        if meta.get("designType") != "morris":
            raise ValueError(
                "Morris.analyze() requires Morris metadata with meta['designType'] == 'morris'."
            )

        self.set("numLevels", meta["numLevels"])
        
    def _analyzeCore(self, problem: Problem, X: np.ndarray, Y: Optional[np.ndarray] = None, meta: Optional[dict] = None,
                      target: str = 'objs', index: AnaIndex = 'all') -> None:
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

        numLevels = meta["numLevels"]
        
        # Set the problem instance for analysis
        self.setProblem(problem)
        
        Y = self.check_Y(X, Y, target, index)
        numY = Y.shape[1]
        
        nInput = problem.nInput
        
        trajectorySize = nInput + 1
        if X.shape[0] % trajectorySize != 0:
            raise ValueError(f"The number of samples must be divisible by {trajectorySize} for Morris analysis.")

        numTrajectory = int(X.shape[0] / trajectorySize)
        
        X, Y = self.__check_X_Y__(X, Y)

        outputLabel = "obj" if target == "objs" else "con"
        
        mu = np.zeros((numY, nInput))
        mu_star = np.zeros((numY, nInput))
        sigma = np.zeros((numY, nInput))
        S1_norm = np.zeros((numY, nInput))
        
        row_label = [f"{outputLabel}{i+1}" for i in range(numY)]
        col_label_1 = problem.xLabels
        
        for i in range(numY):
            
            Y_i = Y[:, i:i+1]
            
            # Initialize an array to store elementary effects
            EE = np.zeros((nInput, numTrajectory))

            # Calculate elementary effects for each trajectory
            for j in range(numTrajectory):
                X_sub = X[j * trajectorySize:(j + 1) * trajectorySize, :]
                Y_sub = Y_i[j * trajectorySize:(j + 1) * trajectorySize, :]

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
                ee = Y_diff / delta_diff
                EE[:, j:j+1] = ee[order]
        

            mu[i] = np.mean(EE, axis=1)
            mu_star[i] = np.mean(np.abs(EE), axis=1)
            sigma[i] = np.std(EE, axis=1, ddof=1)
            total = np.sum(mu_star[i])
            if np.isclose(total, 0.0):
                S1_norm[i] = 0.0
            else:
                S1_norm[i] = mu_star[i] / total
            
        res = [('mu', mu, row_label, col_label_1, 'decsDim1'), 
               ('mu_star', mu_star, row_label, col_label_1, 'decsDim1'), 
               ('sigma', sigma, row_label, col_label_1, 'decsDim1'), 
               ('S1_norm', S1_norm, row_label, col_label_1, 'decsDim1')]
        
        self.recordResult(X, Y, res, target=target, meta=meta)
        
        return None
    
