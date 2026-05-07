# Delta test
import numpy as np
from scipy.spatial import KDTree
from typing import Optional

from ..base import AnaIndex, AnalysisABC
from ...problem import ProblemABC, Problem

class DeltaTest(AnalysisABC):
    """
    Delta Test
    Non-parametric sensitivity analysis based on nearest-neighbor prediction error.
    
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
    
    def __init__(self, nNeighbors: int = 2,
                 verboseFlag: bool = True, logFlag: bool = False, saveFlag: bool = False):
        """
        Initialize the Delta Test method.

        Args:
            nNeighbors: Number of nearest neighbors used by the delta estimate.
            verboseFlag: Whether to print compact runtime summaries.
            logFlag: Whether to write a log file.
            saveFlag: Whether to persist results to sqlite.
        """
        super().__init__(verboseFlag, logFlag, saveFlag)
        self.set("nNeighbors", nNeighbors)

    def _analyzeCore(self, problem, X: np.ndarray, Y: Optional[np.ndarray] = None, meta: Optional[dict] = None,
                     target: str = 'objs', index: AnaIndex = 'all') -> None:
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
        self.setProblem(problem)
        
        # Evaluate the problem if Y is not provided
        Y = self.check_Y(X, Y, target, index)
        
        X, Y = self.__check_X_Y__(X, Y)
        nInput = problem.nInput
        numY = Y.shape[1]
        nNeighbors = self.get("nNeighbors")
        
        outputLabel = "obj" if target == "objs" else "con"
        
        S1 = np.zeros((numY, nInput))
        S1_norm = np.zeros((numY, nInput))
        row_label = [f"{outputLabel}{i+1}" for i in range(numY)]
        col_label_1 = problem.xLabels
        
        for i in range(numY):
            Y_i = Y[:, i:i+1]
            base = self._cal_delta(X, Y_i, nNeighbors)
            for j in range(nInput):
                XSub = np.delete(X, [j], axis=1)
                deltaWithoutVar = self._cal_delta(XSub, Y_i, nNeighbors)
                S1[i, j] = deltaWithoutVar - base

            total = np.sum(S1[i])
            if np.isclose(total, 0.0):
                S1_norm[i] = 0.0
            else:
                S1_norm[i] = S1[i] / total

        res = [('S1', S1, row_label, col_label_1, 'decsDim1'), ('S1_norm', S1_norm, row_label, col_label_1, 'decsDim1')]
        
        self.recordResult(X, Y, res, target=target, meta=meta)

        return None
    
    def findCombEA(self, problem, X: np.ndarray, Y: Optional[np.ndarray] = None, 
                   FEs: int = 10000, 
                   verboseFlag: bool = True, saveFlag: bool = True):
        """
        Find the best combination using Evolutionary Algorithm.

        Args:
            problem: Analysis problem.
            X: Input sample matrix.
            Y: Optional output matrix corresponding to `X`.
            FEs: Maximum number of function evaluations.
            verboseFlag: Whether the helper GA should print progress.
            saveFlag: Whether the helper GA should persist results.

        Returns:
            The optimization result returned by the configured GA.
        """
        from ...optimization.soea import GA
        
        # Set the problem instance for analysis
        self.setProblem(problem)
        
        # Retrieve the number of nearest neighbors for analysis
        nNeighbors = self.get('nNeighbors')
        
        # Evaluate outputs if Y is not provided
        if Y is None:
            Y = self.evaluate(X, target="objs")
        
        X, Y = self.__check_X_Y__(X, Y)
        
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
            XSub = X[:, Indices]
            
            if np.sum(x_) == 0:
                return np.inf
            else:
                return self._cal_delta(XSub, Y, nNeighbors)
        
        # Create the optimization problem
        nInput = problem.nInput
        nObj = 1
        ub = [1] * nInput
        lb = [0] * nInput
        varType = [1] * nInput
        
        problem = Problem(nInput=nInput, nObj=nObj, ub=ub, lb=lb, 
                          varType=varType, objFunc=objective, optType='min')
        
        # Initialize the GA
        ga = GA(maxFEs=FEs, verboseFlag=verboseFlag, saveFlag=saveFlag)
        
        # Run the GA
        res = ga.run(problem)
        
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
        nNeighbors = self.get('nNeighbors')
        
        # Evaluate outputs if Y is not provided
        if Y is None:
            Y = self.evaluate(X, target="objs")
        
        X, Y = self.__check_X_Y__(X, Y)
        
        # Generate all possible combinations of input variables
        combinations = list(product([0, 1], repeat=nInput))
        
        # Initialize an array to store objective values for each combination
        objs = np.zeros((len(combinations), 1))
        
        # Evaluate each combination
        for i in range(len(combinations)):
            x_ = np.array(combinations[i])
            Indices = np.where(x_ == 1)[0]
            XSub = X[:, Indices]
            
            if np.sum(x_) == 0:
                objs[i] = np.inf
            else:
                objs[i] = self._cal_delta(XSub, Y, nNeighbors)
        
        # Find the best combination based on the objective values
        best_index = np.argmin(objs)
        best_combination = combinations[best_index]
        
        # Return the labels of the most sensitive variables
        return [problem.xLabels[i] for i in range(nInput) if best_combination[i] == 1]
    
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
        N, _ = X.shape
        
        # Build a KDTree for fast nearest neighbor search
        tree = KDTree(X)
        
        # Query the nearest neighbors for each point
        _, neighbors_indices = tree.query(X, k=nNeighbors + 1)  # +1 to include the point itself
        
        # Exclude the point itself from the neighbors
        neighbors_indices = neighbors_indices[:, 1:]
        
        Delta = 0
        for i in range(N):
            d = (Y[i] - Y[neighbors_indices[i]])**2
            Delta += float(np.mean(d))
        
        return Delta / (nNeighbors * N)     
