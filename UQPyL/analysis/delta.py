# Delta test
import numpy as np
from scipy.spatial import KDTree
from typing import Optional, Tuple

from .base import AnalysisABC
from ..doe import LHS, Sampler
from ..problem import ProblemABC, Problem
from ..util import Scaler, Verbose
from ..optimization.soea import GA

class DeltaTest(AnalysisABC):
    """
    -------------------------------------------------
    Delta Test
    -------------------------------------------------
    This class implements the Delta Test, which is 
    a non-parametric method for sensitivity analysis.
    
    Methods:
        sample: Generate a sample for Delta Test analysis
        analyze: Perform Delta Test analysis from the X and Y you provided.
        findCombEA: Find the best combination using Evolutionary Algorithm.
        findCombVio: Find the best combination using brute-force approach.
    
    Examples:
        # `problem` is an instance of ProblemABC or Problem from UQPyL.problems
        # You must create a problem instance before using this method.
        >>> delta_method = Delta_Test(nNeighbors=2)
        >>> X = delta_method.sample(problem, N=1000)
        >>> res = delta_method.analyze(problem, X)
        >>> print(res)
        
    References:
        [1] E. Eirola et al, Using the Delta Test for Variable Selection, 
            Artificial Neural Networks, 2008.
        [2] SALib, https://github.com/SALib/SALib
    -------------------------------------------------
    """
    
    name = "DeltaTest"
    
    def __init__(self, scalers: Tuple[Optional[Scaler], Optional[Scaler]] = (None, None), 
                 verboseFlag: bool = True, logFlag: bool = False, saveFlag: bool = False):
        """
        Initialize the Delta Test method.
        ------------------------------------------------------------
        :param scalers: Tuple[Optional[Scaler], Optional[Scaler]] - Tuple containing scalers for input (X) and output (Y) data. Defaults to (None, None).
        :param nNeighbors: int - The number of nearest neighbors used in Delta Test estimation. Defaults to 2.
        :param verboseFlag: bool - If True, enables verbose mode for logging. Defaults to False.
        :param logFlag: bool - If True, saves logging to a file. Defaults to False.
        :param saveFlag: bool - If True, saves the results to a file. Defaults to False.
        """
        
        # Initialize the base class with provided scalers and flags
        super().__init__(scalers, verboseFlag, logFlag, saveFlag)

        
    def sample(self, problem: Problem, N: int = 500, sampler: Sampler = LHS('classic'), seed: Optional[int] = None):
        """
        Generate a sample set for the Delta Test.
        --------------------------------------------------
        :param problem: Problem - The problem instance defining the input space.
        :param N: int - The number of samples to generate. Defaults to 500.
        :param sampler: Sampler - The sampling method to use. Defaults to Latin Hypercube Sampling (LHS) with 'classic' mode.

        :return: np.ndarray - A 2D array of shape `(N, nInput)`, where `nInput` is the number of input variables.
        """
        
        self.rng = np.random.default_rng(seed) if seed is not None else np.random.default_rng()
        
        nInput = problem.nInput
        
        # Generate samples using the specified sampler
        sampler_seed = self.rng.integers(1, 1000000)
        X = sampler.sample(problem, N, seed = sampler_seed)
        
        # Transform the samples to the problem's input space
        return problem._transform_unit_X(X)
    
    @Verbose.analyze
    def analyze(self, problem, X: np.ndarray, Y: np.ndarray = None, target = 'objFunc', index = 'all', nNeighbors: int = 2):
        """
        Perform the Delta Test analysis on the input data.
        --------------------------------------------------
        :param problem: Problem - The problem instance that defines the input and output space.
        :param X: np.ndarray - A 2D array of shape `(N, n_input)`, representing the input data for analysis.
        :param Y: np.ndarray - A 1D array of length `N` representing the output values corresponding to `X`. 
                  If None, it will be computed by evaluating the problem with `X`.
        :param target: str - The target of the analysis, set 'objFunc' or 'conFunc'. Defaults to 'objFunc'.
        :param index: list - The index of the output variables to analyze. Defaults to 'all'.
        :param nNeighbors: int - The number of nearest neighbors used in Delta Test estimation. Defaults to 2.
        
        :return: Result - An object containing the sensitivity indices 'S1', 'S2', and 'ST', 
                          representing first-order, second-order, and total-order indices. 
                          You can use result.Si to get the sensitivity indices.
        """
        
        # Set the problem instance for analysis
        self.setProblem(problem)
        
        # Retrieve the number of nearest neighbors for analysis
        self.setParaValue('nNeighbors', nNeighbors)
        
        # Evaluate the problem if Y is not provided
        Y = self.check_Y(X, Y, target, index)
        
        # Scale the input and output data if scalers are provided
        X, Y = self.__check_and_scale_xy__(X, Y)
        nInput = problem.nInput
        numY = Y.shape[1]
        
        outputLabel = "obj" if target == "objFunc" else "con"
        
        S1 = np.zeros((numY, nInput))
        S1_scaled = np.zeros((numY, nInput))
        row_label = [f"{outputLabel}{i+1}" for i in range(numY)]
        col_label_1 = problem.xLabels
        
        for i in range(numY):
            Y_i = Y[:, i:i+1]
            base = self._cal_delta(X, Y_i, nNeighbors)
            for j in range(nInput):
                XSub = np.delete(X, [j], axis=1)
                S1[i, j] = self._cal_delta(XSub, Y_i, nNeighbors)
            S1_scaled[i] = S1[i] / np.sum(S1[i])

        res = [('S1', S1, row_label, col_label_1, 'decsDim1'), ('S1_scale', S1_scaled, row_label, col_label_1, 'decsDim1')]
        
        X, Y = self.__reverse_X_Y__(X, Y)
        
        self.recordResult(X, Y, res)

        return self.result.generateNetCDF()
    
    def findCombEA(self, problem, X: np.ndarray, Y: np.ndarray = None, 
                   FEs: int = 10000, 
                   verboseFlag: bool = True, saveFlag: bool = True):
        """
        Find the best combination using Evolutionary Algorithm.
        -----------------------------------------------------
        :param problem: Problem - The problem instance.
        :param X: np.ndarray - Input data array.
        :param Y: np.ndarray - Output data array. If None, it will be computed.
        :param FEs: int - Maximum number of function evaluations. Defaults to 10000.
        :param verboseFlag: bool - If True, enables verbose mode. Defaults to True.
        :param saveFlag: bool - If True, saves the results. Defaults to True.

        :return: Result - The result of the optimization.
        """
        
        # Set the problem instance for analysis
        self.setProblem(problem)
        
        # Retrieve the number of nearest neighbors for analysis
        nNeighbors = self.getParaValue('nNeighbors')
        
        # Evaluate the problem if Y is not provided
        if Y is None:
            Y = self.evaluate(X)
        
        # Scale the input and output data if scalers are provided
        X, Y = self.__check_and_scale_xy__(X, Y)
        
        @ProblemABC.singleFunc
        def objFunc(x_):
            """
            Minimize the delta value.

            :param x_: Binary array indicating selected variables.

            :return: float - The Delta value for the selected variables.
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
        nOutput = 1
        ub = [1] * nInput
        lb = [0] * nInput
        varType = [1] * nInput
        
        problem = Problem(nInput=nInput, nOutput=nOutput, ub=ub, lb=lb, 
                          varType=varType, objFunc=objFunc, optType='min')
        
        # Initialize the GA
        ga = GA(maxFEs=FEs, verboseFlag=verboseFlag, saveFlag=saveFlag)
        
        # Run the GA
        res = ga.run(problem)
        
        return res
    
    def findCombVio(self, problem, X: np.ndarray, Y: np.ndarray = None):
        """
        Find the best combination using a brute-force approach.
        -----------------------------------------------------
        :param problem: Problem - The problem instance.
        :param X: np.ndarray - Input data array.
        :param Y: np.ndarray - Output data array. If None, it will be computed.

        :return: List[str] - List of labels for the most sensitive variables.
        """
        
        from itertools import product
        
        # Set the problem instance for analysis
        self.setProblem(problem)
        
        nInput = problem.nInput
        
        # Retrieve the number of nearest neighbors for analysis
        nNeighbors = self.getParaValue('nNeighbors')
        
        # Evaluate the problem if Y is not provided
        if Y is None:
            Y = self.evaluate(X)
        
        # Scale the input and output data if scalers are provided
        X, Y = self.__check_and_scale_xy__(X, Y)
        
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
    
    #--------------------Private Function--------------------------#
    def _cal_delta(self, X: np.ndarray, Y: np.ndarray, nNeighbors: int):
        """
        Calculate the Delta value using KDTree for nearest neighbor search.
        ---------------------------------------------------------------
        :param X: np.ndarray - The input data array.
        :param Y: np.ndarray - The output data array.
        :param nNeighbors: int - The number of nearest neighbors to consider.

        :return: float - The calculated Delta value.
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