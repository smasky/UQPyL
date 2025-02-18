#Delta test
import numpy as np
from scipy.spatial import KDTree
from typing import Optional, Tuple

from .saABC import SA
# from .util._binary_ga import Binary_GA
from ..DoE import LHS, Sampler
from ..problems import ProblemABC as Problem
from ..utility import Scaler, Verbose

class Delta_Test(SA):
    """
    -------------------------------------------------
    Delta Test
    -------------------------------------------------
    This class implements the Delta Test, which is 
    a non-parametric method for sensibility analysis.
    
    Methods:
        sample: Generate a sample for Delta Test analysis
        analyze: perform Delta Test analyze from the X and Y you provided.
    
    Examples:
        >>> delta_method = Delta_Test(nNeighbors = 2)
        >>> X = delta_method.sample(problem, N = 1000)
        >>> res = delta_method.analyze(problem, X)
        >>> print(res)
        
    References:
        [1] E. Eirola et al, Using the Delta Test for Variable Selection, 
                                Artificial Neural Networks, 2008.
        [2] SALib, https://github.com/SALib/SALib
    --------------------------------------------------------------------------
    """
    def __init__(self, scalers: Tuple[Optional[Scaler], Optional[Scaler]]=(None, None), 
                       nNeighbors: int=2,
                       verboseFlag: bool = False, logFlag: bool = False, saveFlag: bool = False):
        '''
        Initializes the Delta Test method. 
        
        args:
            scaler (Tuple[Optional[Scaler], Optional[Scaler]]): 
                Tuple containing scalers for input (X) and output (Y) data. 
                Defaults to (None, None).   
            nNeighbors (int): 
                The number of nearest neighbors used in Delta Test estimation. Defaults to 2.
            verboseFlag (bool): 
                If True, enables verbose mode for logging. Defaults to False.
            logFlag (bool): 
                If True, enables logging of results. Defaults to False.
            saveFlag (bool): 
                If True, saves the results to a file. Defaults to False.           
        '''
        
        #Attribute
        self.firstOrder = True
        self.secondOrder = False
        self.totalOrder = False
        
        super().__init__(scalers, verboseFlag, logFlag, saveFlag)

        self.setParameters('nNeighbors', nNeighbors)
        
    def sample(self, problem: Problem, N: int=500, sampler: Sampler = LHS('classic')):
        """
        Generate a sample set for the Delta Test.

        This method generates a sample of input data `X` using the specified sampling method.
        The generated data is transformed into the unit space of the given problem.

        Args:
            problem (Problem): 
                The problem instance defining the input space.
            N (int, optional): 
                The number of samples to generate. Defaults to 500.
            sampler (Sampler, optional): 
                The sampling method to use. Defaults to Latin Hypercube Sampling (LHS) with 'classic' mode.

        Returns:
            np.ndarray: 
                A 2D array of shape `(N, nInput)`, where `nInput` is the number of input variables.
        """
        
        nInput = problem.nInput
        
        X = sampler.sample(N, nInput)
        
        return problem._transform_unit_X(X)
    
    @Verbose.decoratorAnalyze
    def analyze(self, problem, X: np.ndarray, Y: np.ndarray=None):
        """
        Perform the Delta Test analysis on the input data.

        This method calculates the Delta Test sensitivity analysis based on the input data `X` 
        and output data `Y`. If `Y` is not provided, it is computed by evaluating the problem.

        Args:
            problem (Problem): 
                The problem instance that defines the input and output space.
            X (np.ndarray): 
                A 2D array of shape `(N, n_input)`, representing the input data for analysis.
            Y (np.ndarray, optional): 
                A 1D array of length `N` representing the output values corresponding to `X`. 
                If None, it will be computed by evaluating the problem with `X`.
                
        Returns:
            res (Result): 
                A class containing the sensitivity result, you can sue `res.si` to obtain results.        
        """
        
        self.setProblem(problem)
        
        nNeighbors = self.getParaValue('nNeighbors')
        
        if Y is None:
            Y = self.evaluate(X)
        
        X, Y=self.__check_and_scale_xy__(X, Y)
        nInput = problem.nInput
        
        S1 = np.zeros(nInput)
        
        base = self._cal_delta(X, Y, nNeighbors)
        
        for i in range(nInput):
            XSub = np.delete(X, [i], axis=1)
            
            S1[i] = self._cal_delta(XSub, Y, nNeighbors)
        
        S1 = S1 - base 
        
        S1 = S1 - np.min(S1)
        
        self.record('S1', problem.xLabels, S1)
        
        self.record('S1(scaled)', problem.xLabels, S1/np.sum(S1))

        return self.result
    
    #TODO Find the best GCV as the most sensitive combination
    def findBestCombination(self, problem, X: np.ndarray, Y: np.ndarray=None):
        
        self.setProblem(problem)
        
        nNeighbors = self.getParaValue('nNeighbors')
        
        if Y is None:
            Y = self.evaluate(X)
        
        pass
        
    #--------------------Private Function--------------------------#
    def _cal_delta(self, X, Y, nNeighbors):
        """
        Calculate the Delta value using KDTree for nearest neighbor search.

        Parameters:
            X (np.ndarray): 
                The input data array.
            Y (np.ndarray): 
                The output data array.
            nNeighbors (int): 
                The number of nearest neighbors to consider.

        Returns:
            float: 
                The calculated Delta value.
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