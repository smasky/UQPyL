import numpy as np
from typing import Optional, Tuple

from .base import AnalysisABC
from ..util import Scaler, Verbose
from ..problem import ProblemABC as Problem

class Morris(AnalysisABC):
    """
    -------------------------------------------------
    Morris Method for Sensitivity Analysis
    -------------------------------------------------
    This class implements the Morris method, which is 
    used for screening and identifying important factors 
    in a model by calculating elementary effects.

    Methods:
        sample: Generate a sample for Morris analysis
        analyze: Perform Morris analysis from the X and Y you provided.

    Examples:
        # `problem` is an instance of ProblemABC or Problem from UQPyL.problems
        #  You must create a problem instance before using this method.
        >>> mor_method = Morris(problem)
        >>> X = mor_method.sample(100, 4)
        >>> Y = problem.evaluate(X)
        >>> mor_method.analyze(X, Y)

    References:
        [1] Max D. Morris (1991) Factorial Sampling Plans for Preliminary Computational Experiments, 
            Technometrics, 33:2, 161-174, doi: 10.2307/1269043
        [2] SALib, https://github.com/SALib/SALib
    -------------------------------------------------
    """
    
    name = "Morris"
    
    def __init__(self, scalers: Tuple[Optional[Scaler], Optional[Scaler]] = (None, None),
                 verboseFlag: bool = True, logFlag: bool = False, saveFlag: bool = False):
        """
        Initialize the Morris method for sensitivity analysis.
        ----------------------------------------------------------------
        :param scalers: Tuple[Optional[Scaler], Optional[Scaler]] - Tuple containing scalers for input (X) and output (Y) data. Defaults to (None, None).
        :param numLevels: int - The number of levels for each input factor. Recommended values are between 4 and 10. Defaults to 4.
        :param verboseFlag: bool - If True, enables verbose mode for logging. Defaults to False.
        :param logFlag: bool - If True, enables logging of results. Defaults to False.
        :param saveFlag: bool - If True, saves the results to a file. Defaults to False.
        """
        
        # Initialize the base class with provided scalers and flags
        super().__init__(scalers, verboseFlag, logFlag, saveFlag)
        
        # Set the number of levels for each input factor
        self.setParaValue("numLevels", 4)
        
    def sample(self, problem: Problem, numTrajectory: int = 100, numLevels: Optional[int] = 4, seed: Optional[int] = None):
        """
        Generate a sample for Morris analysis.
        -----------------------------------------------------
        :param problem: Problem - The problem instance defining the input space.
        :param numTrajectory: int, optional - The number of trajectories. Each trajectory is a sequence of input points used to compute the elementary effects. Defaults to 500.
        :param numLevels: int, optional - The number of levels for each input factor. If not provided, the initialized value of `numLevels` is used.

        :return X: np.ndarray - A 2D array of shape `(numTrajectory * (nInput + 1), nInput)`, representing the generated sample points.
        """
        
        self.rng = np.random.default_rng(seed) if seed is not None else np.random.default_rng()
        
        nt = numTrajectory
        
        self.setParaValue("numLevels", numLevels)
        
        nInput = problem.nInput
        
        # Initialize the sample array
        X = np.zeros((nt*(nInput+1), nInput))
        
        # Generate trajectories for each input factor
        for i in range(nt):
            X[i*(nInput+1):(i+1)*(nInput+1), :] = self._generate_trajectory(nInput, numLevels)
        
        # Transform the samples to the problem's input space
        return problem._transform_unit_X(X)
    
    @Verbose.analyze
    def analyze(self, problem: Problem, X: np.ndarray, Y: Optional[np.ndarray] = None, target = 'objFunc', index = 'all') -> dict:
        """
        Perform Morris analysis.
        --------------------------------------------
        :param problem: Problem - The problem instance defining the input and output space.
        :param X: np.ndarray - A 2D array representing the input data for analysis.
        :param Y: np.ndarray, optional - A 1D array representing the output values corresponding to `X`. If None, it will be computed by evaluating the problem with `X`.
        :param target: str - The target of the analysis, set 'objFunc' or 'conFunc'. Defaults to 'objFunc'.
        :param index: list - The index of the output variables to analyze. Defaults to 'all'.
        :return: Result - An object containing result of the analysis.
        """
        # Retrieve the number of levels for each input factor
        numLevels = self.getParaValue("numLevels")
        
        # Set the problem instance for analysis
        self.setProblem(problem)
        
        self.check_Y(X, Y, target, index)
        numY = Y.shape[1]
        
        nInput = problem.nInput
        
        numTrajectory = int(X.shape[0]/(nInput+1))
        
        # Scale the input and output data if scalers are provided
        X, Y = self.__check_and_scale_xy__(X, Y)

        outputLabel = "obj" if target == "objFunc" else "con"
        
        mu = np.zeros((numY, nInput))
        mu_star = np.zeros((numY, nInput))
        sigma = np.zeros((numY, nInput))
        S1_scaled = np.zeros((numY, nInput))
        
        row_label = [f"{outputLabel}{i+1}" for i in range(numY)]
        col_label_1 = problem.xLabels
        
        for i in range(numY):
            
            Y_i = Y[:, i:i+1]
            
            # Initialize an array to store elementary effects
            EE = np.zeros((nInput, numTrajectory))
            
            N = int(X.shape[0]/numLevels)
            
            # Calculate elementary effects for each trajectory
            for j in range(numTrajectory):
                X_sub = X[j*(nInput+1):(j+1)*(nInput+1), :]
                Y_sub = Y_i[j*(nInput+1):(j+1)*(nInput+1), :]

                Y_diff = np.diff(Y_sub, axis=0)
                
                tmp_indice = list(np.argmax(np.diff(X_sub, axis=0) != 0, axis=1))
                indice = [tmp_indice.index(j) for j in range(len(tmp_indice))]
                delta_diff = np.sum(np.diff(X_sub, axis=0), axis=1).reshape(-1,1)
                ee = Y_diff/delta_diff
                EE[:, j:j+1] = ee[indice]
        

            mu[i] = np.mean(EE, axis=1)
            mu_star[i] = np.mean(np.abs(EE), axis=1)
            sigma[i] = np.std(EE, axis=1, ddof=1)
            S1_scaled[i] = mu[i]/np.sum(mu[i])
            
        res = [('mu', mu, row_label, col_label_1, 'decsDim1'), 
               ('mu_star', mu_star, row_label, col_label_1, 'decsDim1'), 
               ('sigma', sigma, row_label, col_label_1, 'decsDim1'), 
               ('S1_scale', S1_scaled, row_label, col_label_1, 'decsDim1')]
        
        X, Y = self.__reverse_X_Y__(X, Y)
        
        self.recordResult(X, Y, res)
        
        # Return the result object containing all sensitivity indices
        return self.result.generateNetCDF()
    
    #-------------------------Private Function-------------------------------------#
    def _generate_trajectory(self, nx: int, num_levels: int=4) -> np.ndarray:
        """
        Generate a random trajectory from Reference[1].
        -------------------------------------------------
        :param nx: int - The number of input factors.
        :param num_levels: int - The number of levels for each input factor.

        :return: np.ndarray - A 2D array of shape `(nx + 1, nx)`, representing the generated trajectory.
        """
        delta = num_levels/(2*(num_levels-1))
        
        B = np.tril(np.ones([nx + 1, nx], dtype=int), -1)
        
        # from paper[1] page 164
        D_star = np.diag(self.rng.choice([-1, 1], nx)) #step1
        J = np.ones((nx+1, nx))
        
        levels_grids = np.linspace(0, 1-delta, int(num_levels / 2))
        x_star = self.rng.choice(levels_grids, nx).reshape(1,-1) #step2
        
        P_star = np.zeros((nx,nx))
        cols = self.rng.choice(nx, nx, replace=False)
        P_star[np.arange(nx), cols]=1 #step3
        
        element_a = J[0, :] * x_star
        element_b = P_star.T
        element_c = np.matmul(2.0 * B, element_b)
        element_d = np.matmul((element_c - J), D_star)

        B_star = element_a + (delta / 2.0) * (element_d + J)
    
        return B_star