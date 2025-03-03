import numpy as np
from typing import Optional, Tuple

from .saABC import SA
from ..utility import Scaler, Verbose
from ..problems import ProblemABC as Problem

class Morris(SA):
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
                 numLevels: int = 4,
                 verboseFlag: bool = False, logFlag: bool = False, saveFlag: bool = False):
        """
        Initialize the Morris method for sensitivity analysis.
        ----------------------------------------------------------------
        :param scalers: Tuple[Optional[Scaler], Optional[Scaler]] - Tuple containing scalers for input (X) and output (Y) data. Defaults to (None, None).
        :param numLevels: int - The number of levels for each input factor. Recommended values are between 4 and 10. Defaults to 4.
        :param verboseFlag: bool - If True, enables verbose mode for logging. Defaults to False.
        :param logFlag: bool - If True, enables logging of results. Defaults to False.
        :param saveFlag: bool - If True, saves the results to a file. Defaults to False.
        """
        
        # Attribute indicating the types of sensitivity indices calculated
        self.firstOrder = True
        self.secondOrder = False
        self.totalOrder = True
        
        # Initialize the base class with provided scalers and flags
        super().__init__(scalers, verboseFlag, logFlag, saveFlag)
        
        # Set the number of levels for each input factor
        self.setParameters("numLevels", numLevels)
        
    def sample(self, problem: Problem, numTrajectory: int = None, numLevels: Optional[int] = None) -> np.ndarray:
        """
        Generate a sample for Morris analysis.
        -----------------------------------------------------
        :param problem: Problem - The problem instance defining the input space.
        :param numTrajectory: int, optional - The number of trajectories. Each trajectory is a sequence of input points used to compute the elementary effects. Defaults to 500.
        :param numLevels: int, optional - The number of levels for each input factor. If not provided, the initialized value of `numLevels` is used.

        :return X: np.ndarray - A 2D array of shape `(numTrajectory * (nInput + 1), nInput)`, representing the generated sample points.
        """
        
        nt = numTrajectory
        
        if numLevels is None:
            numLevels = self.getParaValue('numLevels')
        else:
            self.setParameters("numLevels", numLevels)
        
        nInput = problem.nInput
        
        # Initialize the sample array
        X = np.zeros((nt*(nInput+1), nInput))
        
        # Generate trajectories for each input factor
        for i in range(nt):
            X[i*(nInput+1):(i+1)*(nInput+1), :] = self._generate_trajectory(nInput, numLevels)
        
        # Transform the samples to the problem's input space
        return problem._transform_unit_X(X)
    
    @Verbose.decoratorAnalyze
    def analyze(self, problem: Problem, X: np.ndarray, Y: Optional[np.ndarray] = None) -> dict:
        """
        Perform Morris analysis.
        --------------------------------------------
        :param problem: Problem - The problem instance defining the input and output space.
        :param X: np.ndarray - A 2D array representing the input data for analysis.
        :param Y: np.ndarray, optional - A 1D array representing the output values corresponding to `X`. If None, it will be computed by evaluating the problem with `X`.

        :return: Result - An object containing the sensitivity indices 'S1', 'S2', and 'ST', 
                          representing first-order, second-order, and total-order indices. 
                          You can use result.Si to get the sensitivity indices.
        """
        # Retrieve the number of levels for each input factor
        numLevels = self.getParaValue("numLevels")
        
        # Set the problem instance for analysis
        self.setProblem(problem)
        
        nInput = problem.nInput
        
        # Evaluate the problem if Y is not provided
        if Y is None:
            Y = self.evaluate(X)
            
        numTrajectory = int(X.shape[0]/(nInput+1))
        
        # Scale the input and output data if scalers are provided
        X, Y = self.__check_and_scale_xy__(X, Y)

        # Initialize an array to store elementary effects
        EE = np.zeros((nInput, numTrajectory))
        
        N = int(X.shape[0]/numLevels)
        
        # Calculate elementary effects for each trajectory
        for i in range(numTrajectory):
            X_sub = X[i*(nInput+1):(i+1)*(nInput+1), :]
            Y_sub = Y[i*(nInput+1):(i+1)*(nInput+1), :]

            Y_diff = np.diff(Y_sub, axis=0)
            
            tmp_indice = list(np.argmax(np.diff(X_sub, axis=0) != 0, axis=1))
            indice = [tmp_indice.index(i) for i in range(len(tmp_indice))]
            delta_diff = np.sum(np.diff(X_sub, axis=0), axis=1).reshape(-1,1)
            ee = Y_diff/delta_diff
            EE[:, i:i+1] = ee[indice]
            
        # Calculate mean, absolute mean, and standard deviation of elementary effects
        mu = np.mean(EE, axis=1)
        mu_star= np.mean(np.abs(EE), axis=1)
        sigma = np.std(EE, axis=1, ddof=1)
        
        # Record the calculated sensitivity indices
        self.record('mu', problem.xLabels, mu)
        self.record('mu_star', problem.xLabels, mu_star)
        self.record('sigma', problem.xLabels, sigma)

        # Record scaled sensitivity indices
        self.record('S1(scaled)', problem.xLabels, mu_star/np.sum(mu_star))
        
        # Return the result object containing all sensitivity indices
        return self.result
    
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
        D_star = np.diag(np.random.choice([-1, 1], nx)) #step1
        J = np.ones((nx+1, nx))
        
        levels_grids = np.linspace(0, 1-delta, int(num_levels / 2))
        x_star = np.random.choice(levels_grids, nx).reshape(1,-1) #step2
        
        P_star = np.zeros((nx,nx))
        cols = np.random.choice(nx, nx, replace=False)
        P_star[np.arange(nx), cols]=1 #step3
        
        element_a = J[0, :] * x_star
        element_b = P_star.T
        element_c = np.matmul(2.0 * B, element_b)
        element_d = np.matmul((element_c - J), D_star)

        B_star = element_a + (delta / 2.0) * (element_d + J)
    
        return B_star