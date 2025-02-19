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

    Parameters:
        problem (Problem): 
            The problem instance defining the input space.
        scalers (Tuple[Scaler, Scaler], optional): 
            Tuple containing scalers for input (X) and output (Y) data. 
            Defaults to (None, None).
        numLevels (int): 
            The number of levels for each input factor. 
            Recommended values are between 4 and 10. Defaults to 4.
        verboseFlag (bool): 
            If True, enables verbose mode for logging. Defaults to False.
        logFlag (bool): 
            If True, enables logging of results. Defaults to False.
        saveFlag (bool): 
            If True, saves the results to a file. Defaults to False.

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
    
    name="Morris"
    
    def __init__(self, scalers: Tuple[Optional[Scaler], Optional[Scaler]] = (None, None),
                       numLevels: int = 4,
                       verboseFlag: bool = False, logFlag: bool = False, saveFlag: bool = False):
        '''
        Initialize the Morris method for sensitivity analysis.
        
        The Morris method is a screening method used to identify important factors
        in a model by calculating the elementary effects of input factors. This 
        initialization sets up the necessary parameters and configurations.

        Parameters:
            scalers (Tuple[Optional[Scaler], Optional[Scaler]]): 
                Tuple containing scalers for input (X) and output (Y) data. 
                Defaults to (None, None), meaning no scaling is applied.
            numLevels (int): 
                The number of levels for each input factor. This determines the 
                granularity of the factor space exploration. Recommended values 
                are between 4 and 10. Defaults to 4.
            verboseFlag (bool): 
                If True, enables verbose mode for logging, providing detailed 
                output during execution. Defaults to False.
            logFlag (bool): 
                If True, enables logging of results to a file or console. 
                Defaults to False.
            saveFlag (bool): 
                If True, saves the results to a file for later analysis. 
                Defaults to False.
        '''
        
        #Attribute
        self.firstOrder = True
        self.secondOrder = False
        self.totalOrder = True
        
        super().__init__(scalers, verboseFlag, logFlag, saveFlag)
        
        #Parameter Setting
        self.setParameters("numLevels", numLevels)
        
        
    def sample(self, problem: Problem, numTrajectory: int = None, numLevels: Optional[int] = None) -> np.ndarray:
        '''
        Generate a sample for Morris analysis
        ---------------------------------------
        This method generates a sample of input data `X` for the Morris method,
        which is used to compute the elementary effects of input factors.

        Parameters:  
            problem (Problem): 
                The problem instance defining the input space.
            numTrajectory (int, optional): 
                The number of trajectories. Each trajectory is a sequence of 
                input points used to compute the elementary effects. 
                Defaults to 500, recommended values are between 500 and 1000.
            numLevels (int, optional): 
                The number of levels for each input factor. If not provided, 
                the initialized value of `numLevels` is used.

        Returns:
            np.ndarray: 
                A 2D array of shape `(numTrajectory * (nInput + 1), nInput)`, 
                representing the generated sample points.
        '''

        nt = numTrajectory
        
        if numLevels is None:
            numLevels = self.getParaValue('numLevels')
        else:
            self.setParameters("numLevels", numLevels)
        
        nInput = problem.nInput
        
        X = np.zeros((nt*(nInput+1), nInput))
        
        for i in range(nt):
            X[i*(nInput+1):(i+1)*(nInput+1), :] = self._generate_trajectory(nInput, numLevels)
        
        return problem._transform_unit_X(X)
    
    @Verbose.decoratorAnalyze
    def analyze(self, problem: Problem, X: np.ndarray, Y: Optional[np.ndarray] = None) -> dict:
        '''
        Perform Morris analysis
        -------------------------
        This method performs the Morris sensitivity analysis by calculating 
        the elementary effects of input factors based on the provided input 
        data `X` and output data `Y`.

        Parameters:
            problem (Problem): 
                The problem instance defining the input and output space.
            X (np.ndarray): 
                A 2D array representing the input data for analysis.
            Y (np.ndarray, optional): 
                A 1D array representing the output values corresponding to `X`. 
                If None, it will be computed by evaluating the problem with `X`.

        Returns:
            dict: 
                A dictionary containing the sensitivity indices 'mu', 'mu_star', 
                and 'sigma', which represent the mean, absolute mean, and standard 
                deviation of the elementary effects, respectively.
        '''
        numLevels = self.getParaValue("numLevels")
        
        self.setProblem(problem)
        
        nInput = problem.nInput
        
        if Y is None:
            Y = self.evaluate(X)
            
        numTrajectory = int(X.shape[0]/(nInput+1))
        
        X, Y = self.__check_and_scale_xy__(X, Y)

        EE = np.zeros((nInput, numTrajectory))
        
        N = int(X.shape[0]/numLevels)
        
        for i in range(numTrajectory):
            X_sub = X[i*(nInput+1):(i+1)*(nInput+1), :]
            Y_sub = Y[i*(nInput+1):(i+1)*(nInput+1), :]

            Y_diff = np.diff(Y_sub, axis=0)
            
            tmp_indice = list(np.argmax(np.diff(X_sub, axis=0) != 0, axis=1))
            indice = [tmp_indice.index(i) for i in range(len(tmp_indice))]
            delta_diff = np.sum(np.diff(X_sub, axis=0), axis=1).reshape(-1,1)
            ee = Y_diff/delta_diff
            EE[:, i:i+1] = ee[indice]
            
        mu = np.mean(EE, axis=1)
        mu_star= np.mean(np.abs(EE), axis=1)
        sigma = np.std(EE, axis=1, ddof=1)
        
        self.record('mu', problem.xLabels, mu)
        self.record('mu_star', problem.xLabels, mu_star)
        self.record('sigma', problem.xLabels, sigma)

        self.record('S1(scaled)', problem.xLabels, mu_star/np.sum(mu_star))
        
        return self.result
    
    #-------------------------Private Function-------------------------------------#
    def _generate_trajectory(self, nx: int, num_levels: int=4) -> np.ndarray:
        '''
            Generate a random trajectory from Reference[1]
        '''
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
        
    def _default_sample(self):
        
        return self.sample(500)