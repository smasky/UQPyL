#Sobol sensibility analysis
import numpy as np
from scipy.stats import qmc
from typing import Optional, Tuple

from .saABC import SA
from ..problems import ProblemABC as Problem
from ..utility import Scaler, Verbose
class Sobol(SA):
    '''
    -------------------------------------------------
    Sobol' Sensitivity Analysis
    -------------------------------------------------
    This class implements the Sobol' method, which is used for 
    global sensitivity analysis of model outputs. It calculates 
    first-order, second-order, and total-order sensitivity indices.

    Parameters:
        problem (Problem): 
            The problem instance defining the input space.
        scalers (Tuple[Scaler, Scaler], optional): 
            Tuple containing scalers for input (X) and output (Y) data. 
            Defaults to (None, None), meaning no scaling is applied.
        calSecondOrder (bool): 
            If True, calculates second-order sensitivity indices. 
            Defaults to False.
        verboseFlag (bool): 
            If True, enables verbose mode for logging. Defaults to False.
        logFlag (bool): 
            If True, enables logging of results. Defaults to False.
        saveFlag (bool): 
            If True, saves the results to a file. Defaults to False.

    Methods:
        sample: Generate a sample for Sobol' analysis
        analyze: Perform Sobol' analysis from the X and Y you provided.

    Examples:
        >>> sob_method = Sobol(problem)
        >>> X = sob_method.sample(500)
        >>> Y = problem.evaluate(X)
        >>> sob_method.analyze(X, Y)

    References:
        [1] I. M. Sobol', Global sensitivity indices for nonlinear mathematical models and their Monte Carlo estimates, 
            Mathematics and Computers in Simulation, vol. 55, no. 1, pp. 271–280, Feb. 2001, 
            doi: 10.1016/S0378-4754(00)00270-6.
        [2] A. Saltelli et al, Variance based sensitivity analysis of model output. Design and estimator for the total sensitivity index, 
            Computer Physics Communications, vol. 181, no. 2, pp. 259–270, Feb. 2010, 
            doi: 10.1016/j.cpc.2009.09.018.
        [3] SALib, https://github.com/SALib/SALib
    -------------------------------------------------
    '''
    
    name="Sobol"
    def __init__(self, scalers: Tuple[Optional[Scaler], Optional[Scaler]]=(None, None),
                       calSecondOrder: bool=False,
                       verboseFlag: bool=False, logFlag: bool=False, saveFlag: bool=False):
        '''
        Initialize the Sobol' method for sensitivity analysis.
        
        The Sobol' method is a variance-based sensitivity analysis technique 
        that decomposes the variance of the model output into contributions 
        from input variables and their interactions.

        Parameters:
            scalers (Tuple[Optional[Scaler], Optional[Scaler]]): 
                Tuple containing scalers for input (X) and output (Y) data. 
                Defaults to (None, None), meaning no scaling is applied.
            calSecondOrder (bool): 
                If True, calculates second-order sensitivity indices, 
                capturing interactions between pairs of input variables. 
                Defaults to False.
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
        self.firstOrder=True
        self.secondOrder=True if calSecondOrder else False
        self.totalOrder=True
        
        super().__init__(scalers, verboseFlag, logFlag, saveFlag)
        #Parameter Setting
        self.setParameters("calSecondOrder", calSecondOrder)
        
    #-------------------------Public Functions--------------------------------#
    def sample(self, problem: Problem, N: Optional[int]=512, 
               skipValue: Optional[int]=0, scramble: Optional[bool]=False):
        '''
        Generate a sample for Sobol' analysis
        ---------------------------------------
        This method generates a sample of input data `X` using Saltelli's 
        sampling technique, which is designed to efficiently estimate 
        Sobol' sensitivity indices.

        Parameters:
            problem (Problem): 
                The problem instance defining the input space.
            N (int, optional): 
                The number of base sequence samples. Must be a power of 2. 
                Defaults to 512.
            skipValue (int, optional): 
                The number of initial samples to skip in the Sobol' sequence. 
                Must be a power of 2. Defaults to 0.
            scramble (bool, optional): 
                If True, applies scrambling to the Sobol' sequence for 
                improved uniformity. Defaults to False.

        Returns:
            np.ndarray: 
                A 2D array representing the generated sample points, with 
                shape determined by the number of input variables and 
                whether second-order indices are calculated.
        '''
                
        nInput=problem.nInput
        
        calSecondOrder=self.getParaValue("calSecondOrder")
                
        if skipValue>0 and isinstance(skipValue, int):
            
            M=skipValue
            
            if not( ( M & ( M - 1 ) ) == 0 and ( M != 0 and M-1 != 0) ):
                raise ValueError("The skip value must be a power of 2!")
            
            if N < M:
                raise ValueError("N must be greater than skip value you set!")
        
        elif skipValue<0 or not isinstance(skipValue, int):
            
            raise ValueError("skip value must be a positive integer!")
        
        if not ( N & ( N-1 ) )==0:
            raise ValueError(f"The sample number must be a power of 2! \n You can use {int(np.power(2, np.ceil(np.log2(N))))}")
        
        sampler = qmc.Sobol(nInput*2, scramble=scramble, seed=1)
        
        if skipValue > 0:
            sampler.fast_forward(skipValue)
        
        if calSecondOrder:
            SS=np.zeros( ( (2 * nInput + 2) * N, nInput) ) #Saltelli Sequence
        else:
            SS=np.zeros( ( (nInput + 2) * N, nInput) ) #Saltelli Sequence
        
        BS=sampler.random(N) #Base Sequence
        
        index=0
        
        for i in range(N):
            
            SS[index, :]=BS[i, :nInput]

            index+=1
            
            SS[index:index+nInput,:]=np.tile(BS[i, :nInput], (nInput, 1))
            SS[index:index+nInput,:][np.diag_indices(nInput)]=BS[i, nInput:]               
            index+=nInput
           
            if calSecondOrder:
                SS[index:index+nInput,:]=np.tile(BS[i, nInput:], (nInput, 1))
                SS[index:index+nInput,:][np.diag_indices(nInput)]=BS[i, :nInput] 
                index+=nInput
            
            SS[index,:]=BS[i, nInput:nInput*2]
            index+=1
        
        X = SS
         
        return problem._unit_X_transform(X)
    
    @Verbose.decoratorAnalyze
    def analyze(self, problem: Problem, X: np.ndarray, Y: Optional[np.ndarray]=None):
        '''
        Perform Sobol' analysis
        -------------------------
        This method performs the Sobol' sensitivity analysis by calculating 
        first-order, second-order, and total-order sensitivity indices based 
        on the provided input data `X` and output data `Y`.

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
                A dictionary containing the sensitivity indices 'S1', 'S2', 
                and 'ST', representing first-order, second-order, and 
                total-order indices, respectively.
        '''
        #Parameters Setting
        calSecondOrder = self.getParaValue("calSecondOrder")
        
        self.setProblem(problem)
        
        if Y is None:
            Y = self.evaluate(X)
            
        X, Y=self.__check_and_scale_xy__(X, Y)
        
        nInput=problem.nInput; 
        
        if calSecondOrder:
            n=int(X.shape[0]/(2*nInput+2))
        else:
            n=int(X.shape[0]/(nInput+2))
        
        Y=(Y-Y.mean())/Y.std()
        
        A, B, AB, BA=self._separateOutputValues(Y, nInput, n, calSecondOrder)
        
        S2 = [] if calSecondOrder else None
        S1=[]; ST=[]
        
        for j in range(nInput):
            S1.append(self._firstOrder(A, AB[:, j:j+1], B))
            ST.append(self._totalOrder(A, AB[:, j:j+1], B))
        
        S2=[]; S_Labels=[]
        if calSecondOrder:
            for j in range(nInput):
                for k in range(j+1, nInput):
                    S2.append(self._secondOrder(A, AB[:, j:j+1], AB[:, k:k+1], BA[:, j:j+1], B))
                    S_Labels.append(f"{problem.xLabels[j]}-{problem.xLabels[k]}")
        
        #Record Data
        self.record('S1', problem.xLabels, S1)
        if calSecondOrder:
            self.record('S2', S_Labels, S2) 
        self.record('ST', problem.xLabels, ST)
        
        return self.result
    
    #-------------------------Private Functions--------------------------------#
    def _secondOrder(self, A, AB1, AB2, BA, B):
        
        Y=np.r_[A,B]
        
        Vjk=float(np.mean(BA*AB2- A*B, axis=0)/np.var(Y, axis=0))
        Sj=self._firstOrder(A, AB1, B)
        Sk=self._firstOrder(A, AB2, B)
        
        return Vjk-Sj-Sk
       
    def _firstOrder(self, A, AB, B):
        
        Y=np.r_[A,B]
        
        return float(np.mean(B*(AB-A), axis=0)/np.var(Y, axis=0))
    
    def _totalOrder(self, A, AB, B):
        
        Y=np.r_[A,B]
        
        return float(0.5*np.mean((A-AB)**2, axis=0)/np.var(Y, axis=0))
            
    def _separateOutputValues(self, Y, d, n, calSecondOrder):
        
        AB=np.zeros((n, d))
        BA=np.zeros((n, d)) if calSecondOrder else None
        
        step=2*d+2 if calSecondOrder else d+2
        
        total=Y.shape[0]
        
        A=Y[0:total:step, :]
        B=Y[(step-1):total:step, :]
        
        for j in range(d):
            
            AB[:, j]=Y[(j+1):total:step, 0]
            
            if calSecondOrder:
                BA[:, j]=Y[(j+1+d):total:step, 0]
        
        return A, B, AB, BA