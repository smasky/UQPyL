#Sobol sensibility analysis
import numpy as np
from scipy.stats import qmc
from typing import Optional, Tuple

from .saABC import SA
from ..problems import ProblemABC as Problem
from ..utility import Scaler, Verbose
class Sobol(SA):
    '''
    Sobol' sensibility analysis
    --------------------------
    Parameters:
        problem: Problem
            the problem you want to analyse
        scaler: Tuple[Scaler, Scaler], default=(None, None)
            used for scaling X or Y
            
        Following parameters derived from the variable 'problem'
        n_input: the input number of the problem
        ub: the upper bound of the problem
        lb: the lower bound of the problem
    
    Methods:
        sample: Generate a sample for sobol' analysis
        analyze: perform sobol analyze from the X and Y you provided.
    
    Examples:
        >>> sob_method=Sobol(problem)
        >>> X=sob_method.sample(500)
        >>> Y=problem.evaluate(X)
        >>> sob_method.analyze(X,Y)
    
    References:
    [1] I. M. Sobol', Global sensitivity indices for nonlinear mathematical models and their Monte Carlo estimates, 
                      Mathematics and Computers in Simulation, vol. 55, no. 1, pp. 271–280, Feb. 2001, 
                      doi: 10.1016/S0378-4754(00)00270-6.
    [2] A. Saltelli et al, Variance based sensitivity analysis of model output. Design and estimator for the total sensitivity index, 
                           Computer Physics Communications, vol. 181, no. 2, pp. 259–270, Feb. 2010, 
                           doi: 10.1016/j.cpc.2009.09.018.
    [3] SALib, https://github.com/SALib/SALib
    '''
    
    name="Sobol"
    def __init__(self, scalers: Tuple[Optional[Scaler], Optional[Scaler]]=(None, None),
                       calSecondOrder: bool=False,
                       N: int=512, skipValue: int=0, scramble: bool=True,
                       verbose: bool=False, logFlag: bool=False, saveFlag: bool=False):
        
        #Attribute
        self.firstOrder=True
        self.secondOrder=True if calSecondOrder else False
        self.totalOrder=True
        
        super().__init__(scalers, verbose, logFlag, saveFlag)
        #Parameter Setting
        self.setParameters("calSecondOrder", calSecondOrder)
        self.setParameters("skipValue", skipValue)
        self.setParameters("scramble", scramble)
        self.setParameters("N", N)
        
    #-------------------------Public Functions--------------------------------#
    def sample(self, problem: Problem, N: Optional[int]=None, 
               skipValue: Optional[int]=0, scramble: Optional[bool]=True):
        '''
            Generate Sobol_sequence using Saltelli's sampling technique in [2]
            ----------------------
            Parameters:
                N: int default=512
                    the number of base sequence. Noted that N should be power of 2.
                cal_second_order: bool default=False
                    the switch to calculate second order or not
                
            Returns:
                X: np.ndarray
                    if cal_second_order
                        the size of X is (N*(n_input+2), n_input)
                    else
                        the size of X is (N*(2*n_input+2), n_input)
        '''
        if N is None:
            N=self.getParaValue("N")
        if skipValue is None:
            skipValue=self.getParaValue("skipValue")
        if scramble is None:
            scramble=self.getParaValue("scramble")
            
        self.setParameters("N", N)
        self.setParameters("skipValue", skipValue)
        self.setParameters("scramble", scramble)
        
        nInput=problem.nInput
        calSecondOrder=self.getParaValue("calSecondOrder")
        
        M=None
        if skipValue>0 and isinstance(skipValue, int):
            
            M=skipValue
            
            if not((M&(M-1))==0 and (M!=0 and M-1!=0)):
                raise ValueError("skip value must be a power of 2!")
            
            if N>M:
                raise ValueError("skip value must be greater than N you set!")
        
        elif skipValue<0 or not isinstance(skipValue, int):
            raise ValueError("skip value must be a positive integer!")
        
        sampler=qmc.Sobol(nInput*2, scramble=scramble, seed=1)
        
        if M:
            sampler.fast_forward(M)
        
        if calSecondOrder:
            saltelliSequence=np.zeros(((2*nInput+2)*N, nInput))
        else:
            saltelliSequence=np.zeros(((nInput+2)*N, nInput))
        
        baseSequence=sampler.random(N)
        
        index=0
        for i in range(N):
            
            saltelliSequence[index, :]=baseSequence[i, :nInput]

            index+=1
            
            saltelliSequence[index:index+nInput,:]=np.tile(baseSequence[i, :nInput], (nInput, 1))
            saltelliSequence[index:index+nInput,:][np.diag_indices(nInput)]=baseSequence[i, nInput:]               
            index+=nInput
           
            if calSecondOrder:
                saltelliSequence[index:index+nInput,:]=np.tile(baseSequence[i, nInput:], (nInput, 1))
                saltelliSequence[index:index+nInput,:][np.diag_indices(nInput)]=baseSequence[i, :nInput] 
                index+=nInput
            
            saltelliSequence[index,:]=baseSequence[i, nInput:nInput*2]
            index+=1
        
        xSample=saltelliSequence
         
        return self.transform_into_problem(problem, xSample)
    
    @Verbose.decoratorAnalyze
    def analyze(self, problem: Problem, X: Optional[np.ndarray]=None, Y: Optional[np.ndarray]=None):
        '''
            Perform sobol' analyze
            Noted that if the X and Y is None, sample(512) is used for generate data 
                       and use the method problem.evaluate to evaluate them.
            In Sobol method, we recommend to indicate X at least.
        -------------------------
            Parameters:
                X: np.ndarray
                    the input data
                Y: np.ndarray
                    the result data
                cal_second_order: bool default=False
                    the switch to calculate second order or not
                verbose: bool
                    the switch to print analysis summary or not
            Returns:
                Si: dict
                    The type of Si is dict. And it contain 'S1', 'S2', 'ST' key value.   
        '''
        #Parameters Setting
        N, skipValue, scramble=self.getParaValue("N", "skipValue", "scramble")
        calSecondOrder = self.getParaValue("calSecondOrder")
        
        self.setProblem(problem)
        
        if X is None or Y is None:
            X=self.sample(problem, N, skipValue, scramble)
            Y=problem.evaluate(X)
            
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
                    S_Labels.append(f"{problem.x_labels[j]}-{problem.x_labels[k]}")
        
        #Record Data
        self.record('S1(First Order)', problem.x_labels, S1)
        if calSecondOrder:
            self.record('S2(Second Order)', S_Labels, S2) 
        self.record('ST(Total Order)', problem.x_labels, ST)
        
        return self.result
    
    #-------------------------Private Functions--------------------------------#
    def _secondOrder(self, A, AB1, AB2, BA, B):
        
        Y=np.r_[A,B]
        
        Vjk=np.mean(BA*AB2- A*B, axis=0)/np.var(Y, axis=0)
        Sj=self._firstOrder(A, AB1, B)
        Sk=self._firstOrder(A, AB2, B)
        
        return Vjk-Sj-Sk
       
    def _firstOrder(self, A, AB, B):
        
        Y=np.r_[A,B]
        
        return np.mean(B*(AB-A), axis=0)/np.var(Y, axis=0)
    
    def _totalOrder(self, A, AB, B):
        
        Y=np.r_[A,B]
        
        return 0.5*np.mean((A-AB)**2, axis=0)/np.var(Y, axis=0)
            
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