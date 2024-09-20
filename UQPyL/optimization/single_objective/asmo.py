# Adaptive Surrogate Modelling-based Optimization <Single> <Surrogate>

from .sce_ua import SCE_UA
from ..algorithmABC import Algorithm, Population, Verbose
from ...problems import PracticalProblem
from ...surrogates import Surrogate
from ...surrogates.kriging import KRG
from ...utility.scalers import StandardScaler

class ASMO(Algorithm):
    '''
        Adaptive Surrogate Modelling-based Optimization <Single> <Surrogate>
        ----------------------------------------------
        Attributes:
            problem: Problem
                the problem you want to solve, including the following attributes:
                n_input: int
                    the input number of the problem
                ub: 1d-np.ndarray or float
                    the upper bound of the problem
                lb: 1d-np.ndarray or float
                    the lower bound of the problem
                evaluate: Callable
                    the function to evaluate the input
            surrogate: Surrogate
                the surrogate model you want to use
            n_init: int, default=50
                Number of initial samples for surrogate modelling
    '''
    name="ASMO"
    type="EA"
    def __init__(self, nInit: int=50,
                 surrogate: Surrogate=None,
                 optimizer: Algorithm=None,
                 maxFEs: int=1000,
                 maxTolerateTimes: int=100,
                 verbose: bool=True, verboseFreq: int=1, logFlag: bool=False, saveFlag=False
                 ):
        
        super().__init__(maxFEs=maxFEs, maxTolerateTimes=maxTolerateTimes, verbose=verbose, verboseFreq=verboseFreq, logFlag=logFlag, saveFlag=saveFlag)
        
        self.setParameters('nInit', nInit)
        
        if surrogate is None:
            scaler=(StandardScaler(0, 1), StandardScaler(0, 1))
            surrogate=KRG(scalers=scaler)
        self.surrogate=surrogate
        
        if optimizer is None:
            optimizer=SCE_UA(maxFEs=5000, verbose=False)
            
        self.optimizer=optimizer
        self.optimizer.verbose=False
        
    @Verbose.decoratorRun
    @Algorithm.initializeRun
    def run(self, problem, xInit=None, yInit=None, oneStep=False):
        '''
        main procedure
        ''' 
        #Initialization
        nInit=self.getParaValue('nInit')
        
        #Problem
        self.problem=problem
        #SubProblem
        subProblem=PracticalProblem(self.surrogate.predict, problem.nInput, 1,problem.ub, problem.lb)
        
        #Termination Condition Setting
        self.FEs=0; self.iters=0; self.tolerateTimes=0
        
        #Population Generation
        if xInit is not None:
            if yInit is not None:
                pop=Population(xInit, yInit)
            else:
                pop=Population(xInit)
                self.evaluate(pop)
        else:
            pop=self.initialize(nInit)
        
        while self.checkTermination():
            # Build surrogate model
            self.surrogate.fit(pop.decs, pop.objs)
            res=self.optimizer.run(subProblem)
            
            offSpring=Population(decs=res.bestDec)
            self.evaluate(offSpring)
            pop.add(offSpring)
            self.record(pop)
            
            if oneStep:
                break
                    
        return self.result
            
        
            
            
