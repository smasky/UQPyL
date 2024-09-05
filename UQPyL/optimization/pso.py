import numpy as np

from ..problems import Problem
from ..DoE import LHS
from .optimizer import Optimizer, Population, verboseForRun
class PSO(Optimizer):
    '''
        Particle Swarm Optimization
        -----------------------------
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
            n_sample: int, default=50
                the number of samples as the population
            w: float, default=0.1
                the inertia weight
            c1: float, default=0.5
                the cognitive parameter
            c2: float, default=0.5
                the social parameter
            maxIterTimes: int, default=1000
                the maximum iteration times
            maxFEs: int, default=50000
                the maximum function evaluations
            maxTolerateTimes: int, default=1000
                the maximum tolerate times which the best objective value does not change
            tolerate: float, default=1e-6
                the tolerate value which the best objective value does not change
        Methods:
            run: run the Particle Swarm Optimization
        
        References:
            [1] J. Kennedy and R. Eberhart, Particle swarm optimization, in Proceedings of ICNN'95 - International Conference on Neural Networks, 1995.
            [2] J. Kennedy and R. Eberhart, Swarm Intelligence, Academic Press, 2001.
            [3] M. Clerc and J. Kennedy, The particle swarm - explosion, stability, and convergence in a multidimensional complex space, IEEE Transactions on Evolutionary Computation, 2002.
            [4] Y. Shi and R. C. Eberhart, A modified particle swarm optimizer, in Proceedings of the IEEE Congress on Evolutionary Computation, 1998.
        
    '''
    name= "Particle Swarm Optimization"
    type= "EA" #Evolutionary Algorithm
    target= "Single"
    def __init__(self, nInit: int=50, nPop: int=50,
                    w: float=0.1, c1: float=0.5, c2: float=0.5,
                    maxIterTimes: int=1000,
                    maxFEs: int=50000,
                    maxTolerateTimes: int=1000, tolerate: float=1e-6,
                    verbose: bool=True, verboseFreq: int=100, logFlag: bool=False):
        
            super().__init__(maxFEs=maxFEs, maxIterTimes=maxIterTimes, 
                         maxTolerateTimes=maxTolerateTimes, tolerate=tolerate, 
                         verbose=verbose, verboseFreq=verboseFreq, logFlag=logFlag)
            
            #user-define setting
            self.w=w;self.c1=c1;self.c2=c2
            self.tolerate=tolerate
            self.nInit=nInit
            self.nPop=nPop
             
            #setting record
            self.setting["nPop"]=nPop
            self.setting["nInit"]=nInit
            self.setting["w"]=w
            self.setting["c1"]=c1
            self.setting["c2"]=c2
                
    @verboseForRun
    def run(self, problem, xInit=None, yInit=None):
        
        self.problem=problem
        self.FEs=0; self.iters=0; self.tolerateTimes=0
        
        if xInit is not None:
            if yInit is not None:
                pop=Population(xInit, yInit)
            else:
                pop=Population(xInit)
                self.evaluate(pop)
        else:
            pop=self.initialize()
        
        pop=pop.getTop(self.nPop)
        
        self.record(pop)
        
        #Init vel and orient
        pBestPop=pop
        gBestPop=pop[pop.argsort()[0]]
        vel=pop.decs
        
        
        while self.checkTermination():
            
            pop, vel=self._operationPSO(pop, vel, pBestPop, gBestPop)
            pop=self._randomParticle(pop)
            self.evaluate(pop)
            
            replace=np.where(pop.objs<pBestPop.objs)[0]
            pBestPop.replace(replace, pop[replace])
            gBestPop=pBestPop[pBestPop.argsort()[0]]
            
            self.record(pop)
            
        return self.result
    
    def _operationPSO(self, pop, vel, pBestPop, gBestPop):
        
        n, d=pop.size()
        
        particleVel=vel
        
        r1=np.random.rand(n, d)
        r2=np.random.rand(n, d)
        
        a=(gBestPop-pop)*self.c2*r2
        offVel=self.w*particleVel+(pBestPop.decs-pop.decs)*self.c1*r1+(gBestPop.decs-pop.decs)*self.c2*r2
        offSpring=pop+offVel
        
        offSpring.clip(self.problem.lb, self.problem.ub)
        
        return offSpring, offVel
    
    def _randomParticle(self, pop):
        
        n, d=pop.size()
        n_to_reinit = int(0.1 * n)
        rows_to_mutate = np.random.choice(n, size=n_to_reinit, replace=False)
        cols_to_mutate = np.random.choice(d, size=n_to_reinit, replace=False)

        pop.decs[rows_to_mutate, cols_to_mutate] = np.random.uniform(self.problem.lb[0, cols_to_mutate], self.problem.ub[0, cols_to_mutate], size=n_to_reinit)
        
        return pop