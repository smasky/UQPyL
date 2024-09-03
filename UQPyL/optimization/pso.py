import numpy as np
from tqdm import tqdm

from ..problems import Problem
from ..DoE import LHS
from .optimizer import Optimizer, verboseForRun
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
    name="Particle Swarm Optimization"
    def __init__(self, problem: Problem, nInit: int=50, nPop: int=50,
                    x_init=None, y_init=None,
                    w: float=0.1, c1: float=0.5, c2: float=0.5,
                    maxIterTimes: int=1000,
                    maxFEs: int=50000,
                    maxTolerateTimes: int=1000,
                    tolerate=1e-6,
                    verbose=True,
                    logFlag=False):
            #problem setting
            self.n_input=problem.n_input
            self.ub=problem.ub.reshape(1,-1);self.lb=problem.lb.reshape(1,-1)
            
            #algorithm setting
            self.w=w;self.c1=c1;self.c2=c2
            self.tolerate=tolerate
            self.nInit=nInit
            self.nPop=nPop
            
            #
            self.x_init=x_init
            self.y_init=y_init
            
            #termination setting
            self.maxIterTimes=maxIterTimes
            self.maxFEs=maxFEs
            self.maxTolerateTimes=maxTolerateTimes
            
            #setting record
            setting={}
            setting["nPop"]=nPop
            setting["nInit"]=nInit
            setting["w"]=w
            setting["c1"]=c1
            setting["c2"]=c2
            setting["maxFEs"]=maxFEs
            setting["maxIterTimes"]=maxIterTimes
            setting["maxTolerateTimes"]=maxTolerateTimes
            self.setting=setting
            
            super().__init__(problem=problem, maxFEs=maxFEs, maxIter=maxIterTimes, 
                         maxTolerateTimes=maxTolerateTimes, tolerate=tolerate, verbose=verbose, logFlag=logFlag)
    
    @verboseForRun
    def run(self, xInit=None, yInit=None) -> dict:
        '''
            Run the Particle Swarm Optimization
            -------------------------------
            Returns:
                Result: dict
                    the result of the Particle Swarm Optimization, including the following keys:
                    decs: np.ndarray
                        the best decision of the Particle Swarm Optimization
                    objs: np.ndarray
                        the best objective value of the Particle Swarm Optimization
                    history_decs: np.ndarray
                        the history of the decision of the Particle Swarm Optimization
                    history_objs: np.ndarray
                        the history of the objective value of the Particle Swarm Optimization
                    iters: int
                        the iteration times of the Particle Swarm Optimization
                    FEs: int
                        the function evaluations of the Particle Swarm Optimization
        '''
        
        
        if xInit is None:
            lhs=LHS('classic', problem=self.problem)
            xInit=lhs.sample(self.nInit, self.n_input)
        
        if yInit is None:
            yInit=self.evaluate(xInit)
        
        decs=xInit
        objs=yInit
    
        self.update(xInit, yInit)
        
        #Init vel and orien
        P_best_decs=np.copy(decs)
        P_best_objs=np.copy(objs)
        ind=np.argmin(P_best_objs)
        G_best_dec=np.copy(P_best_decs[ind])      
        vel=np.copy(decs)
        
        while self.checkTermination():
            decs, vel=self._operationPSO(decs, vel, P_best_decs, G_best_dec, self.w)
            decs=self._randomParticle(decs)
            objs=self.evaluate(decs)
            
            replace=np.where(objs<P_best_objs)[0]
            P_best_decs[replace]=np.copy(decs[replace])
            P_best_objs[replace]=np.copy(objs[replace])
            
            ind=np.argmin(P_best_objs)
            G_best_dec=np.copy(P_best_decs[ind])      
            
            self.update(decs, objs)
      
    def _operationPSO(self, decs, vel, P_best_decs, G_best_dec, w):
        
        N, D=decs.shape
        
        PatricleVel=vel
        
        r1=np.random.rand(N,D)
        r2=np.random.rand(N,D)
        
        offVel=w*PatricleVel+self.c1*r1*(P_best_decs-decs)+self.c2*r2*(G_best_dec-decs)
        offDecs=decs+offVel
        
        offDecs = np.clip(offDecs, self.lb, self.ub)
        return offDecs, offVel
    
    
    def _randomParticle(self, decs):
        
        n_to_reinit = int(0.1 * decs.shape[0])
        rows_to_mutate = np.random.choice(decs.shape[0], size=n_to_reinit, replace=False)
        cols_to_mutate = np.random.choice(decs.shape[1], size=n_to_reinit, replace=False)

        decs[rows_to_mutate, cols_to_mutate] = np.random.uniform(self.lb[0, cols_to_mutate], self.ub[0, cols_to_mutate], size=n_to_reinit)
                
        return decs