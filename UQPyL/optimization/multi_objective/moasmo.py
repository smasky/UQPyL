### Multi-Objective Adaptive Surrogate Modelling-based Optimization
import numpy as np
from scipy.spatial.distance import cdist

from ...DoE import LHS
from ...problems import PracticalProblem
from ...surrogates import Mo_Surrogates
from ..algorithmABC import Algorithm, Population, Verbose
from .nsga_ii import NSGAII


class MOASMO(Algorithm):
    '''
    Multi-Objective Adaptive Surrogate Modelling-based Optimization <Multi-objective> <Surrogate>
    -----------------------------------------------------------------
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
        surrogates: Surrogates
            the surrogates you want to use, you should implement Mo_Surrogate class
        Pct: float, default=0.2
            the percentage of the population to be selected for infilling
        n_init: int, default=50
            the number of initial samples
        n_pop: int, default=100
            the number of population for evolution optimizer
        maxFEs: int, default=1000
            the maximum number of function evaluations
        maxIter: int, default=100
            the maximum number of iterations
        x_init: 2d-np.ndarray, default=None
            the initial input samples
        y_init: 2d-np.ndarray, default=None
            the initial output samples
        advance_infilling: bool, default=False
            the switch to use advanced infilling or not
            
    Methods:
        run()
            run the optimization
    
    References:
        [1] W. Gong et al., Multiobjective adaptive surrogate modeling-based optimization for parameter estimation of large, complex geophysical models, 
                            Water Resour. Res., vol. 52, no. 3, pp. 1984–2008, Mar. 2016, doi: 10.1002/2015WR018230.
    '''
    
    name="MOASMO"
    type="MOEA"
    
    def __init__(self, surrogates: Mo_Surrogates=None,
                 optimizer: Algorithm=None,
                 pct: float=0.2, nInit: int=50, nPop: int=50, 
                 maxFEs: int=1000, 
                 maxIterTimes: int=100,
                 maxTolerateTimes=None, tolerate=1e-6,
                 verbose=True, verboseFreq=1, logFlag=True, saveFlag=False):

        super().__init__(maxFEs, maxIterTimes, maxTolerateTimes, tolerate, verbose, verboseFreq, logFlag, saveFlag)
        
        self.setParameters('pct', pct)
        self.setParameters('nInit', nInit)
        self.setParameters('nPop', nPop)
        
        if surrogates is not None:
            self.surrogates=surrogates
        
        self.optimizer=optimizer
        
    @Verbose.decoratorRun
    @Algorithm.initializeRun
    def run(self, problem, xInit=None, yInit=None):
        
        pct=self.getParaValue('pct')
        nInit=self.getParaValue('nInit')
        
        nInfilling=int(pct*nInit)
        
        self.FEs=0; self.iters=0; self.tolerateTimes=0
        
        #Problem
        self.problem=problem
        #SubProblem
        subProblem=PracticalProblem(self.surrogates.predict, problem.nInput, problem.nOutput, problem.ub, problem.lb)
        
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
            
            #Build surrogate models
            self.surrogates.fit(pop.decs, pop.objs)
            #Run optimization
            res=self.optimizer.run(subProblem)
            
            offSpring=Population(decs=res.bestDec, objs=res.bestObj)
            
            if offSpring.nPop>nInfilling:
                bestOff=offSpring.getBest(nInfilling)
            else:
                bestOff=offSpring
            
            self.evaluate(bestOff)
            
            pop.add(bestOff)
            self.record(pop)
            
            # offSpring=Population(decs=res.bestDec)
            # self.evaluate(offSpring)
            
            # pop.add(offSpring)
            # nsga_ii=NSGAII(self.subProblem, self.n_pop)
            # #main optimization
            # Result=nsga_ii.run()
            # BestX=Result['pareto_x']
            # BestY=Result['pareto_y']
            # CrowdDis=Result['crowdDis']
            
            # if self.advance_infilling==False:
            #     #Origin version
            #     if BestY.shape[0]>n_infilling:
            #         idx=CrowdDis.argsort()[::-1][:n_infilling]
            #         BestX=np.copy(BestX[idx])
            #         BestY=np.copy(BestY[idx])
            # else:
            #     #Advanced version Using crowding-based strategy
            #     if BestY.shape[0]>n_infilling:
                    
            #         Known_FrontNo, _ =nsga_ii.NDSort(YPop, YPop.shape[0])
            #         Unknown_FrontNo, _=nsga_ii.NDSort(BestY, BestY.shape[0])
            #         Known_best_Y=YPop[np.where(Known_FrontNo==1)]
            #         Unknown_best_Y=BestY[np.where(Unknown_FrontNo==1)]
            #         Unknown_best_X=BestX[np.where(Unknown_FrontNo==1)]

            #         added_points_Y=[]
            #         added_points_X=[]
            #         for _ in range(n_infilling):
                        
            #             if len(added_points_Y)==0:
            #                 distances = cdist(Unknown_best_Y, Known_best_Y)
            #             else:
            #                 distances = cdist(Unknown_best_Y, np.append(Known_best_Y, added_points_Y, axis=0))

            #             max_distance_index = np.argmax(np.min(distances, axis=1))

            #             added_point = Unknown_best_Y[max_distance_index]
            #             added_points_Y.append(added_point)
            #             added_points_X.append(Unknown_best_X[max_distance_index])
            #             Known_best_Y = np.append(Known_best_Y, [added_point], axis=0)

            #             Unknown_best_Y = np.delete(Unknown_best_Y, max_distance_index, axis=0)
            #             Unknown_best_X = np.delete(Unknown_best_X, max_distance_index, axis=0)
            #         BestX=np.copy(np.array(added_points_X))
            #         BestY=np.copy(np.array(added_points_Y))
            
            # FE+=BestX.shape[0]
            # # show_process.update(BestX.shape[0])
            
            # BestY=self.evaluate(BestX)
            # XPop=np.vstack((XPop,BestX))
            # YPop=np.vstack((YPop,BestY))      
        return self.result
          
        
                
        
            
        
            
            
        
        
        