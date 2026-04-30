### Multi-Objective Adaptive Surrogate Modelling-based Optimization
import numpy as np
from scipy.spatial.distance import cdist
from typing import Optional

from ..moea.nsga_ii import NSGAII
from ..base import AlgorithmABC
from ..core import NDSort
from ..population import Population

from ...problem import Problem
from ...surrogate import MultiSurrogate
from ...surrogate.rbf.radial_basis_function import RBF

class MOASMO(AlgorithmABC):
    """
    Multi-objective adaptive surrogate modelling-based optimization algorithm.
    """
    
    name = "MOASMO"
    alg_type = "MOEA"
    
    def __init__(self, surrogates: MultiSurrogate = None,
                 optimizer: AlgorithmABC = None,
                 pct: float = 0.2, nInit: int = 50, nPop: int = 50, 
                 advance_infilling: bool = False,
                 maxFEs: int = 1000, 
                 maxIters: int = 100,
                 maxTolerates: int = None, tolerate: float = 1e-6,
                 verboseFlag: bool = True, verboseFreq: int = 1, logFlag: bool = False, saveFlag: bool = False,
                 saveFreq: int = 100):
        """
        Initialize the algorithm.

        :param surrogates: Surrogate ensemble.
        :param optimizer: Inner optimizer.
        :param pct: Infill percentage.
        :param nInit: Number of initial samples.
        :param nPop: Population size for the inner optimizer.
        :param advance_infilling: Whether to use advanced infilling.
        :param maxFEs: Maximum number of function evaluations.
        :param maxIters: Maximum number of iterations.
        :param maxTolerates: Maximum tolerated non-improving iterations.
        :param tolerate: Improvement tolerance.
        :param verboseFlag: Whether to print terminal output.
        :param verboseFreq: Summary output frequency.
        :param logFlag: Whether to save full text logs.
        :param saveFlag: Whether to save sqlite results.
        :param saveFreq: Snapshot save frequency.
        """
        
        super().__init__(maxFEs, maxIters, maxTolerates, tolerate, 
                         verboseFlag, verboseFreq, logFlag, saveFlag, saveFreq)
        
        # Set user-defined parameters
        self.setParaVal('pct', pct)
        self.setParaVal('nInit', nInit)
        self.setParaVal('advance_infilling', advance_infilling)
        
        # Initialize surrogate models
        self.surrogates = surrogates

        # Initialize optimizer
        if optimizer is not None:
            if not isinstance(optimizer, AlgorithmABC):
                raise ValueError("Please append the type of optimizer!")
            self.optimizer = optimizer
        else:
            self.optimizer = NSGAII(maxFEs = 5000)
        
        self.optimizer.verboseFlag, self.optimizer.logFlag, self.optimizer.saveFlag = False, False, False
        
    def run(self, problem, xInit = None, yInit = None, seed: Optional[int] = None):
        """
        Run the algorithm on the given problem.

        :param problem: Problem instance.
        :param xInit: Optional initial decision samples.
        :param yInit: Optional initial objective samples.
        :param seed: Random seed.
        :return OptResult: Final optimization result.
        """
        # setup algorithm
        self.setup(problem, seed)
        
        # Initialize surrogate models
        if self.surrogates is None:
            nObj = getattr(problem, "nObj", getattr(problem, "nOutput"))
            self.surrogates = MultiSurrogate(n_surrogates = nObj, models_list=[RBF() for _ in range(nObj)])

        # Retrieve parameter values
        pct = self.getParaVal('pct')
        nInit = self.getParaVal('nInit')
        advance_infilling = self.getParaVal('advance_infilling')
        
        nInfilling = int(pct*nInit)
        
        # Create a subproblem for surrogate model optimization
        nObj = getattr(problem, "nObj", getattr(problem, "nOutput"))
        subProblem = Problem(nInput = problem.nInput, nObj = nObj, 
                             ub = problem.ub, lb = problem.lb, objFunc = self.surrogates.predict,
                             varType = problem.varType, 
                             varSet = problem.varSet, optType = problem.optType, 
                             xLabels = problem.xLabels)
        
        # Generate initial population
        if xInit is not None:
            if yInit is not None:
                pop = Population(xInit, yInit)
            else:
                pop = Population(xInit)
                self.evaluate(pop)
            
            if nInit > len(pop):
                pop.merge(self.initPop(nInit-len(pop)))
            
        else: 
            pop = self.initPop(nInit)
        self.update(pop)
        
        # Iterative optimization process
        while self.checkTermination(pop):
            
            # Build surrogate models
            self.surrogates.fit(pop.decs, pop.objs)
            
            # Run optimization on the surrogate model
            res = self.optimizer.run(subProblem)
            bestDecs = np.asarray(res.bestDecs)
            bestObjs = np.asarray(res.bestObjs)
            offSpring = Population(decs=bestDecs, objs=bestObjs)
            
            if advance_infilling==False:
                
                if offSpring.nPop > nInfilling:
                    bestOff = offSpring.getBest(nInfilling)
                else:
                    bestOff = offSpring
                    
            else:
                
                if offSpring.nPop > nInfilling:
                    Known_FrontNo, _ = NDSort(pop.objs, pop.cons)
                    Unknown_FrontNo, _ = NDSort(offSpring.objs, offSpring.cons)
                    
                    Known_best_Y = pop.objs[np.where(Known_FrontNo==1)]
                    Unknown_best_Y = offSpring.objs[np.where(Unknown_FrontNo==1)]
                    Unknown_best_X = offSpring.decs[np.where(Unknown_FrontNo==1)]
                    
                    added_points_Y = []
                    added_points_X = []
                    
                    for _ in range(nInfilling):
                        
                        if len(added_points_Y)==0:
                            distances = cdist(Unknown_best_Y, Known_best_Y)
                        else:
                            distances = cdist(Unknown_best_Y, np.append(Known_best_Y, added_points_Y, axis=0))

                        max_distance_index = np.argmax(np.min(distances, axis=1))
                        
                        added_point = Unknown_best_Y[max_distance_index]
                        added_points_Y.append(added_point)
                        added_points_X.append(Unknown_best_X[max_distance_index])
                        Known_best_Y = np.append(Known_best_Y, [added_point], axis=0)
                        
                        Unknown_best_Y = np.delete(Unknown_best_Y, max_distance_index, axis=0)
                        Unknown_best_X = np.delete(Unknown_best_X, max_distance_index, axis=0)
                    
                    BestX = np.copy(np.array(added_points_X))
                    # BestY = np.copy(np.array(added_points_Y))
                    bestOff = Population(decs = BestX)
            
            # Evaluate the selected offspring
            self.evaluate(bestOff)
            
            pop.add(bestOff)
            self.update(pop)
                            
        return self.finalize()
          
        
                
        
            
        
            
            
        
        
        
