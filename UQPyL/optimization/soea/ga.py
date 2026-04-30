# Genetic Algorithm <Single>
import numpy as np

from typing import Optional

from ..base import AlgorithmABC
from ..population import Population

from ..core.constraint import calcConstraintViolation
from ..core.ga_operator import gaOperator
from ..core.tournament import tourSelect

class GA(AlgorithmABC):
    """
    Single-objective genetic algorithm.
    """
    
    name = "GA"
    alg_type = "EA"
    
    def __init__(self, nPop: int = 50,
                 proC: float = 1, disC: float = 20, proM: float = 1, disM: float = 20,
                 maxFEs: int = 50000,
                 maxIters: int = 1000,
                 maxTolerates: Optional[int] = None, tolerate: float = 1e-6,
                 verboseFlag: bool = True, verboseFreq: int = 10, logFlag: bool = False, saveFlag = True,
                 saveFreq: int = 100):
        """
        Initialize the algorithm.

        :param nPop: Population size.
        :param proC: Crossover probability.
        :param disC: Crossover distribution index.
        :param proM: Mutation probability.
        :param disM: Mutation distribution index.
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
        
        super().__init__(maxFEs = maxFEs, maxIters = maxIters, 
                         maxTolerates = maxTolerates, tolerate = tolerate,
                         verboseFlag = verboseFlag, verboseFreq = verboseFreq, logFlag = logFlag, saveFlag = saveFlag,
                         saveFreq = saveFreq)
        
        # Set user-defined parameters
        self.setParaVal('proC', proC)
        self.setParaVal('disC', disC)
        self.setParaVal('proM', proM)
        self.setParaVal('disM', disM)
        self.setParaVal('nPop', nPop)
        
    #--------------------Public Functions---------------------#
    def run(self, problem, seed: Optional[int] = None):
        """
        Run the algorithm on the given problem.

        :param problem: Problem instance.
        :param seed: Random seed.
        :return OptResult: Final optimization result.
        """
        # setup algorithm
        self.setup(problem, seed)
        
        # Retrieve parameter values
        proC, disC, proM, disM = self.getParaVal('proC', 'disC', 'proM', 'disM')
        nPop = self.getParaVal('nPop')
        
        # Generate initial population
        pop = self.initPop(nPop)
        self.update(pop)
       
        # Iterative process
        while self.checkTermination(pop):
            
            # Select mating pool using tournament selection
            cv = calcConstraintViolation(pop.cons, pop.conWgt)
            feasible = np.zeros((len(pop), 1), dtype=float) if cv is None else (cv > 0).astype(float).reshape(-1, 1)
            violation = np.zeros((len(pop), 1), dtype=float) if cv is None else cv.reshape(-1, 1)
            matingIdx = tourSelect(2, len(pop), feasible, violation, pop.objs)
            matingPool = pop[matingIdx]
            
            # Generate offspring using genetic operator
            offspringDecs = gaOperator(matingPool.decs, problem.ub, problem.lb, proC, disC, proM, disM)
            offspring = Population(offspringDecs)
            
            # Evaluate the offspring
            self.evaluate(offspring)
            
            # Merge offspring with current population
            pop = pop.merge(offspring)
            
            # Select the best individuals to form the new population
            pop = pop.getBest(nPop)
            self.update(pop)
                    
        # Return the final result
        return self.finalize()
