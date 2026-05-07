import abc
import os
import numpy as np

from .population import Population
from .runtime import OptResult, Result, SqliteStorage, Verbose
from ..doe import LHS
from ..core.params import Params
from ..core.runtime_session import RunSession

class AlgorithmABC(metaclass = abc.ABCMeta):
    """
    Base class for optimization algorithms.
    """
    def __init__(self, maxFEs: int = None, maxIters: int = None, maxTolerates: int = None, tolerate: float = 1e-6, 
                 verboseFlag: bool = True, verboseFreq: int = 10, logFlag: bool = True, saveFlag: bool = False,
                 saveFreq: int = 100, hvRefPoint = None):
        
        self.params = Params()
        self.setting = self.params
        self.result = Result(self)
        self.state = self.result
        
        self.problem = None
        self.maxFEs = maxFEs
        self.maxIter = maxIters
        self.maxTolerates = maxTolerates
        self.tolerate = tolerate
        
        self.verboseFlag = verboseFlag
        self.verboseFreq = verboseFreq
        self.logFlag = logFlag
        self.saveFlag = saveFlag
        self.saveFreq = saveFreq
        self.hvRefPoint = None if hvRefPoint is None else np.asarray(hvRefPoint, dtype=float).copy()
        self.storage = None
        self.session: RunSession | None = None
        self.runId = None

        if self.hvRefPoint is not None:
            self.set('hvRefPoint', self.hvRefPoint.copy())
    
    def reset(self):
        
        self.FEs = 0; self.iters = 0; self.tolerateTimes = 0
        
        self.state.reset()
    
    def setup(self, problem, seed):
        
        self.setProblem(problem)
        
        self.reset()
        Verbose.setupContext(self, problem)
        if self.saveFlag:
            rootDir = getattr(problem, "workDir", os.getcwd())
            self.storage = SqliteStorage(rootDir)
        self.runId = None
        self.session = None
        
        if seed is None:
            seed = int(np.random.default_rng().integers(0, 1000000))
        self.rng = np.random.default_rng(seed)
        
        self.set('seed', seed)
        self.set('saveFreq', self.saveFreq)
        if self.saveFlag:
            self.session = self.storage.create_run(self)
            self.runId = self.session.run_id
        Verbose.printSettings(self)
    
    def initPop(self, nInit):
        
        lhs = LHS('classic')
        
        seed = int(self.rng.integers(0, 1000000))
        xInit = lhs.sample(self.problem, nInit, seed)

        pop = Population(xInit)
        self.evaluate(pop)
        
        return pop
    
    def setProblem(self, problem):
        self.problem = problem
        self.optType = getattr(problem, "optType", None)
        if hasattr(problem, "optType"):
            self.set('optType', problem.optType)
    
    def evaluate(self, pop):
        decs = self.problem.apply_var_type(pop.decs)
        res = self.problem.evaluate(decs)
        objs = res.objs * self.problem.opt
        pop.assignEval(objs, res.cons)
        self.FEs += pop.nPop
        return pop

    def updateState(self, pop):
        algType = 'EA' if self.problem.nObj == 1 else 'MOEA'
        self.state.update(pop, self.problem, self.FEs, self.iters, algType)
        return self.state

    def update(self, pop):
        self.updateState(pop)
        if self.verboseFlag > 0 or self.logFlag > 0 or self.saveFlag > 0:
            Verbose.printIteration(self)
        if self.saveFlag and self.session is not None and self.iters % self.saveFreq == 0:
            self.storage.saveSnapshot(self.session, self, self.buildResult(), isFinal=False)
        return self.state
    
    def checkTermination(self, pop):
        
        signalFlag = False
        previousBest = None if self.state.bestObjs is None else np.copy(self.state.bestObjs)
        
        if self.FEs < self.maxFEs:
            if self.maxIter is None or self.iters <= self.maxIter:
                if self.maxTolerates is None or self.tolerateTimes <= self.maxTolerates:
                    
                    signalFlag = True
                    
                    # For GUI version
                    if hasattr(self.problem, 'GUI'):
                        self.problem.iterEmit.send()
                        if self.problem.isStop == True:
                            return False

        # Check termination for single-objective optimization
        if self.problem.nObj == 1 and self.tolerate is not None and previousBest is not None:
            old_best = float(np.ravel(previousBest)[0])
            new_best = float(np.ravel(pop.getBest(k=1).objs)[0])
            if abs(old_best - new_best) > self.tolerate:
                self.tolerateTimes = 0
            else:
                self.tolerateTimes += 1
        
        self.iters += 1
        
        return signalFlag
    
    # NOTE: setProblem is defined above; keep a single implementation.
    
    def saveResult(self):
        return self.state.toNpzPayload()

    def buildResult(self):
        result = self.state.buildResult()
        if not isinstance(result, OptResult):
            raise TypeError("buildResult() must return OptResult.")
        return result

    def finalize(self):
        result = self.buildResult()
        Verbose.printConclusion(self, result)
        if self.logFlag:
            Verbose.saveLog(self)
        if self.saveFlag:
            if self.session is not None:
                self.storage.saveSnapshot(self.session, self, result, isFinal=True)
                self.storage.close(self.session)
                self.session = None
        return result

    @abc.abstractmethod
    def run(self, problem, seed=None):
        raise NotImplementedError
                    
    def set(self, key, value):
        self.params.set(key, value)

    def get(self, *args):
        return self.params.get(*args)
    
