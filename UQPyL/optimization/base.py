import abc
import time
import numpy as np

from .population import Population
from .core.constraint import calcConstraintViolation
from .runtime import OptResult, Result, SqliteStorage, Verbose
from ..doe import LHS
from ..core import config
from ..core.params import Params
from ..core.runtime_session import RunSession
from ..core.runtime_lifecycle import RunLifecycle
from ..problem.eval import Eval

class AlgorithmABC(RunLifecycle, metaclass = abc.ABCMeta):
    """
    Base class for bounded optimization in unit coordinates.

    Search populations contain unit decisions and minimization-oriented scores.
    Public initial populations contain real decisions and original objectives.
    Runtime decision snapshots are decoded; exported results also restore
    objective directions. Problem bounds are never overwritten.
    """
    def __init__(self, maxFEs: int = None, maxIters: int = None, maxTolerates: int = None, tolerate: float = 1e-6, 
                 verboseFlag: bool = True, verboseFreq: int = 10, logFlag: bool = True, saveFlag: bool = False,
                 saveFreq: int = 100, hvRefPoint = None, historyFreq: int = 10):
        
        self.params = Params()
        self.state = Result(self)
        
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
        self.setHistoryFreq(historyFreq)
        self.hvRefPoint = None if hvRefPoint is None else np.asarray(hvRefPoint, dtype=float).copy()
        self.storage = None
        self.session: RunSession | None = None
        self.runId = None

        if self.hvRefPoint is not None:
            self.set('hvRefPoint', self.hvRefPoint.copy())
    
    def reset(self):
        
        self.FEs = 0; self.iters = 0; self.tolerateTimes = 0
        self.startTime = time.perf_counter()
        
        self.state.reset()
        self.state.extra['history_freq'] = self.historyFreq

    def setHistoryFreq(self, historyFreq):
        if historyFreq is not None:
            if isinstance(historyFreq, (bool, np.bool_)) or not isinstance(historyFreq, (int, np.integer)) or historyFreq <= 0:
                raise ValueError("historyFreq must be a positive integer or None.")
            historyFreq = int(historyFreq)
        self.historyFreq = historyFreq
        self.params.set('historyFreq', historyFreq)
    
    def setup(self, problem, seed):
        
        self.setProblem(problem)
        self._startRun()
        
        self.reset()
        Verbose.setupContext(self, problem)
        if self.saveFlag:
            rootDir = config.resolveWorkDir(getattr(problem, "workDir", None))
            self.storage = SqliteStorage(rootDir)
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
    
    def initPop(self, nInit, initialPop=None):
        if initialPop is None:
            return self._sampleInitialPop(nInit)

        pop = self._coerceInitialPop(initialPop)
        if len(pop) > nInit:
            raise ValueError(
                f"initialPop has {len(pop)} members, but this algorithm expects at most {nInit}."
            )

        if not pop.isEvaluated:
            self.evaluate(pop)

        if len(pop) < nInit:
            pop.add(self._sampleInitialPop(nInit - len(pop)))

        return pop

    def _sampleInitialPop(self, nInit):
        lhs = LHS('classic')

        seed = int(self.rng.integers(0, 1000000))
        xInit = lhs.sample(self.problem, nInit, seed, output="unit")

        pop = Population(xInit)
        self.evaluate(pop)

        return pop

    def _coerceInitialPop(self, initialPop):
        # Public initial populations always contain real decision values.
        pop = initialPop.copy() if isinstance(initialPop, Population) else Population(initialPop)
        if pop.isEvaluated:
            objs, cons = self.problem._validate_common_eval_result(Eval(objs=pop.objs, cons=pop.cons), pop.decs)
            if self.problem.nCon > 0 and cons is None:
                raise ValueError("Evaluated initialPop requires cons when nCon > 0.")
            pop.assignEval(objs, cons)
        pop.setConstraintWeights(self.problem.conWgt)
        pop.decs = self.problem.space_to_unit(pop.decs)
        if pop.isEvaluated:
            pop.objs = pop.objs * self.problem.opt
            if self.problem.nObj > 1:
                self.state.observeMulti(Population(self.problem.unit_to_space(pop.decs),
                                                  pop.objs, pop.cons, pop.conWgt),
                                        self.FEs, self.iters)
        return pop

    def setProblem(self, problem):
        self.problem = problem
        self.searchLb = np.zeros((1, problem.nInput))
        self.searchUb = np.ones((1, problem.nInput))
        problem.canonicalize_unit(self.searchLb)  # Validate bounded conversion before running.
        self.optType = getattr(problem, "optType", None)
        if hasattr(problem, "optType"):
            self.set('optType', problem.optType)
    
    def evaluate(self, pop):
        decs = self.problem.unit_to_space(pop.decs)
        res = self.problem.evaluate(decs)
        objs = res.objs * self.problem.opt
        pop.assignEval(objs, res.cons)
        pop.setConstraintWeights(self.problem.conWgt)
        self.FEs += pop.nPop
        if self.problem.nObj > 1:
            self.state.observeMulti(Population(decs, objs, res.cons, pop.conWgt),
                                    self.FEs, self.iters)
        return pop

    def updateState(self, pop):
        self.state.runtime = time.perf_counter() - self.startTime
        algType = 'EA' if self.problem.nObj == 1 else 'MOEA'
        realPop = pop.copy()
        realPop.decs = self.problem.unit_to_space(pop.decs)
        self.state.update(realPop, self.problem, self.FEs, self.iters, algType)
        return self.state

    def update(self, pop, *, completed=False):
        """Record initialization, or commit one completed iteration."""
        trackStagnation = (completed and self.problem.nObj == 1
                           and self.tolerate is not None and self.state.bestObjs is not None)
        if trackStagnation:
            previousBest = self.state.bestObj
            previousCons = self.state.bestCons
        if completed:
            self.iters += 1
        self.updateState(pop)
        if trackStagnation:
            previousViolation = calcConstraintViolation(previousCons, self.problem.conWgt)
            currentViolation = calcConstraintViolation(self.state.bestCons, self.problem.conWgt)
            previousCV = 0.0 if previousViolation is None else float(previousViolation[0])
            currentCV = 0.0 if currentViolation is None else float(currentViolation[0])
            improved = (currentCV < previousCV or
                        (previousCV == 0 and currentCV == 0
                         and previousBest - self.state.bestObj > self.tolerate))
            self.tolerateTimes = 0 if improved else self.tolerateTimes + 1
        if self.verboseFlag > 0 or self.logFlag > 0 or self.saveFlag > 0:
            Verbose.printIteration(self)
        if self.saveFlag and self.session is not None and self.iters % self.saveFreq == 0:
            self.storage.saveSnapshot(self.session, self, self.state.buildResult(includeHistory=False), isFinal=False)
        return self.state
    
    def checkTermination(self, pop):
        """Check iteration boundaries; initialization and started iterations finish fully.

        maxFEs is a stopping threshold, not a per-evaluation hard cap.
        """
        if self.maxFEs is not None and self.FEs >= self.maxFEs:
            return False
        if self.maxIter is not None and self.iters >= self.maxIter:
            return False
        if (self.problem.nObj == 1 and self.tolerate is not None
                and self.maxTolerates is not None and self.tolerateTimes >= self.maxTolerates):
            return False
        if hasattr(self.problem, 'GUI'):
            self.problem.iterEmit.send()
            if self.problem.isStop:
                return False
        return True
    
    # NOTE: setProblem is defined above; keep a single implementation.
    
    def saveResult(self):
        return self.state.toNpzPayload()

    def exportConfig(self):
        """Export scalar/array settings; live model components require caller restoration."""
        config = dict(self.params.items())
        config.update({name: getattr(self, name) for name in (
            'maxFEs', 'maxTolerates', 'tolerate', 'verboseFlag', 'verboseFreq',
            'logFlag', 'saveFlag', 'saveFreq', 'historyFreq', 'hvRefPoint')})
        config['maxIters'] = self.maxIter
        config['_unrestored_components'] = [name for name in ('surrogate', 'surrogates', 'optimizer') if hasattr(self, name)]
        return {key: value.tolist() if isinstance(value, np.ndarray) else
                value.item() if isinstance(value, np.generic) else value for key, value in config.items()}

    def buildResult(self):
        self.state.runtime = time.perf_counter() - self.startTime
        result = self.state.buildResult()
        if not isinstance(result, OptResult):
            raise TypeError("buildResult() must return OptResult.")
        return result

    def finalize(self):
        self.state.recordFinalSnapshot()
        result = self.buildResult()
        Verbose.printConclusion(self, result)
        if self.logFlag:
            Verbose.saveLog(self)
        if self.saveFlag:
            if self.session is not None:
                self.storage.saveSnapshot(self.session, self, result, isFinal=True)
                self._closeStandaloneSession()
        return result

    @abc.abstractmethod
    def run(self, problem, seed=None, initialPop=None):
        raise NotImplementedError
                    
    def set(self, key, value):
        if key == 'historyFreq':
            self.setHistoryFreq(value)
        else:
            self.params.set(key, value)

    def get(self, *args):
        return self.params.get(*args)
    
