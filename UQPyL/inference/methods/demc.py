import numpy as np
from typing import Union

from ..base import InferenceABC
from ...problem import ProblemABC


class DEMC(InferenceABC):
    """
    Differential Evolution Markov Chain Inference
    Multi-chain differential evolution sampling for posterior exploration.

    Examples:
        >>> demc = DEMC(nChains=6, warmUp=500, maxIters=2000)
        >>> res = demc.run(problem, seed=1234)
        >>> print(res.bestObjs)

    References:
        [1] C. J. F. ter Braak, A Markov Chain Monte Carlo version of the genetic algorithm
            Differential Evolution: easy Bayesian computing for real parameter spaces,
            Statistics and Computing, vol. 16, no. 3, pp. 239-249, 2006.
    """

    name = "DEMC"
    minChains = 3
    boundaryPolicy = "reject"
    updateMode = "sequential"
    proposalFamily = "differential_evolution"

    def __init__(
        self,
        nChains: int = 3,
        warmUp: int = 1000,
        maxIters: int = 1000,
        verboseFlag: bool = True,
        verboseFreq: int = 10,
        logFlag: bool = False,
        saveFlag: bool = True,
        saveFreq: int = 100,
        logProbFunc=None,
        maxInitAttempts: int = 1000,
    ):
        """
        Initialize the DEMC inference method.

        Args:
            nChains: Number of sampling chains.
            warmUp: Number of warm-up iterations before formal sampling.
            maxIters: Number of formal sampling draws.
            verboseFlag: Whether to print compact runtime summaries.
            verboseFreq: Iteration interval for terminal and log summaries.
            logFlag: Whether to write a log file.
            saveFlag: Whether to persist snapshots and final result to sqlite.
            saveFreq: Iteration interval for sqlite snapshots.
            logProbFunc: Optional custom log-probability function.
            maxInitAttempts: Maximum LHS batches used to find feasible initial chains with positive probability.
        """

        self.validateInteger("nChains", nChains, self.minChains)
        super().__init__(
            maxIters,
            verboseFlag,
            verboseFreq,
            logFlag,
            saveFlag,
            saveFreq,
            logProbFunc,
            maxInitAttempts,
        )

        self.set("nChains", nChains)
        self.set("warmUp", warmUp)

    def run(self, problem: ProblemABC, gamma: Union[float, np.ndarray] = None, seed: int = None):
        self.setup(problem, seed)
        nChains, warmUp = self.get("nChains", "warmUp")
        if gamma is not None:
            gamma = self._check_gamma_(gamma)
        current, currentObjs, currentCons = self.initialSampling(problem, nChains)

        def advance(chains=None):
            # Metropolis-within-Gibbs: subsequent chains see accepted updates.
            for index in range(nChains):
                proposed = self.f_prop(current, problem.ub, problem.lb, gamma, chainIndex=index)
                consRow = None if currentCons is None else currentCons[index : index + 1]
                objs, cons, valid = self.evaluateProposal(proposed, currentObjs[index : index + 1], consRow)
                accepted = valid[0] and self.accept(
                    objs[0],
                    currentObjs[index],
                    cons[0] if problem.nCons else None,
                    decStar=proposed[0],
                    decCur=current[index],
                    consCur=currentCons[index] if problem.nCons else None,
                )
                if accepted:
                    self.updateChainState(
                        index,
                        current,
                        currentObjs,
                        currentCons,
                        proposed[0],
                        objs[0],
                        cons[0] if problem.nCons else None,
                    )
                if chains is not None:
                    self.recordChainState(chains[index], index, current, currentObjs, currentCons, accepted)

        for _ in range(warmUp):
            advance()
        chains = self.initChains(nChains, current, currentObjs, currentCons)
        self.setSamplerDiagnostics(
            gamma,
            proposalSettings={
                "unit_jump_probability": 0.1 if gamma is None else 0.0,
                "noise_scale": 1e-6,
            },
        )
        self.update(chains)
        while self.checkTermination(chains):
            advance(chains)
            self.update(chains)
        return self.finalize()

    def f_prop(self, X_cur, ub, lb, gamma=None, chainIndex=None):
        nChains = X_cur.shape[0]
        active = max(1, np.count_nonzero(ub > lb))
        defaultGamma = gamma is None
        if gamma is None:
            gamma = np.full((nChains, self.problem.nInput), 2.38 / np.sqrt(2 * active))
        indices = range(nChains) if chainIndex is None else [chainIndex]
        proposals = []
        span = (ub - lb).ravel()
        for index in indices:
            pool = [other for other in range(nChains) if other != index]
            left, right = self.rng.choice(pool, 2, replace=False)
            # Occasional unit-scale DE jumps help move between modes and let
            # isolated chains rejoin the population (ter Braak & Vrugt, 2008).
            scale = 1.0 if defaultGamma and self.rng.random() < 0.1 else gamma[index]
            # Symmetric, independent perturbations preserve full-dimensional
            # support even when the population lies in an affine subspace.
            noise = self.rng.normal(size=self.problem.nInput) * (1e-6 * span)
            proposals.append(X_cur[index] + scale * (X_cur[left] - X_cur[right]) + noise)
        return np.asarray(proposals)
