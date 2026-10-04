from weakref import WeakKeyDictionary

import numpy as np
from typing import Literal, Union

from ..base import InferenceABC
from ...problem import ProblemABC


class AMH(InferenceABC):
    """
    Adaptive Metropolis-Hastings Inference
    Metropolis-Hastings sampling with adaptive proposal covariance updates.

    Examples:
        >>> amh = AMH(nChains=4, warmUp=500, maxIters=2000)
        >>> res = amh.run(problem, gamma=0.1, seed=1234)
        >>> print(res.acceptanceRate)

    References:
        [1] H. Haario, E. Saksman, and J. Tamminen, An adaptive Metropolis algorithm,
            Bernoulli, vol. 7, no. 2, pp. 223-242, 2001.
    """

    name = "AMH"
    boundaryPolicy = "reject"
    adaptationPhase = "formal_sampling"

    def __init__(
        self,
        nChains: int = 1,
        warmUp: int = 1000,
        maxIters: int = 1000,
        propDist: Literal["gauss", "uniform"] = "gauss",
        verboseFlag: bool = True,
        verboseFreq: int = 10,
        logFlag: bool = False,
        saveFlag: bool = True,
        saveFreq: int = 100,
        logProbFunc=None,
        maxInitAttempts: int = 1000,
    ):
        """
        Initialize the adaptive Metropolis-Hastings inference method.

        Args:
            nChains: Number of sampling chains.
            warmUp: Number of warm-up iterations before formal sampling.
            maxIters: Number of formal sampling draws.
            propDist: Proposal distribution type.
            verboseFlag: Whether to print compact runtime summaries.
            verboseFreq: Iteration interval for terminal and log summaries.
            logFlag: Whether to write a log file.
            saveFlag: Whether to persist snapshots and final result to sqlite.
            saveFreq: Iteration interval for sqlite snapshots.
            logProbFunc: Optional custom log-probability function.
            maxInitAttempts: Maximum LHS batches used to find feasible initial chains with positive probability.
        """
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
        self.set("propDist", propDist)

        if propDist not in ["gauss", "uniform"]:
            raise ValueError("propDist must be 'gauss' or 'uniform'")

    def run(self, problem: ProblemABC, gamma: Union[float, np.ndarray] = 0.1, seed: int = None):
        self.setup(problem, seed)

        nChains = self.get("nChains")
        warmUp = self.get("warmUp")
        propDist = self.get("propDist")
        self.set("gamma", gamma)

        gamma = self._check_gamma_(gamma)
        sd = 2.38**2 / problem.nInput

        X_init, Objs_init, Cons_init = self.initialSampling(problem, nChains)

        chains = self.initChains(nChains, X_init, Objs_init, Cons_init)

        propCovs = []

        for i in range(nChains):
            cov = (gamma[i] * (problem.ub - problem.lb)) ** 2
            propCovs.append(np.diag(cov.ravel()))

        X_cur = X_init
        Objs_cur = Objs_init
        Cons_cur = Cons_init

        for _ in range(warmUp):
            X_star = self.f_prop(X_cur, propDist, propCovs, problem.ub, problem.lb)

            Objs_star, Cons_star, admissible = self.evaluateProposal(X_star, Objs_cur, Cons_cur)

            for i in range(nChains):
                if admissible[i] and self.accept(
                    Objs_star[i],
                    Objs_cur[i],
                    Cons_star[i] if problem.nCons > 0 else None,
                    decStar=X_star[i],
                    decCur=X_cur[i],
                    consCur=Cons_cur[i] if problem.nCons > 0 else None,
                ):
                    self.updateChainState(
                        i,
                        X_cur,
                        Objs_cur,
                        Cons_cur,
                        X_star[i],
                        Objs_star[i],
                        Cons_star[i] if problem.nCons > 0 else None,
                    )

        self.setSamplerDiagnostics(gamma)
        chains = self.initChains(nChains, X_cur, Objs_cur, Cons_cur)
        self.update(chains)
        while self.checkTermination(chains):
            X_star = self.f_prop(X_cur, propDist, propCovs, problem.ub, problem.lb)

            Objs_star, Cons_star, admissible = self.evaluateProposal(X_star, Objs_cur, Cons_cur)

            for i, chain in enumerate(chains):
                accepted = admissible[i] and self.accept(
                    Objs_star[i],
                    Objs_cur[i],
                    Cons_star[i] if problem.nCons > 0 else None,
                    decStar=X_star[i],
                    decCur=X_cur[i],
                    consCur=Cons_cur[i] if problem.nCons > 0 else None,
                )
                if accepted:
                    self.updateChainState(
                        i,
                        X_cur,
                        Objs_cur,
                        Cons_cur,
                        X_star[i],
                        Objs_star[i],
                        Cons_star[i] if problem.nCons > 0 else None,
                    )

                self.recordChainState(chain, i, X_cur, Objs_cur, Cons_cur, accepted)

            propCovs = self.updateCovs(chains, sd, propCovs)
            self.update(chains)

        return self.finalize()

    def reset(self):
        super().reset()
        self._covarianceMoments = WeakKeyDictionary()

    def _historyCovariance(self, chain):
        """Incremental unbiased covariance for append-only chain history.

        Center relative to the first point to retain accuracy under large offsets.
        Rejected repeated states count as draws. Cached matrices never alias output.
        """
        if not hasattr(self, "_covarianceMoments"):
            self._covarianceMoments = WeakKeyDictionary()
        cached = self._covarianceMoments.get(chain)
        if cached is None or cached[0] > chain.count:
            origin = chain.decs[0].copy()
            shifted = chain.decs[: chain.count] - origin
            mean = shifted.mean(axis=0)
            centered = shifted - mean
            count, scatter = chain.count, centered.T @ centered
        else:
            count, origin, mean, scatter = cached
            for point in chain.decs[count : chain.count]:
                delta = (point - origin) - mean
                nextCount = count + 1
                scatter += np.outer(delta, delta) * (count / nextCount)
                mean += delta / nextCount
                count = nextCount
        self._covarianceMoments[chain] = (count, origin, mean, scatter)
        return scatter / (count - 1)

    def updateCovs(self, chains, sd, currentCovs=None):

        propCovs = []
        rangeCovariance = np.diag((self.problem.ub - self.problem.lb).ravel() ** 2)
        # A dimensionless floor retains its meaning when input units change.
        floor = 1e-3 * rangeCovariance

        for i, chain in enumerate(chains):
            if chain.count < 3:
                if hasattr(self, "_covarianceMoments"):
                    self._covarianceMoments.pop(chain, None)
                if currentCovs is not None:
                    propCovs.append(currentCovs[i].copy())
                else:
                    propCovs.append(rangeCovariance * sd)
                continue

            cov = self._historyCovariance(chain) * sd
            propCovs.append(cov + floor)

        return propCovs

    def f_prop(self, X_cur, propDist, propCovs, ub, lb):

        X_star = np.zeros(X_cur.shape)

        for i in range(X_cur.shape[0]):
            if propDist == "gauss":
                X_star[i] = self.rng.multivariate_normal(X_cur[i].ravel(), propCovs[i])
            elif propDist == "uniform":
                halfWidth = np.sqrt(propCovs[i].diagonal())
                X_star[i] = self.rng.uniform(X_cur[i] - halfWidth, X_cur[i] + halfWidth)
            else:
                raise ValueError("propDist must be 'gauss' or 'uniform'")

        # Keep the raw Gaussian direction; run() rejects proposals outside the
        # box. Coordinate reflection would require a nontrivial Hastings ratio.
        return X_star
