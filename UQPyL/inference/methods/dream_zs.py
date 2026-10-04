import numpy as np
import warnings
from typing import Union

from ..base import InferenceABC
from ...problem import ProblemABC


class DREAM_ZS(InferenceABC):
    """
    DREAM-ZS Inference
    Differential evolution MCMC with snooker updates and adaptive crossover weights.
    Archive and crossover/scale adaptation are frozen after warm-up.

    Examples:
        >>> dream = DREAM_ZS(nChains=8, warmUp=500, maxIters=2000)
        >>> res = dream.run(problem, seed=1234)
        >>> print(res.acceptanceRate)

    References:
        [1] J. A. Vrugt, C. J. F. ter Braak, C. G. H. Diks, D. Higdon, B. A. Robinson,
            and J. M. Hyman, Accelerating Markov chain Monte Carlo simulation by
            differential evolution with self-adaptive randomized subspace sampling,
            International Journal of Nonlinear Sciences and Numerical Simulation,
            vol. 10, no. 3, pp. 273-290, 2009.
        [2] J. A. Vrugt, C. J. F. ter Braak, M. P. Clark, J. M. Hyman, and B. A. Robinson,
            Treatment of input uncertainty in hydrologic modeling: Doing hydrology backward
            with Markov chain Monte Carlo simulation, Water Resources Research, vol. 44, no. 12, 2008.
    """

    name = "DREAM-ZS"
    minChains = 3
    boundaryPolicy = "reject"
    updateMode = "independent_given_archive"
    adaptationPhase = "warmup_only"
    proposalFamily = "differential_evolution_snooker"

    def __init__(
        self,
        nChains: int = 10,
        warmUp: int = 1000,
        ps: float = 0.1,
        k: int = 1,
        jitter: float = 0.1,
        adpInterval: int = 50,
        archSize: int = 10,
        acTarget: float = 0.25,
        nCR: int = 5,
        maxIters: int = 1000,
        verboseFlag: bool = True,
        verboseFreq: int = 10,
        logFlag: bool = False,
        saveFlag: bool = True,
        saveFreq: int = 100,
        logProbFunc=None,
        maxInitAttempts: int = 1000,
        snookerRefreshProb: float = 0.1,
    ):
        """
        Initialize the DREAM-ZS inference method.

        Args:
            nChains: Number of sampling chains.
            warmUp: Number of warm-up iterations before formal sampling.
            ps: Probability of snooker update.
            k: Number of differential evolution pairs.
            jitter: Multiplicative proposal jitter scale.
            adpInterval: Warm-up interval for adaptive crossover and scale updates.
            archSize: Warm-up reservoir capacity multiplier relative to chain count; frozen for formal draws.
            acTarget: Target acceptance rate for gamma scaling.
            nCR: Number of crossover rate candidates.
            maxIters: Number of formal sampling draws.
            verboseFlag: Whether to print compact runtime summaries.
            verboseFreq: Iteration interval for terminal and log summaries.
            logFlag: Whether to write a log file.
            saveFlag: Whether to persist snapshots and final result to sqlite.
            saveFreq: Iteration interval for sqlite snapshots.
            logProbFunc: Optional custom log-probability function.
            maxInitAttempts: Maximum LHS batches used to find feasible initial chains with positive probability.
            snookerRefreshProb: Full-dimensional Gaussian refresh probability when ps=1.
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
        self.set("ps", ps)
        self.set("k", k)
        self.set("nCR", nCR)
        self.set("jitter", jitter)
        self.set("archSize", archSize)
        self.set("adpInterval", adpInterval)
        self.set("acTarget", acTarget)
        self.set("snookerRefreshProb", snookerRefreshProb)

    def run(self, problem: ProblemABC, gamma: Union[float, np.ndarray] = None, seed: int = None):
        self.setup(problem, seed)
        nChains, warmUp = self.get("nChains", "warmUp")
        ps, k, jitter = self.get("ps", "k", "jitter")
        refreshProbability = self.get("snookerRefreshProb") if ps == 1 and np.any(problem.ub > problem.lb) else 0.0
        if refreshProbability:
            warnings.warn(
                f"DREAM_ZS: ps=1 uses a {refreshProbability:.0%} full-dimensional Gaussian refresh "
                "to avoid confinement to the archive subspace; this policy is recorded in diagnostics.",
                RuntimeWarning,
                stacklevel=2,
            )
        capacity = nChains * self.get("archSize")
        interval, nCR, target = self.get("adpInterval", "nCR", "acTarget")
        crSet = np.linspace(1 / nCR, 1.0, nCR)
        pCR = np.ones(nCR) / nCR
        gains, tries = np.zeros(nCR), np.zeros(nCR)
        acceptedCounts = np.zeros(nChains)
        if gamma is None:
            gamma = 2.38 / np.sqrt(2 * max(1, np.count_nonzero(problem.ub > problem.lb)))
        gamma = self._check_gamma_(gamma)
        gammaScale = 1.0
        current, currentObjs, currentCons = self.initialSampling(problem, nChains)
        archive = [point.copy() for point in current]
        # Archive points need not be target draws to define a valid fixed kernel.
        # Extra prior points permit k pairs even with fewer live chains.
        while len(archive) < max(3, 2 * k):
            archive.append((problem.lb + self.rng.random(problem.nInput) * (problem.ub - problem.lb)).ravel())
        seen = len(archive)
        span = (problem.ub - problem.lb).astype(float).ravel()

        def advance(chains=None):
            proposed, logRatios, crIndices = self.f_prop_ratio(
                current,
                archive,
                ps,
                k,
                jitter,
                gamma,
                gammaScale,
                crSet,
                pCR,
                tries,
                problem.ub,
                problem.lb,
                logRatio=True,
            )
            objs, cons, valid = self.evaluateProposal(proposed, currentObjs, currentCons)
            # Scale jump gains by the pre-update population variance in unit axes.
            unit = np.divide(current - problem.lb, span, out=np.zeros_like(current), where=span > 0)
            variance = np.maximum(np.var(unit, axis=0, ddof=1), 1e-12)
            for index in range(nChains):
                accepted = valid[index] and self.accept(
                    objs[index],
                    currentObjs[index],
                    cons[index] if problem.nCons else None,
                    decStar=proposed[index],
                    decCur=current[index],
                    consCur=currentCons[index] if problem.nCons else None,
                    logQRatio=logRatios[index],
                )
                if accepted:
                    if chains is None:
                        if crIndices[index] >= 0:
                            jump = np.divide(
                                proposed[index] - current[index], span, out=np.zeros_like(span), where=span > 0
                            )
                            gains[crIndices[index]] += np.sum(jump**2 / variance)
                        acceptedCounts[index] += 1
                    self.updateChainState(
                        index,
                        current,
                        currentObjs,
                        currentCons,
                        proposed[index],
                        objs[index],
                        cons[index] if problem.nCons else None,
                    )
                if chains is not None:
                    self.recordChainState(chains[index], index, current, currentObjs, currentCons, accepted)

        for step in range(warmUp):
            advance()
            # Reservoir sampling retains occupation states, including rejected
            # repeats, without favoring the accepted-jump chain or recent states.
            for point in current:
                seen += 1
                if len(archive) < capacity:
                    archive.append(point.copy())
                else:
                    slot = int(self.rng.integers(seen))
                    if slot < capacity:
                        archive[slot] = point.copy()
            if (step + 1) % interval == 0 or step + 1 == warmUp:
                attempts = step % interval + 1
                pCR, gains, tries, gammaScale = self.adaption(
                    pCR,
                    gains,
                    tries,
                    acceptedCounts / attempts,
                    gammaScale,
                    target,
                )
                acceptedCounts[:] = 0

        # Freeze the archive and all adaptive settings for formal draws. Every
        # chain now uses an independent transition conditional on this archive.
        chains = self.initChains(nChains, current, currentObjs, currentCons)
        self.setSamplerDiagnostics(
            gamma,
            proposalSettings={
                "snooker_probability": ps,
                "pair_count": k,
                "jitter": jitter,
                "full_support_refresh_probability": refreshProbability,
                "refresh_scale": 0.1,
                "effective_snooker_probability": (1 - refreshProbability) * ps,
            },
            archive_policy="warmup_reservoir_frozen",
            archive_size=len(archive),
            gamma_scale=float(gammaScale),
            crossover_probabilities=pCR.tolist(),
        )
        self.update(chains)
        while self.checkTermination(chains):
            advance(chains)
            self.update(chains)
        return self.finalize()

    def adaption(self, pCR, cr_gain, cr_tries, ac_local, gamma_scale, acTarget=None):

        if np.any(cr_tries > 0):
            avg = cr_gain / np.maximum(cr_tries, 1e-6)

            if np.all(avg == 0):
                avg = np.ones(cr_gain.shape[0])

            w = avg / np.sum(avg)

            w = 0.8 * w + 0.2 * (np.ones_like(w)) / w.shape[0]

            pCR = w / np.sum(w)

        cr_gain = cr_gain * 0.0
        cr_tries = cr_tries * 0.0

        if acTarget is not None:
            ac_mean = float(np.mean(ac_local))

            gamma_scale *= np.exp(0.1 * (ac_mean - acTarget))

            gamma_scale = np.clip(gamma_scale, 0.3, 3.0)

        return pCR, cr_gain, cr_tries, gamma_scale

    def validateParameters(self):
        super().validateParameters()
        for key in ["k", "archSize", "adpInterval", "nCR"]:
            self.validateInteger(key, self.get(key), 1)
        for key in ["ps", "acTarget"]:
            value = self.get(key)
            if not np.isfinite(value) or not 0 <= value <= 1:
                raise ValueError(f"DREAM_ZS requires {key} in [0, 1].")
        jitter = self.get("jitter")
        if not np.isfinite(jitter) or jitter < 0:
            raise ValueError("DREAM_ZS requires finite jitter >= 0.")
        if self.get("nChains") * self.get("archSize") < max(3, 2 * self.get("k")):
            raise ValueError("DREAM_ZS archive capacity must be at least max(3, 2*k).")
        refresh = self.get("snookerRefreshProb")
        if not np.isscalar(refresh) or not np.isfinite(refresh) or not 0 < refresh <= 1:
            raise ValueError("snookerRefreshProb must be in (0, 1].")

    def f_prop_ratio(
        self, X_cur, archive, ps, k, jitter, gamma, gamma_scale, crSet, pCR, cr_tries, ub, lb, logRatio=False
    ):
        proposals = np.empty_like(X_cur)
        logRatios = np.zeros(len(X_cur))
        crIndices = np.full(len(X_cur), -1)
        # Do not mix live chains into donors: formal chains are conditionally
        # independent only when every donor is drawn from the frozen archive.
        for index in range(len(X_cur)):
            refresh = self.get("snookerRefreshProb") if ps == 1 and np.any(ub > lb) else 0.0
            if refresh and self.rng.random() < refresh:
                # Symmetric full-support kernel; never project or reflect this move.
                proposals[index] = X_cur[index] + self.rng.normal(size=X_cur.shape[1]) * (0.1 * (ub - lb).ravel())
            elif self.rng.random() < ps:
                proposals[index], logRatios[index] = self.snooker_update(
                    index,
                    X_cur,
                    archive,
                    self.rng.uniform(1.2, 2.2),
                    logRatio=True,
                )
            else:
                crIndex = int(self.rng.choice(len(crSet), p=pCR))
                crIndices[index] = crIndex
                cr_tries[crIndex] += 1
                scale = gamma[index] * gamma_scale * (1 + jitter * self.rng.normal(size=X_cur.shape[1]))
                proposals[index], _ = self.de_prop(index, X_cur, archive, k, crSet[crIndex], scale)
        if logRatio:
            return proposals, logRatios, crIndices
        # Public helper only; the sampling loop uses logs to avoid overflow.
        with np.errstate(over="ignore", under="ignore"):
            ratios = np.exp(logRatios)
        return proposals, ratios, crIndices

    def snooker_update(self, i, X_cur, archive, gamma, logRatio=False):
        current = X_cur[i]
        anchorIndex, leftIndex, rightIndex = self.rng.choice(len(archive), size=3, replace=False)
        anchor = np.asarray(archive[anchorIndex])
        span = (self.problem.ub - self.problem.lb).ravel()
        active = span > 0
        dimension = np.count_nonzero(active)
        axis = np.divide(current - anchor, span, out=np.zeros_like(current), where=active)
        distance = np.hypot.reduce(axis)
        if dimension == 0 or distance == 0:
            return current.copy(), 0.0 if logRatio else 1.0
        direction = axis / distance
        difference = np.divide(
            np.asarray(archive[leftIndex]) - archive[rightIndex], span, out=np.zeros_like(current), where=active
        )
        # A scalar step preserves the anchor-current line in unit coordinates.
        scale = np.asarray(gamma)
        if scale.size != 1:
            raise ValueError("Snooker gamma must be a scalar.")
        displacement = float(scale.item()) * np.dot(difference, direction)
        proposed = current + displacement * direction * span
        proposedDistance = abs(distance + displacement)
        if dimension == 1:
            correction = 0.0
        elif proposedDistance == 0:
            correction = -np.inf
        else:
            correction = (dimension - 1) * (np.log(proposedDistance) - np.log(distance))
        if logRatio:
            return proposed, correction
        with np.errstate(over="ignore", under="ignore"):
            return proposed, float(np.exp(correction))

    def de_prop(self, i, X_cur, archive, k, cr, gamma):
        dimension = X_cur.shape[1]
        selected = self.rng.choice(len(archive), size=2 * k, replace=False)
        points = np.asarray(archive)
        delta = np.sum(points[selected[::2]] - points[selected[1::2]], axis=0)
        span = (self.problem.ub - self.problem.lb).ravel()
        active = span > 0
        mask = (self.rng.random(dimension) < cr) & active
        if not mask.any() and active.any():
            mask[self.rng.choice(np.flatnonzero(active))] = True
        proposed = X_cur[i].copy()
        if mask.any():
            scaling = np.sqrt(np.count_nonzero(active) / (k * np.count_nonzero(mask)))
            noise = self.rng.normal(size=dimension) * (1e-6 * span)
            proposed[mask] += np.asarray(gamma)[mask] * scaling * delta[mask] + noise[mask]
        return proposed, 1.0
