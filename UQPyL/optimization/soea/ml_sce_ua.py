# M&L Shuffled Complex Evolution-UA <Single>

import numpy as np

from typing import Optional

from ..base import AlgorithmABC
from ..population import Population
from ..core.constraint import compareSolutions


class ML_SCE_UA(AlgorithmABC):
    """
    Single-objective M&L shuffled complex evolution algorithm.

    Examples:
        >>> ml_sce = ML_SCE_UA(maxFEs=5000)
        >>> res = ml_sce.run(problem, seed=1234)
        >>> print(res.bestObjs)

    References:
        [1] N. Muttil and S.-Y. Liong, Assessment of the improved shuffled complex evolution algorithm,
            Proceedings of the iEMSs Third Biennial Meeting, 2006.
    """

    name = "ML-SCE-UA"
    alg_type = "EA"

    def __init__(
        self,
        ngs: int = 3,
        npg: Optional[int] = None,
        nps: Optional[int] = None,
        nspl: Optional[int] = None,
        alpha: float = 1.0,
        beta: float = 0.5,
        sita: float = 0.2,
        maxFEs: int = 50000,
        maxIters: int = 1000,
        maxTolerates: int = 1000,
        tolerate: float = 1e-6,
        verboseFlag: bool = True,
        verboseFreq: int = 10,
        logFlag: bool = False,
        saveFlag: bool = True,
        saveFreq: int = 100,
        historyFreq: int = 10,
    ):
        """
        Initialize the algorithm.

        Args:
            ngs: Number of complexes.
            npg: Points per complex; None uses 2 * nInput + 1.
            nps: Points per simplex; None uses nInput + 1.
            nspl: Evolution steps per complex; None uses npg.
            alpha: Reflection coefficient.
            beta: Contraction coefficient.
            sita: Smoothing parameter.
            maxFEs: Maximum number of function evaluations.
            maxIters: Maximum number of iterations.
            maxTolerates: Maximum tolerated non-improving iterations.
            tolerate: Improvement tolerance.
            verboseFlag: Whether to print terminal output.
            verboseFreq: Summary output frequency.
            logFlag: Whether to save full text logs.
            saveFlag: Whether to save sqlite results.
            saveFreq: SQLite snapshot save frequency.
            historyFreq: Full in-memory snapshot interval; None keeps only the final snapshot.
        """

        super().__init__(
            maxFEs=maxFEs,
            maxIters=maxIters,
            maxTolerates=maxTolerates,
            tolerate=tolerate,
            verboseFlag=verboseFlag,
            verboseFreq=verboseFreq,
            logFlag=logFlag,
            saveFlag=saveFlag,
            saveFreq=saveFreq,
            historyFreq=historyFreq,
        )

        # Set algorithm parameters
        self.set("ngs", ngs)
        self.set("npg", npg)
        self.set("nps", nps)
        self.set("nspl", nspl)
        self.set("alpha", alpha)
        self.set("beta", beta)
        self.set("sita", sita)

    def run(self, problem, seed: Optional[int] = None, initialPop=None):
        """
        Run the algorithm on the given problem.

        Args:
            problem: Problem instance.
            seed: Random seed.
            initialPop: Optional initial population or decision matrix.

        Returns:
            OptResult: Final optimization result.
        """
        # setup algorithm
        self.setup(problem, seed)

        # Retrieve parameter values
        ngs, npg, nps, nspl = self.get("ngs", "npg", "nps", "nspl")
        alpha, beta, sita = self.get("alpha", "beta", "sita")

        # Adjust number of complexes if necessary
        if ngs == 0:
            ngs = problem.nInput
            if ngs > 15:
                ngs = 15

        # Initialize SCE parameters
        npg = 2 * problem.nInput + 1 if npg is None else npg
        nps = problem.nInput + 1 if nps is None else nps
        nspl = npg if nspl is None else nspl
        for label, value, minimum in [("ngs", ngs, 1), ("npg", npg, 2), ("nps", nps, 2), ("nspl", nspl, 1)]:
            if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < minimum:
                raise ValueError(f"{label} must be an integer >= {minimum}.")
        if nps > npg:
            raise ValueError("nps must not exceed npg.")
        nInit = npg * ngs

        # Generate initial population
        pop = self.initPop(nInit, initialPop=initialPop)
        self.update(pop)

        # Sort the population in order of increasing function values
        idx = pop.argsort()
        pop = pop[idx]

        # Iterative process
        while self.checkTermination(pop):
            for igs in range(ngs):
                # Partition the population into complexes (sub-populations)
                outerIdx = np.linspace(0, npg - 1, npg, dtype=np.int64) * ngs + igs
                igsPop = pop[outerIdx]

                # Evolve sub-population igs for nspl steps
                for _ in range(nspl):
                    # Select simplex by sampling the complex according to a linear probability distribution
                    p = 2 * (npg + 1 - np.linspace(1, npg, npg)) / ((npg + 1) * npg)
                    innerIdx = self.rng.choice(npg, nps, p=p, replace=False)
                    innerIdx = np.sort(innerIdx)
                    sPop = igsPop[innerIdx]
                    bPop = igsPop[0]

                    # Execute CCE for simplex
                    sNew = self._cce(sPop, bPop, alpha, beta, sita)
                    igsPop.replace(innerIdx[-1], sNew)
                    igsPop = igsPop[igsPop.argsort()]

                # End of inner loop for competitive evolution of simplexes
                pop.replace(outerIdx, igsPop)

            # Sort the population again
            idx = pop.argsort()
            pop = pop[idx]
            self.update(pop, completed=True)

        # Return the final result
        return self.finalize()

    def _cce(self, sPop, bPop, alpha, beta, sita):
        """
        Competitive Complex Evolution (CCE) for a given simplex.

        Args:
            sPop: The current simplex population.
            bPop: The best population member.
            alpha: Reflection coefficient.
            beta: Contraction coefficient.
            sita: Smoothing parameter.

        Returns:
            The new population after CCE.
        """

        N, D = sPop.size()

        sPopDecs = sPop.decs
        bPopDecs = bPop.decs

        sWorst = sPop[-1:]
        sWorstDecs = sWorst.decs
        sWorstObjs = sWorst.objs

        # Calculate the centroid of the simplex
        ce = np.mean(sPopDecs[:-1], axis=0).reshape(1, -1)

        # Reflection step
        sNewDecs = ((sWorstDecs - ce) * alpha * -1 + ce) * (1 - sita) + bPopDecs * sita
        np.clip(sNewDecs, self.searchLb, self.searchUb, out=sNewDecs)

        sNew = Population(sNewDecs)
        self.evaluate(sNew)

        # Contraction step if reflection fails
        if compareSolutions(sNew.objs, sNew.cons, sWorstObjs, sWorst.cons, self.problem.conWgt) >= 0:
            sNewDecs = (sWorstDecs + (ce - sWorstDecs) * beta) * (1 - sita) + bPopDecs * sita
            np.clip(sNewDecs, self.searchLb, self.searchUb, out=sNewDecs)

            sNew = Population(sNewDecs)
            self.evaluate(sNew)

        # Random point if both reflection and contraction fail
        if compareSolutions(sNew.objs, sNew.cons, sWorstObjs, sWorst.cons, self.problem.conWgt) >= 0:
            sNewDecs = self.searchLb + self.rng.random(D) * (self.searchUb - self.searchLb)
            sNew = Population(sNewDecs)
            self.evaluate(sNew)

        # End of CCE
        return sNew
