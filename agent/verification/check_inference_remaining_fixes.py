"""Independent moments and covariance-cost audit; run with PYTHONPATH=."""
import argparse
import json
import time
import warnings
from pathlib import Path

import numpy as np

from UQPyL.inference import AMH, DREAM_ZS, MH
from UQPyL.inference.chain import Chain
from UQPyL.inference.diagnostics import computeChainDiagnostics
from UQPyL.problem import Problem
from check_inference_distributions import makeCase


class FullHistoryAMH(AMH):
    """Previous covariance calculation, retaining the current transition code."""
    def _historyCovariance(self, chain):
        return np.atleast_2d(np.cov(chain.decs[:chain.count].T))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--covariance-only", action="store_true")
    parser.add_argument("--snooker-control", action="store_true")
    parser.add_argument("--output", default="agent/verification/1003-inference-remaining-audit.json")
    args = parser.parse_args()
    records = []
    output = Path(args.output)

    def record(value):
        records.append(value)
        output.write_text(json.dumps(records, indent=2))
        print(value["kind"], value.get("case", value.get("draws")), value.get("method", ""), value.get("seed", ""), flush=True)

    # Same append-only samples for both implementations; timings include all
    # prefix updates, not model calls or complete MCMC runs.
    samples = np.random.default_rng(17).normal(size=(16000, 4))
    problem = Problem(nInput=4, nObj=1, lb=-10., ub=10., objFunc=lambda x: np.zeros((len(x), 1)))
    for count in ([] if args.snooker_control else [2000, 8000, 16000]):
        for methodClass in [FullHistoryAMH, AMH]:
            elapsed = []
            for repeat in range(3):
                method = methodClass(saveFlag=False, verboseFlag=False)
                method.setProblem(problem)
                chain = Chain(4, 1, 0, count)
                chain.decs[:] = samples[:count]
                start = time.perf_counter()
                for prefix in range(3, count+1):
                    chain.count = prefix
                    covariance = method.updateCovs([chain], 1.)[0]
                elapsed.append(time.perf_counter()-start)
            expected = np.cov(samples[:count].T)+np.eye(4)*.4
            record(dict(kind="covariance_cost", draws=count, method=methodClass.__name__,
                        seconds=elapsed, median_seconds=float(np.median(elapsed)),
                        max_abs_error=float(np.max(np.abs(covariance-expected)))))

    if args.covariance_only:
        return

    configs = [(name, cls, seed, 8000, 1000) for name in ["normal", "bounded"]
               for cls in [FullHistoryAMH, AMH] for seed in [5, 17, 41]]
    configs += [("uniform4", DREAM_ZS, seed, 30000, 1000) for seed in [5, 17, 41]]
    configs += [("normal_snooker", DREAM_ZS, seed, 15000, 1000) for seed in [5, 17, 41]]
    configs += [("restricted_support", MH, seed, 8000, 1000) for seed in [5, 17, 41]]
    if args.snooker_control:
        configs = [("uniform4", DREAM_ZS, 41, 120000, 1000)]
    for name, methodClass, seed, draws, warmup in configs:
        kwargs = {}
        if name == "uniform4":
            problem = Problem(nInput=4, nObj=1, lb=-1., ub=1., objFunc=lambda x: np.zeros((len(x), 1)))
            reference = dict(mean=[0.]*4, variance=[1/3]*4)
            kwargs.update(ps=1., archSize=1)
        elif name == "normal_snooker":
            problem, reference = makeCase("normal")
            kwargs.update(ps=1., archSize=1)
        elif name == "restricted_support":
            problem = Problem(nInput=1, nObj=1, lb=-1., ub=1., objFunc=lambda x: np.zeros((len(x), 1)))
            reference = dict(mean=[.7], variance=[.03])
            kwargs["logProbFunc"] = lambda y, decs=None, cons=None: np.where(np.atleast_2d(decs)[:, 0] >= .4, 0., -np.inf)
        else:
            problem, reference = makeCase(name)
        method = methodClass(nChains=3 if methodClass is DREAM_ZS else 4, warmUp=warmup,
                             maxIters=draws, saveFlag=False, verboseFlag=False, logFlag=False, **kwargs)
        start = time.perf_counter()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result = method.run(problem, seed=seed)
        elapsed = time.perf_counter()-start
        flat = result.decs[:, draws//2:].reshape(-1, problem.nInput)
        covariance = np.atleast_2d(np.cov(flat.T, ddof=0))
        record(dict(kind="distribution", case=name, method=methodClass.__name__, seed=seed,
                    draws=draws, warmup=warmup, seconds=elapsed, reference=reference,
                    mean=flat.mean(0).tolist(), variance=flat.var(0).tolist(), covariance=covariance.tolist(),
                    diagnostics=computeChainDiagnostics(result.decs[:, draws//2:]),
                    rank=int(np.linalg.matrix_rank(flat-flat.mean(0))),
                    warnings=[str(item.message) for item in caught]))


if __name__ == "__main__":
    main()
