"""Independent distribution audit. Run from repository with PYTHONPATH=."""
import argparse
import json
import time
import warnings
from pathlib import Path

import numpy as np
from scipy.special import logsumexp
from scipy.stats import truncnorm

from UQPyL.inference import MH, MH_Gibbs, AMH, DEMC, DREAM_ZS
from UQPyL.inference.diagnostics import computeChainDiagnostics
from UQPyL.problem import Problem


class AMHRejectBounds(AMH):
    """Audit-only control: reject raw Gaussian proposals outside the box."""
    def f_prop(self, current, distribution, covariances, ub, lb):
        proposed = np.stack([self.rng.multivariate_normal(x, cov)
                             for x, cov in zip(current, covariances)])
        outside = np.any((proposed < lb) | (proposed > ub), axis=1)
        proposed[outside] = current[outside]
        return proposed


class DreamNoSnooker(DREAM_ZS):
    """Audit-only ablation; other known defects remain active."""
    def __init__(self, **kwargs):
        super().__init__(ps=0., **kwargs)


def makeCase(name):
    if name == "discrete":
        values = np.array([10., 20., 30.])
        probabilities = np.array([.1, .3, .6])
        def energy(x):
            indices = np.searchsorted(values, x[:, 0])
            return (-np.log(probabilities[indices])+.5*x[:, 1]**2)[:, None]
        return Problem(nInput=2, nObj=1, lb=[0., -6.], ub=[1., 6.],
                       varType=[2, 0], varSet={0: values.tolist()}, objFunc=energy,
                       name="InferenceDiscreteAudit"), {"mean": [25., 0.], "variance": [45., 1.],
                                                        "covariance": 0., "category_mass": probabilities.tolist()}
    if name == "normal":
        bounds = 6.0
        energy = lambda x: .5 * np.sum(x*x, axis=1, keepdims=True)
        reference = {"mean": [0., 0.], "variance": [float(truncnorm.var(-6, 6))]*2, "covariance": 0.}
    elif name == "correlated":
        bounds = 6.0
        precision = np.linalg.inv([[1., .9], [.9, 1.]])
        energy = lambda x: .5 * np.einsum("ni,ij,nj->n", x, precision, x)[:, None]
        reference = {"mean": [0., 0.], "variance": [1., 1.], "covariance": .9}
    elif name == "bounded":
        bounds = 1.0
        precision = np.linalg.inv([[1., .9], [.9, 1.]])
        energy = lambda x: .5 * np.einsum("ni,ij,nj->n", x, precision, x)[:, None]
        nodes, weights = np.polynomial.legendre.leggauss(180)
        grid = np.stack(np.meshgrid(nodes, nodes), axis=-1).reshape(-1, 2)
        mass = np.outer(weights, weights).ravel()*np.exp(-energy(grid)[:, 0])
        mass /= mass.sum()
        reference = {"mean": [0., 0.], "variance": (mass @ (grid*grid)).tolist(), "covariance": float(mass @ np.prod(grid, axis=1))}
    elif name == "mixture":
        bounds = 6.0
        def energy(x):
            terms = np.stack([np.log(.25)-.5*((x[:, 0]+2)/.5)**2,
                              np.log(.75)-.5*((x[:, 0]-2)/.5)**2])
            return (-logsumexp(terms, axis=0)+.5*x[:, 1]**2)[:, None]
        reference = {"mean": [1., 0.], "variance": [3.25, 1.], "covariance": 0., "positive_mass": .7499841643790835}
    else:
        raise ValueError(name)
    return Problem(nInput=2, nObj=1, lb=[-bounds]*2, ub=[bounds]*2,
                   objFunc=energy, name="InferenceDistributionAudit"), reference


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", nargs="+", default=["normal", "correlated", "bounded", "mixture"])
    parser.add_argument("--methods", nargs="+", default=["MH", "MH_Gibbs", "AMH", "DEMC", "DREAM_ZS"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[5, 17, 41])
    parser.add_argument("--draws", type=int, default=6000)
    parser.add_argument("--warmup", type=int, default=1000)
    parser.add_argument("--chains", type=int, default=4)
    parser.add_argument("--output", default="agent/verification/1002-inference-distributions.json")
    args = parser.parse_args()
    records = []
    for caseName in args.cases:
        problem, reference = makeCase(caseName)
        for methodName in args.methods:
            for seed in args.seeds:
                started = time.perf_counter()
                method = globals()[methodName](nChains=args.chains, warmUp=args.warmup, maxIters=args.draws,
                                               saveFlag=False, verboseFlag=False)
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter("always")
                    result = method.run(problem, seed=seed)
                # Also inspect the latter half: continuing adaptation can affect early draws.
                samples = result.decs[:, args.draws//2:, :]
                flat = samples.reshape(-1, 2)
                covariance = np.cov(flat.T, ddof=0)
                diagnostic = computeChainDiagnostics(samples)
                record = {"case": caseName, "method": methodName, "seed": seed,
                          "draws": args.draws, "warmup": args.warmup, "reference": reference,
                          "n_chains": args.chains,
                          "mean": flat.mean(axis=0).tolist(), "variance": covariance.diagonal().tolist(),
                          "covariance": float(covariance[0, 1]), "positive_mass": float(np.mean(flat[:, 0]>0)),
                          "acceptance": np.asarray(result.acceptanceRate).tolist(), "diagnostics": diagnostic,
                          "warnings": [str(w.message) for w in caught], "seconds": time.perf_counter()-started}
                record["quantiles"] = np.quantile(flat, [.025, .5, .975], axis=0).tolist()
                if caseName == "discrete":
                    record["category_mass"] = [float(np.mean(flat[:, 0] == value)) for value in [10., 20., 30.]]
                records.append(record)
                Path(args.output).write_text(json.dumps(records, indent=2))
                print(json.dumps({k: record[k] for k in ["case", "method", "seed", "mean", "variance", "covariance", "seconds"]}), flush=True)


if __name__ == "__main__":
    main()
