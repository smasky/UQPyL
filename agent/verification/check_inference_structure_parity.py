"""Capture/compare exact traces, RNG continuation and callback order across refactoring."""
import argparse
import json
from pathlib import Path

import numpy as np

from UQPyL.inference import AMH, DEMC, DREAM_ZS, MH, MH_Gibbs
from UQPyL.problem import Problem


def collect():
    records = {}
    configurations = []
    for method in [MH, MH_Gibbs, AMH]:
        for distribution in ["gauss", "uniform"]:
            configurations.append((method, {"propDist": distribution}, .17))
    configurations.extend([(DEMC, {}, None), (DEMC, {}, np.array([.2, .3])),
                           (DREAM_ZS, {"ps": 0.}, None), (DREAM_ZS, {"ps": .1}, .8),
                           (DREAM_ZS, {"ps": 1.}, None)])
    for configIndex, (methodClass, extra, gamma) in enumerate(configurations):
        for case in ["continuous", "constrained", "mixed", "fixed"]:
            for warmUp in [0, 4]:
                for seed in [5, 17]:
                    evaluations, logCalls = [], []
                    def objective(x):
                        evaluations.append(x.tolist())
                        return np.sum(x*x, axis=1, keepdims=True)/10
                    def logProbability(y, decs=None, cons=None):
                        logCalls.append([np.asarray(y).tolist(), np.asarray(decs).tolist(),
                                         None if cons is None else np.asarray(cons).tolist()])
                        return -np.asarray(y).reshape(-1)
                    options = dict(nInput=2, nObj=1, lb=[-2., -2.], ub=[2., 2.], objFunc=objective,
                                   optType="max" if seed == 17 else "min")
                    if case == "constrained":
                        options.update(nCon=1, conFunc=lambda x: x[:, [0]]+.25*x[:, [1]])
                    elif case == "mixed":
                        options.update(lb=[0., 1.], ub=[1., 3.], varType=[2, 1], varSet={0: [1., 2., 4.]})
                    elif case == "fixed":
                        options.update(lb=[-2., 1.], ub=[2., 1.])
                    method = methodClass(nChains=4, warmUp=warmUp, maxIters=40, verboseFlag=False,
                                         saveFlag=False, logFlag=False, logProbFunc=logProbability, **extra)
                    result = method.run(Problem(**options), gamma=gamma, seed=seed)
                    key = f"{configIndex}/{case}/{warmUp}/{seed}"
                    records[key] = {name: None if getattr(result, name) is None else np.asarray(getattr(result, name)).tolist()
                                    for name in ["decs", "objs", "cons", "logProb", "accepted", "acceptanceRate",
                                                 "bestDecs", "bestObjs", "bestCons", "feasibleMask"]}
                    records[key].update(fes=result.FEs, iters=result.iters, stop_reason=result.stopReason,
                                        rng_next=method.rng.random(8).tolist(), evaluations=evaluations, log_calls=logCalls)
    return records


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["capture", "compare"])
    parser.add_argument("--baseline", default="agent/verification/1003-inference-structure-before.json")
    args = parser.parse_args()
    current = collect()
    if args.mode == "capture":
        Path(args.baseline).write_text(json.dumps(current, separators=(",", ":")))
        print(f"Captured {len(current)} configurations.")
    else:
        baseline = json.loads(Path(args.baseline).read_text())
        assert baseline.keys() == current.keys()
        mismatches = [(key, field) for key in baseline for field in baseline[key]
                      if baseline[key][field] != current[key][field]]
        report = {"configurations": len(current), "exact_match": not mismatches, "mismatches": mismatches}
        Path("agent/verification/1003-inference-structure-parity.json").write_text(json.dumps(report, indent=2))
        print(json.dumps(report))
        assert not mismatches


if __name__ == "__main__":
    main()
