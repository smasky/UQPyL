"""Start at the exact target, compare batch and conditional chain updates."""
import json
from pathlib import Path

import numpy as np

from UQPyL.inference import DEMC
from UQPyL.problem import Problem


def main():
    records = []
    nReplicas, nChains, nInput = 6000, 4, 2
    initial = np.random.default_rng(2026).normal(size=(nReplicas, nChains, nInput))
    problem = Problem(nInput=nInput, nObj=1, lb=[-100.]*2, ub=[100.]*2,
                      objFunc=lambda x: .5*np.sum(x*x, axis=1, keepdims=True))
    for mode in ["production_batch", "conditional_control"]:
        method = DEMC(nChains=nChains, saveFlag=False, verboseFlag=False)
        method.problem = problem
        method.rng = np.random.default_rng(17)
        samples = initial.copy()
        for step in range(11):
            if step in [0, 1, 5, 10]:
                # Replicas, unlike the chains within each population, are independent.
                perReplica = np.mean(samples*samples, axis=(1, 2))
                records.append({"mode": mode, "step": step, "replicas": nReplicas,
                                "second_moment": float(perReplica.mean()),
                                "independent_replica_se": float(perReplica.std(ddof=1)/np.sqrt(nReplicas)),
                                "target_second_moment": 1.})
                print(json.dumps(records[-1]), flush=True)
            if step == 10:
                break
            for current in samples:
                if mode == "production_batch":
                    proposal = method.f_prop(current, problem.ub, problem.lb)
                    logRatio = -.5*np.sum(proposal*proposal-current*current, axis=1)
                    accepted = np.log(method.rng.random(nChains)) < logRatio
                    current[accepted] = proposal[accepted]
                else:
                    for index in range(nChains):
                        # Use the same production proposal, but condition on already updated chains.
                        proposal = method.f_prop(current, problem.ub, problem.lb)[index]
                        logRatio = -.5*np.sum(proposal*proposal-current[index]*current[index])
                        if np.log(method.rng.random()) < logRatio:
                            current[index] = proposal
        Path("agent/verification/1002-demc-stationarity.json").write_text(json.dumps(records, indent=2))


if __name__ == "__main__":
    main()
