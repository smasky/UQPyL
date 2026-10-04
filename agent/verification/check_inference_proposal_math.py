"""Deterministic proposal checks, independent of convergence thresholds."""
import json
from itertools import product
from pathlib import Path

import numpy as np
from scipy.stats import multivariate_normal

from UQPyL.inference import DEMC, DREAM_ZS
from UQPyL.problem import Problem


class FixedRng:
    def integers(self, *args, **kwargs):
        return 0

    def choice(self, *args, **kwargs):
        return np.array([1, 2])

    def random(self, *args, **kwargs):
        return 0.


def reflectedDensity(source, target, covariance, extent):
    # All preimages of componentwise reflection onto [0, 1]^2.
    preimages = np.array([2*np.array(shift)+np.array(sign)*target
                         for shift in product(range(-extent, extent+1), repeat=2)
                         for sign in product([-1, 1], repeat=2)])
    return float(multivariate_normal.pdf(preimages, mean=source, cov=covariance).sum())


def main():
    records = {}
    source, target = np.array([.05, .4]), np.array([.3, .6])
    for label, covariance in [("correlated", [[.09, .072], [.072, .09]]),
                              ("diagonal_control", [[.09, 0.], [0., .09]])]:
        records["reflection_"+label] = [
            {"extent": extent, "forward": reflectedDensity(source, target, covariance, extent),
             "reverse": reflectedDensity(target, source, covariance, extent)} for extent in [3, 6]]

    dream = DREAM_ZS(verboseFlag=False, saveFlag=False)
    dream.rng = FixedRng()
    current = np.array([[0., 0.], [2., 1.], [-1., .5]])
    anchor = np.array([1., 2.])
    proposal, ratio = dream.snooker_update(0, current, [anchor], np.array([1.7, 1.7]))
    axis = current[0]-anchor
    difference = current[1]-current[2]
    expectedProposal = current[0]+1.7*np.dot(difference, axis)/np.dot(axis, axis)*axis
    expectedRatio = np.linalg.norm(expectedProposal-anchor)/np.linalg.norm(current[0]-anchor)
    records["snooker"] = {"actual_proposal": proposal.tolist(), "reference_proposal": expectedProposal.tolist(),
                          "actual_ratio": float(ratio), "reference_ratio": float(expectedRatio),
                          "off_axis_cross_product": float(proposal[0]*axis[1]-proposal[1]*axis[0])}
    currentOne = np.array([[0.], [2.], [-1.]])
    proposal, ratio = dream.snooker_update(0, currentOne, [np.array([.5])], np.array([1.7]))
    records["snooker_one_dimension"] = {"actual_ratio": float(ratio), "reference_ratio": 1.}

    # Rank-deficient populations must be able to gain transverse random motion.
    demc = DEMC(verboseFlag=False, saveFlag=False)
    demc.problem = Problem(nInput=2, nObj=1, lb=[-100.]*2, ub=[100.]*2,
                           objFunc=lambda x: np.sum(x*x, axis=1, keepdims=True))
    demc.rng = np.random.default_rng(17)
    collinear = np.array([[-1., -1.], [0., 0.], [1., 1.]])
    proposals = np.stack([demc.f_prop(collinear, demc.problem.ub, demc.problem.lb) for _ in range(100)])
    records["demc_transverse_motion"] = {"maximum_axis_difference": float(np.max(np.abs(proposals[..., 0]-proposals[..., 1])))}

    # Adaptation helper receives the statistic accumulated by the run loop.
    warmup, interval, accepted = 1000, 50, 25
    _, _, _, scale = dream.adaption(np.ones(2)/2, np.zeros(2), np.ones(2),
                                    np.full(4, accepted/warmup), 1., .25)
    records["dream_acceptance_adaptation"] = {"true_interval_rate": accepted/interval,
                                              "actual_used_rate": accepted/warmup,
                                              "actual_scale": float(scale),
                                              "reference_scale": float(np.exp(.1*(accepted/interval-.25)))}
    records["dream_jump_score"] = {"jump": [1., -1.], "actual": float(np.sum([1., -1.])**2),
                                    "squared_euclidean": 2.}
    Path("agent/verification/1002-inference-proposal-math.json").write_text(json.dumps(records, indent=2))
    print(json.dumps(records, indent=2))


if __name__ == "__main__":
    main()
