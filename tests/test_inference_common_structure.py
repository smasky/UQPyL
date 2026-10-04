"""Shared entry contracts and sampler-policy exports for all inference methods."""
from copy import deepcopy

import numpy as np
import pytest

from UQPyL.inference import AMH, DEMC, DREAM_ZS, MH, MH_Gibbs, InfReader
from UQPyL.problem import Problem


methodClasses = [MH, MH_Gibbs, AMH, DEMC, DREAM_ZS]
quiet = dict(verboseFlag=False, logFlag=False, saveFlag=False)


def makeProblem(calls):
    def objective(x):
        calls.append(x.copy())
        return np.sum(x*x, axis=1, keepdims=True)
    return Problem(nInput=2, nObj=1, lb=-1., ub=1., objFunc=objective)


@pytest.mark.parametrize("methodClass", methodClasses)
def testCommonInvalidSettingsStopBeforeModelEvaluation(methodClass):
    invalidSettings = [
        {"nChains": True}, {"nChains": 0}, {"nChains": 1.5},
        {"warmUp": -1}, {"warmUp": False}, {"warmUp": .5},
        {"maxIters": 0}, {"maxIters": 2.5}, {"maxInitAttempts": 0},
        {"verboseFreq": 0}, {"saveFreq": 0}, {"logProbFunc": "invalid"},
    ]
    for invalid in invalidSettings:
        calls = []
        options = dict(quiet, nChains=4, warmUp=0, maxIters=4)
        options.update(invalid)
        with pytest.raises(ValueError, match=next(iter(invalid))):
            methodClass(**options).run(makeProblem(calls), seed=17)
        assert calls == []


@pytest.mark.parametrize("methodClass", methodClasses)
def testInvalidGammaStopsBeforeModelEvaluation(methodClass):
    for gamma in [True, "0.2", -.1, np.nan, np.inf, 1j, [.2], [.1, .2, .3],
                  np.zeros((3, 2)), np.zeros((4, 2, 1))]:
        calls = []
        method = methodClass(nChains=4, warmUp=0, maxIters=4, **quiet)
        with pytest.raises(ValueError, match="gamma"):
            method.run(makeProblem(calls), gamma=gamma, seed=17)
        assert calls == []


@pytest.mark.parametrize("methodClass", methodClasses)
def testScalarVectorAndMatrixGammaProduceIdenticalSampling(methodClass):
    reference = None
    for gamma in [.2, np.float64(.2), [.2, .2], np.array([.2, .2]),
                  np.array([[.2, .2]]), np.full((4, 2), .2)]:
        method = methodClass(nChains=4, warmUp=2, maxIters=12, **quiet)
        result = method.run(makeProblem([]), gamma=gamma, seed=17)
        if reference is None:
            reference = result
        for name in ["decs", "objs", "accepted", "logProb"]:
            np.testing.assert_array_equal(getattr(result, name), getattr(reference, name))
        assert result.FEs == reference.FEs
        np.testing.assert_array_equal(result.diagnostics["sampler"]["proposal_settings"]["gamma"], np.full((4, 2), .2))


@pytest.mark.parametrize("methodClass", methodClasses)
def testMutatedParametersAreValidatedOnReuse(methodClass):
    calls = []
    method = methodClass(nChains=4, warmUp=0, maxIters=3, **quiet)
    problem = makeProblem(calls)
    first = method.run(problem, seed=17)
    saved = first.decs.copy()
    method.set("nChains", 2.5)
    calls.clear()
    with pytest.raises(ValueError, match="nChains"):
        method.run(problem, seed=17)
    assert calls == []
    np.testing.assert_array_equal(first.decs, saved)


@pytest.mark.parametrize("methodClass", [MH, MH_Gibbs, AMH])
def testMutatedProposalDistributionIsValidatedBeforeEvaluation(methodClass):
    calls = []
    method = methodClass(nChains=4, warmUp=0, maxIters=3, **quiet)
    method.set("propDist", "invalid")
    with pytest.raises(ValueError, match="propDist"):
        method.run(makeProblem(calls), seed=17)
    assert calls == []


@pytest.mark.parametrize("methodClass,boundary,mode,adaptation", [
    (MH, "reflect", "independent", "none"),
    (MH_Gibbs, "reflect", "coordinate_wise", "none"),
    (AMH, "reject", "independent", "formal_sampling"),
    (DEMC, "reject", "sequential", "none"),
    (DREAM_ZS, "reject", "independent_given_archive", "warmup_only"),
])
def testSamplerPolicyIsConsistentIsolatedAndPersisted(methodClass, boundary, mode, adaptation, tmp_path):
    problem = makeProblem([])
    problem.workDir = str(tmp_path)
    method = methodClass(nChains=4, warmUp=2, maxIters=4, verboseFlag=False, saveFlag=True, saveFreq=1)
    result = method.run(problem, seed=17)
    report = result.diagnostics["sampler"]
    assert {"boundary_policy", "update_mode", "adaptation_phase", "proposal_family", "proposal_settings"} <= report.keys()
    assert (report["boundary_policy"], report["update_mode"], report["adaptation_phase"]) == (boundary, mode, adaptation)
    assert {"gamma", "distribution"} <= report["proposal_settings"].keys()
    if methodClass is DREAM_ZS:
        assert report["archive_policy"] == "warmup_reservoir_frozen"
        assert len(report["crossover_probabilities"]) == method.get("nCR")
    with InfReader(next(tmp_path.rglob("*.sqlite3"))) as reader:
        assert reader.load_result().diagnostics["sampler"] == report
    saved = deepcopy(report)
    method.state.diagnostics["sampler"]["proposal_settings"]["gamma"] = ["changed"]
    assert result.diagnostics["sampler"] == saved
    exported = result.toDict()
    exported["diagnostics"]["sampler"]["proposal_settings"]["gamma"] = ["changed"]
    assert result.diagnostics["sampler"] == saved
