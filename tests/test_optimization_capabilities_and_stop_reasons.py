"""Optimization capabilities and stop reasons.

Migrated from test_review_c16_c18_c19_c21.py; original regression provenance is retained below.
"""

from algorithm_capability_test_support import makeProblem
import json
from types import SimpleNamespace
import numpy as np
import pytest
from UQPyL.optimization.soea import GA, DE, PSO, ABC, CSA, SCE_UA, ML_SCE_UA
from UQPyL.optimization.moea import NSGAII, NSGAIII, MOEAD, RVEA
from UQPyL.optimization.expensive import EGO, ASMO, MOASMO
from UQPyL.optimization.runtime import OptReader
from UQPyL.inference import MH, MH_Gibbs, AMH, DEMC, DREAM_ZS, InfReader
from UQPyL.calibration import ES, IES, GLUE, SUFI2
from UQPyL.problem import Problem
from UQPyL.surrogate.rbf import RBF

QUIET = dict(verboseFlag=False, logFlag=False, saveFlag=False)

SINGLE = [GA, DE, PSO, ABC, CSA, SCE_UA, ML_SCE_UA, EGO, ASMO]

MULTI = [NSGAII, NSGAIII, MOEAD, RVEA, MOASMO]


# Regression source: test_review_c16_c18_c19_c21.py::testUnsupportedObjectiveCountRejectedBeforeEvaluation
@pytest.mark.parametrize("methodClass", SINGLE + MULTI)
def testUnsupportedObjectiveCountRejectedBeforeEvaluation(methodClass):
    calls = []
    method = methodClass(**QUIET)
    with pytest.raises(ValueError, match="objective"):
        method.run(makeProblem(2 if methodClass in SINGLE else 1, calls), seed=1)
    assert calls == []
    capabilities = method.getCapabilities()
    assert capabilities["max_objectives"] == (1 if methodClass in SINGLE else None)
    assert capabilities["min_objectives"] == (1 if methodClass in SINGLE else 2)
    capabilities["variable_types"].clear()
    assert method.getCapabilities()["variable_types"] == ["continuous", "integer", "discrete"]


# Regression source: test_review_c16_c18_c19_c21.py::testWrongInnerOptimizerRejectedBeforeExpensiveEvaluation
@pytest.mark.parametrize("methodClass", [EGO, ASMO, MOASMO])
def testWrongInnerOptimizerRejectedBeforeExpensiveEvaluation(methodClass):
    method = methodClass(**QUIET)
    method.optimizer = GA(**QUIET) if methodClass is MOASMO else NSGAII(**QUIET)
    calls = []
    with pytest.raises(ValueError, match="objective"):
        method.run(makeProblem(2 if methodClass is MOASMO else 1, calls), seed=1)
    assert calls == []
    assert method.getCapabilities()["constraint_handling"] == "evaluation_only"


# Regression source: test_review_c16_c18_c19_c21.py::testEgoRequiresDeclaredPredictiveVarianceBeforeEvaluation
def testEgoRequiresDeclaredPredictiveVarianceBeforeEvaluation():
    method = EGO(**QUIET)
    method.surrogate = RBF()
    calls = []
    with pytest.raises(ValueError, match="variance"):
        method.run(makeProblem(calls=calls), seed=1)
    assert calls == []
    assert EGO.getCapabilities()["requires_predictive_variance"]


# Regression source: test_review_c16_c18_c19_c21.py::testCapabilityDeclarationsDistinguishConstraintSemantics
def testCapabilityDeclarationsDistinguishConstraintSemantics():
    assert GA.getCapabilities()["constraint_handling"] == "feasibility_first"
    assert MH.getCapabilities()["constraint_handling"] == "hard_rejection"
    for cls in [ES, IES]:
        assert cls.getCapabilities()["variable_types"] == ["continuous"]
        assert cls.getCapabilities()["constraint_handling"] == "unsupported"
    for cls in [GLUE, SUFI2]:
        assert cls.getCapabilities()["constraint_handling"] == "not_used"


# Regression source: test_review_c16_c18_c19_c21.py::testOptimizationStopReasonPersistsAndDoesNotLeak
@pytest.mark.parametrize(
    "options,reason",
    [
        ({"maxIters": 0}, "max_iters"),
        ({"maxFEs": 0, "maxIters": 0}, "max_fes"),
        ({"maxIters": 3, "maxTolerates": 0}, "stagnation"),
        ({"maxIters": 5, "maxTolerates": 2}, "stagnation"),
        ({"maxIters": 1}, "max_iters"),
    ],
)
def testOptimizationStopReasonPersistsAndDoesNotLeak(options, reason, tmp_path):
    problem = makeProblem(constant=True)
    problem.workDir = str(tmp_path)
    method = GA(nPop=4, verboseFlag=False, logFlag=True, saveFlag=True, **options)
    result = method.run(problem, seed=1)
    assert result.stopReason == result.summary()["stop_reason"] == result.toDict()["stop_reason"] == reason
    with OptReader(next(tmp_path.rglob("*.sqlite3"))) as reader:
        assert reader.get_run_summary()["stop_reason"] == reader.load_result().stopReason == reason
    assert reason in next(tmp_path.rglob("*.log")).read_text()
    payload = method.saveResult()
    assert json.loads(payload["summaryJson"].item())["stop_reason"] == reason
    method.reset()
    assert method.state.stopReason is None and result.stopReason == reason


# Regression source: test_review_c16_c18_c19_c21.py::testUserStopReason
def testUserStopReason():
    problem = makeProblem()
    problem.GUI = object()
    problem.iterEmit = SimpleNamespace(send=lambda: None)
    problem.isStop = True
    result = GA(nPop=4, maxIters=3, **QUIET).run(problem, seed=1)
    assert result.stopReason == "user_stop" and result.iters == 0


# Regression source: test_review_c16_c18_c19_c21.py::testNoNovelCandidateReason
@pytest.mark.parametrize("methodClass", [EGO, ASMO, MOASMO])
def testNoNovelCandidateReason(methodClass, monkeypatch):
    method = methodClass(nInit=4, maxIters=2, **QUIET)
    method.optimizer = SimpleNamespace(
        run=lambda problem, seed: SimpleNamespace(bestDecs=np.array([[0.2]]), bestObjs=np.ones((1, problem.nObj)))
    )
    monkeypatch.setattr(method, "_fitSurrogate", lambda model, pop: None)
    monkeypatch.setattr(method, "_novelCandidates", lambda *args, **kwargs: np.empty((0, 1)))
    result = method.run(makeProblem(2 if methodClass is MOASMO else 1), seed=1)
    assert result.stopReason == "no_novel_candidates"
    assert result.FEs == 4 and result.iters == 0


# Regression source: test_review_c16_c18_c19_c21.py::testAsmoOneStepReason
def testAsmoOneStepReason():
    method = ASMO(nInit=4, maxIters=5, surrogate=RBF(), optimizer=GA(nPop=4, maxIters=0, **QUIET), **QUIET)
    result = method.run(makeProblem(), oneStep=True, seed=1)
    assert result.stopReason == "one_step" and result.iters == 1
